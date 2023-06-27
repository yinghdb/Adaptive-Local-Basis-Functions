import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR, StepLR
import torch.distributed as dist
from pointnet2_ops import pointnet2_utils

from src.utils.data import drop_anchors, m_dist
from src.utils.ops import furthest_point_sample

from src.net.local_encoder import Local_Encoder
from src.net.transformer_global import Transformer_Global
from src.net.transformer_local import Transformer_Local
from src.net.head_nets import ShapeCodeHead, GlobalPointsHead, SigmaHead, RotateHead, OffsetHead
from src.net.dsdf_decoder import Decoder

class MisModule(pl.LightningModule):

    def __init__(self, config, encode_ckpt=None, pretrained_ckpt=None):
        super().__init__()

        self.config = config

        # init model
        embed_dim = config.local_encoder.embed_dim
        self.transformer_local = Transformer_Local(self.config.transformer_local, embed_dim)
        self.local_encoder = Local_Encoder(self.config.local_encoder, 3)
        self.transformer_global = Transformer_Global(config.transformer_global, embed_dim)
        self.globalpoints_head = GlobalPointsHead(embed_dim, config.transformer_global.num_query_per_obj)
        self.local_decoder = Decoder(config.local_decoder, embed_dim)
        self.d_sigma_head = SigmaHead(embed_dim)
        self.d_rotate_head = RotateHead(embed_dim)
        self.d_offset_head = OffsetHead(embed_dim)
        self.d_shapecode_head = ShapeCodeHead(embed_dim, embed_dim)
        self.e_sigma_head = SigmaHead(embed_dim)
        self.e_rotate_head = RotateHead(embed_dim)

        # load checkpoint
        if encode_ckpt:
            state_dict = torch.load(encode_ckpt, map_location='cpu')['state_dict']

            self.local_encoder.load_state_dict(state_dict['local_encoder'])
            self.transformer_global.load_state_dict(state_dict['transformer_global'])
            self.globalpoints_head.load_state_dict(state_dict['globalpoints_head'])
            self.local_decoder.load_state_dict(state_dict['local_decoder'])

            self.e_sigma_head.load_state_dict(state_dict['sigma_head'])
            self.e_rotate_head.load_state_dict(state_dict['rotate_head'])
            self.d_sigma_head.load_state_dict(state_dict['sigma_head'])
            self.d_rotate_head.load_state_dict(state_dict['rotate_head'])
            self.d_shapecode_head.load_state_dict(state_dict['shapecode_head'])

            print(f"Load \'{encode_ckpt}\' as encode checkpoint")

        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']

            self.transformer_local.load_state_dict(state_dict['transformer_local']) 
            self.local_encoder.load_state_dict(state_dict['local_encoder']) 
            self.transformer_global.load_state_dict(state_dict['transformer_global']) 
            self.globalpoints_head.load_state_dict(state_dict['globalpoints_head']) 
            self.local_decoder.load_state_dict(state_dict['local_decoder']) 
            self.d_sigma_head.load_state_dict(state_dict['d_sigma_head']) 
            self.d_rotate_head.load_state_dict(state_dict['d_rotate_head'])
            self.d_offset_head.load_state_dict(state_dict['d_offset_head'])
            self.d_shapecode_head.load_state_dict(state_dict['d_shapecode_head']) 
            self.e_sigma_head.load_state_dict(state_dict['e_sigma_head']) 
            self.e_rotate_head.load_state_dict(state_dict['e_rotate_head']) 

            print(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = {
            "transformer_local": self.transformer_local.state_dict(),
            "local_encoder": self.local_encoder.state_dict(),
            "transformer_global": self.transformer_global.state_dict(),
            "globalpoints_head": self.globalpoints_head.state_dict(),
            "local_decoder": self.local_decoder.state_dict(),

            "d_sigma_head": self.d_sigma_head.state_dict(),
            "d_rotate_head": self.d_rotate_head.state_dict(),
            "d_offset_head": self.d_offset_head.state_dict(),
            "d_shapecode_head": self.d_shapecode_head.state_dict(),
            "e_sigma_head": self.e_sigma_head.state_dict(),
            "e_rotate_head": self.e_rotate_head.state_dict(),
        }

    def configure_optimizers(self):
        optim_config = self.config.optimizer
        optim_type = optim_config.type
        lr_rate = optim_config.lr_rate
        param_list = [
            dict(params=self.transformer_local.parameters(), lr=optim_config.lr),
            dict(params=self.d_shapecode_head.parameters(), lr=optim_config.lr),
            dict(params=self.d_sigma_head.parameters(), lr=optim_config.lr),
            dict(params=self.d_rotate_head.parameters(), lr=optim_config.lr),
            dict(params=self.d_offset_head.parameters(), lr=optim_config.lr),
            dict(params=self.local_decoder.parameters(), lr=optim_config.lr),
        ]
        if optim_type == "sgd":
            optimizer = torch.optim.SGD(param_list, lr=optim_config.lr, weight_decay=optim_config.weight_decay)
        elif optim_type == "adam":
            optimizer = torch.optim.Adam(param_list, lr=optim_config.lr, weight_decay=optim_config.weight_decay)
        elif optim_type == "adamw":
            optimizer = torch.optim.AdamW(param_list, lr=optim_config.lr, weight_decay=optim_config.weight_decay)
        else:
            raise ValueError(f"OPTIMIZER = {optim_type} is not a valid optimizer!")

        sche_config = self.config.scheduler
        scheduler = {'interval': sche_config.interval,
                     'name': "AutoEncoder LR"}
        sched_type = sche_config.type

        if sched_type == 'MultiStepLR':
            scheduler.update(
                {'scheduler': MultiStepLR(optimizer, sche_config.mslr_milestones, gamma=sche_config.gamma)})
        elif sched_type == 'CosineAnnealing':
            scheduler.update(
                {'scheduler': CosineAnnealingLR(optimizer, sche_config.cosa_tmax)})
        elif sched_type == 'ExponentialLR':
            scheduler.update(
                {'scheduler': ExponentialLR(optimizer, sche_config.gamma)})
        elif sched_type == "StepLR":
            scheduler.update(
                {'scheduler': StepLR(optimizer, sche_config.step_size, sche_config.gamma)})
        else:
            raise NotImplementedError()

        return [optimizer], [scheduler]

    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_idx,
        optimizer_closure, on_tpu, using_native_amp, using_lbfgs):

        # learning rate warm up
        warmup_step = self.config.scheduler.warmup_step
        if self.trainer.global_step < warmup_step:
            lr = 0.01 * self.config.optimizer.lr
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def _fetch_global_samples(self, nonface_sdfs):
        r"""
        input:
        nonface_sdfs: (B, M, 4)  
        
        output:
        nonface_points_sampled: (B, M, 3) 
        nonface_sdfs_sampled: (B, M, 1) 
        """

        nonface_points_sampled = nonface_sdfs[:, :, :3].contiguous()
        nonface_sdfs_sampled = nonface_sdfs[:, :, 3:].contiguous()

        return nonface_points_sampled, nonface_sdfs_sampled

    def _to_knn_training_samples(self, ref_pos, ref_code, points, sdfs):
        r"""
        input:
        ref_pos: (B, N, k, 3)
        ref_code: (B, N, k, d)
        points: (B, N, 3)
        sdfs: (B, N, 1)
        
        output:
        input_data: (B*N, k, d+3) 
        target_data: (B*N, 1) 
        """

        # to local coord
        points_local = points[:, :, None, :] - ref_pos
        points_local = torch.flatten(points_local, start_dim=0, end_dim=1)

        # prepare anchor code
        ref_code = torch.flatten(ref_code, start_dim=0, end_dim=1)

        input_data = torch.cat([ref_code, points_local], dim=-1)

        target_data = torch.flatten(sdfs, start_dim=0, end_dim=1)

        return input_data, target_data

    def on_train_epoch_start(self):
        self.local_encoder.eval()
        self.transformer_global.eval()
        self.globalpoints_head.eval()
        self.e_sigma_head.eval()
        self.e_rotate_head.eval()

    def training_step(self, batch, batch_idx):
        ######################### Visible Anchors ######################
        with torch.no_grad():
            # get input infos
            input_points = batch["input_points"].transpose(1,2).contiguous()

            # encode local point cloud
            anchor_pos_vis, anchor_embed_vis = self.local_encoder(input_points, None, trans_type="coord_only")
            anchor_pos_vis = anchor_pos_vis.transpose(1, 2).contiguous()
            anchor_embed_vis = anchor_embed_vis.transpose(1, 2).contiguous()
            # anchor shape codes
            anchor_sigma_vis = self.e_sigma_head(anchor_embed_vis)
            anchor_rotate_vis = self.e_rotate_head(anchor_embed_vis)
            
            # drop anchors
            to_drop_feas_list = [
                anchor_pos_vis, anchor_embed_vis
            ]
            fea_sizes = [item.shape[-1] for item in to_drop_feas_list]
            to_drop_feas = torch.cat(to_drop_feas_list, dim=-1)
            dropped_feas = drop_anchors(anchor_pos_vis, anchor_sigma_vis, anchor_rotate_vis, self.config.local_encoder.drop_count, to_drop_feas)
            (anchor_pos_vis, anchor_embed_vis) = torch.split(dropped_feas, fea_sizes, dim=-1)
                
            # fetch global features
            mid_embed = self.transformer_global(anchor_pos_vis, anchor_embed_vis.detach())
            pred_mis_points = self.globalpoints_head(mid_embed)

        # global shape code
        anchor_embed_g = self.transformer_local(anchor_pos_vis, anchor_embed_vis, pred_mis_points)

        anchor_pos_g = torch.cat([anchor_pos_vis, pred_mis_points], dim=1)
        anchor_code_g = self.d_shapecode_head(anchor_embed_g)
        anchor_sigma_g = self.d_sigma_head(anchor_embed_g)
        anchor_rotate_g = self.d_rotate_head(anchor_embed_g)
        anchor_offset_g = self.d_offset_head(anchor_embed_g)
        anchor_pos_g = anchor_pos_g + anchor_offset_g

        ######################### Training Targets for Global Anchors ######################
        ## prepare for knn query loss
        knn_points, knn_sdfs = self._fetch_global_samples(batch['nonface_sdfs'])
        
        # knn query in RBF dist
        num_k = self.config.sdf.global_query_k
        mdist = m_dist(anchor_pos_g, anchor_sigma_g, anchor_rotate_g, knn_points) # B, N, ng
        mdist_top, mdist_ids_top = torch.topk(mdist, num_k, dim=-1, largest=True, sorted=True) # B, N, 2

        _B, _N, _k = mdist_top.shape
        gather_ids = torch.flatten(mdist_ids_top, start_dim=1, end_dim=2)
        batch_ids = torch.arange(0, _B, device=gather_ids.device).view(-1, 1).repeat(1, _N*_k).contiguous()
        gather_ids = gather_ids.view(-1)
        batch_ids = batch_ids.view(-1)

        ref_pos = anchor_pos_g[batch_ids, gather_ids].view(_B, _N, _k, -1)
        ref_code = anchor_code_g[batch_ids, gather_ids].view(_B, _N, _k, -1)
        input_g_knn, target_g_knn = self._to_knn_training_samples(ref_pos, ref_code, knn_points, knn_sdfs)
        mdist_top = mdist_top.view(_B*_N, _k, 1) + 1e-12
        combine_weight = mdist_top / mdist_top.sum(dim=1, keepdim=True)

        # knn query in Euclidean dist
        with torch.no_grad():
            gdist = (anchor_pos_g[:, None, :, :] - knn_points[:, :, None, :]).pow(2).sum(dim=-1) # B, N, ng
            gdist_top, gdist_ids_top = gdist.min(dim=-1)  # B, N

            _B, _N = gdist_top.shape
            gather_ids = gdist_ids_top
            batch_ids = torch.arange(0, _B, device=gather_ids.device).view(-1, 1).repeat(1, _N).contiguous()
            gather_ids = gather_ids.view(-1)
            batch_ids = batch_ids.view(-1)

        ref_pos = anchor_pos_g[batch_ids, gather_ids].view(_B, _N, -1)
        ref_code = anchor_code_g[batch_ids, gather_ids].view(_B, _N, -1)
        # to local coord
        points_local = knn_points - ref_pos
        points_local = torch.flatten(points_local, start_dim=0, end_dim=1)
        # prepare anchor code
        ref_code = torch.flatten(ref_code, start_dim=0, end_dim=1)
        input_g_knn_geo = torch.cat([ref_code, points_local], dim=-1)

        ######################### Forward ######################
        pred_global_knn = self.local_decoder(input_g_knn)
        pred_global_knn_geo = self.local_decoder(input_g_knn_geo)

        ######################### Loss ######################
        clamp_dist = self.config.sdf.clamp_dist
        target_g_knn = torch.clamp(target_g_knn, -clamp_dist, clamp_dist)
        pred_global_knn = torch.clamp(pred_global_knn, -clamp_dist, clamp_dist)
        pred_global_knn_geo = torch.clamp(pred_global_knn_geo, -clamp_dist, clamp_dist)

        # sdf loss
        sdf_loss_global = (combine_weight[:, 0, :] * (pred_global_knn[:, 0, :] - target_g_knn).abs()).mean() + \
            (combine_weight[:, 1, :] * (pred_global_knn[:, 1, :] - target_g_knn).abs()).mean()
        sdf_loss_global_geo = (pred_global_knn_geo - target_g_knn).abs().mean()

        # smooth loss
        if self.current_epoch > -1:
            smooth_loss = ((pred_global_knn[:, 0, :] - pred_global_knn[:, 1, :]).abs()).mean()
        else:
            smooth_loss = torch.zeros_like(sdf_loss_global)

        # offset reg loss
        offreg_loss = anchor_offset_g.abs().mean()
        if self.current_epoch >= 1:
            self.config.loss.offreg_lambda = 0.0

        loss =  sdf_loss_global*self.config.loss.sdf_global_lambda + \
                sdf_loss_global_geo*self.config.loss.sdf_global_geo_lambda + \
                smooth_loss*self.config.loss.smooth_lambda + \
                offreg_loss*self.config.loss.offreg_lambda

        loss_dict = {
            'loss': loss,
            'sdf_loss_global': sdf_loss_global.detach(),
            'sdf_loss_global_geo': sdf_loss_global_geo.detach(),
            'smooth_loss': smooth_loss.detach(),
            'offreg_loss': offreg_loss.detach()
        }
        self.log_dict(loss_dict, prog_bar=True, logger=False, sync_dist=True)

        return loss_dict

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_sdf_loss_global = torch.stack([x['sdf_loss_global'] for x in outputs]).mean()
        avg_smooth_loss = torch.stack([x['smooth_loss'] for x in outputs]).mean()
        avg_offreg_loss = torch.stack([x['offreg_loss'] for x in outputs]).mean()
        avg_sdf_loss_global_geo = torch.stack([x['sdf_loss_global_geo'] for x in outputs]).mean()

        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar('train/avg_loss', avg_loss,
                global_step=self.current_epoch)
            self.logger.experiment.add_scalar('train/avg_sdf_loss_global', avg_sdf_loss_global,
                global_step=self.current_epoch)
            self.logger.experiment.add_scalar('train/avg_sdf_loss_global_geo', avg_sdf_loss_global_geo,
                global_step=self.current_epoch)
            self.logger.experiment.add_scalar('train/avg_smooth_loss', avg_smooth_loss,
                global_step=self.current_epoch)
            self.logger.experiment.add_scalar('train/avg_offreg_loss', avg_offreg_loss,
                global_step=self.current_epoch)

            self.print('========= Epoch {} Training =========='.format(self.current_epoch))
            self.print('total loss: {:.8f}; sdf loss: {:.8f}; sdf geo loss: {:.8f}; smooth loss: {:.8f}; offreg loss: {:.8f}'\
                .format(avg_loss, avg_sdf_loss_global, avg_sdf_loss_global_geo, avg_smooth_loss, avg_offreg_loss))

    def validation_step(self, batch, batch_idx):
        ######################### Visible Anchors ######################
        # get input infos
        input_points = batch["input_points"].transpose(1,2).contiguous()

        # encode local point cloud
        anchor_pos_vis, anchor_embed_vis = self.local_encoder(input_points, None, trans_type="coord_only")
        anchor_pos_vis = anchor_pos_vis.transpose(1, 2).contiguous()
        anchor_embed_vis = anchor_embed_vis.transpose(1, 2).contiguous()
        # anchor shape codes
        anchor_sigma_vis = self.e_sigma_head(anchor_embed_vis)
        anchor_rotate_vis = self.e_rotate_head(anchor_embed_vis)
        
        # drop anchors
        to_drop_feas_list = [
            anchor_pos_vis, anchor_embed_vis
        ]
        fea_sizes = [item.shape[-1] for item in to_drop_feas_list]
        to_drop_feas = torch.cat(to_drop_feas_list, dim=-1)
        dropped_feas = drop_anchors(anchor_pos_vis, anchor_sigma_vis, anchor_rotate_vis, self.config.local_encoder.drop_count, to_drop_feas)
        (anchor_pos_vis, anchor_embed_vis) = torch.split(dropped_feas, fea_sizes, dim=-1)
            
        # fetch global features
        mid_embed = self.transformer_global(anchor_pos_vis, anchor_embed_vis.detach())
        pred_mis_points = self.globalpoints_head(mid_embed)

        # global shape code
        anchor_embed_g = self.transformer_local(anchor_pos_vis, anchor_embed_vis, pred_mis_points)

        anchor_pos_g = torch.cat([anchor_pos_vis, pred_mis_points], dim=1)
        anchor_code_g = self.d_shapecode_head(anchor_embed_g)
        anchor_sigma_g = self.d_sigma_head(anchor_embed_g)
        anchor_rotate_g = self.d_rotate_head(anchor_embed_g)
        anchor_offset_g = self.d_offset_head(anchor_embed_g)
        anchor_pos_g = anchor_pos_g + anchor_offset_g
        
        ######################### Training Targets for Global Anchors ######################
        ## prepare for knn query loss
        knn_points, knn_sdfs = self._fetch_global_samples(batch['nonface_sdfs'])
        
        # knn query in RBF dist
        num_k = self.config.sdf.global_query_k
        mdist = m_dist(anchor_pos_g, anchor_sigma_g, anchor_rotate_g, knn_points) # B, N, ng
        mdist_top, mdist_ids_top = torch.topk(mdist, num_k, dim=-1, largest=True, sorted=True) # B, N, 2

        _B, _N, _k = mdist_top.shape
        gather_ids = torch.flatten(mdist_ids_top, start_dim=1, end_dim=2)
        batch_ids = torch.arange(0, _B, device=gather_ids.device).view(-1, 1).repeat(1, _N*_k).contiguous()
        gather_ids = gather_ids.view(-1)
        batch_ids = batch_ids.view(-1)

        ref_pos = anchor_pos_g[batch_ids, gather_ids].view(_B, _N, _k, -1)
        ref_code = anchor_code_g[batch_ids, gather_ids].view(_B, _N, _k, -1)
        input_g_knn, target_g_knn = self._to_knn_training_samples(ref_pos, ref_code, knn_points, knn_sdfs)
        mdist_top = mdist_top.view(_B*_N, _k, 1) + 1e-12
        combine_weight = mdist_top / mdist_top.sum(dim=1, keepdim=True)

        # knn query in Euclidean dist
        gdist = (anchor_pos_g[:, None, :, :] - knn_points[:, :, None, :]).pow(2).sum(dim=-1) # B, N, ng
        gdist_top, gdist_ids_top = gdist.min(dim=-1)  # B, N

        _B, _N = gdist_top.shape
        gather_ids = gdist_ids_top
        batch_ids = torch.arange(0, _B, device=gather_ids.device).view(-1, 1).repeat(1, _N).contiguous()
        gather_ids = gather_ids.view(-1)
        batch_ids = batch_ids.view(-1)

        ref_pos = anchor_pos_g[batch_ids, gather_ids].view(_B, _N, -1)
        ref_code = anchor_code_g[batch_ids, gather_ids].view(_B, _N, -1)
        # to local coord
        points_local = knn_points - ref_pos
        points_local = torch.flatten(points_local, start_dim=0, end_dim=1)
        # prepare anchor code
        ref_code = torch.flatten(ref_code, start_dim=0, end_dim=1)
        input_g_knn_geo = torch.cat([ref_code, points_local], dim=-1)

        ######################### Forward ######################
        pred_global_knn = self.local_decoder(input_g_knn)
        pred_global_knn_geo = self.local_decoder(input_g_knn_geo)

        ######################### Loss ######################
        clamp_dist = self.config.sdf.clamp_dist
        target_g_knn = torch.clamp(target_g_knn, -clamp_dist, clamp_dist)
        pred_global_knn = torch.clamp(pred_global_knn, -clamp_dist, clamp_dist)
        pred_global_knn_geo = torch.clamp(pred_global_knn_geo, -clamp_dist, clamp_dist)

        # sdf loss
        sdf_loss_global = (combine_weight[:, 0, :] * (pred_global_knn[:, 0, :] - target_g_knn).abs()).mean() + \
            (combine_weight[:, 1, :] * (pred_global_knn[:, 1, :] - target_g_knn).abs()).mean()
        sdf_loss_global_geo = (pred_global_knn_geo - target_g_knn).abs().mean()

        # smooth loss
        if self.current_epoch > -1:
            smooth_loss = ((pred_global_knn[:, 0, :] - pred_global_knn[:, 1, :]).abs()).mean()
        else:
            smooth_loss = torch.zeros_like(sdf_loss_global)
            
        # offset reg loss
        offreg_loss = anchor_offset_g.abs().mean()
        if self.current_epoch >= 1:
            self.config.loss.offreg_lambda = 0.0

        loss =  sdf_loss_global*self.config.loss.sdf_global_lambda + \
                sdf_loss_global_geo*self.config.loss.sdf_global_geo_lambda + \
                smooth_loss*self.config.loss.smooth_lambda + \
                offreg_loss*self.config.loss.offreg_lambda

        loss_dict = {
            'loss': loss,
            'sdf_loss_global': sdf_loss_global.detach(),
            'sdf_loss_global_geo': sdf_loss_global_geo.detach(),
            'smooth_loss': smooth_loss.detach(),
            'offreg_loss': offreg_loss.detach()
        }
        self.log_dict(loss_dict, prog_bar=True, logger=False, sync_dist=True)

        return loss_dict

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_sdf_loss_global = torch.stack([x['sdf_loss_global'] for x in outputs]).mean()
        avg_smooth_loss = torch.stack([x['smooth_loss'] for x in outputs]).mean()
        avg_offreg_loss = torch.stack([x['offreg_loss'] for x in outputs]).mean()
        avg_sdf_loss_global_geo = torch.stack([x['sdf_loss_global_geo'] for x in outputs]).mean()

        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar('valid/avg_loss', avg_loss,
                global_step=self.current_epoch)
            self.logger.experiment.add_scalar('valid/avg_sdf_loss_global', avg_sdf_loss_global,
                global_step=self.current_epoch)
            self.logger.experiment.add_scalar('valid/avg_sdf_loss_global_geo', avg_sdf_loss_global_geo,
                global_step=self.current_epoch)
            self.logger.experiment.add_scalar('valid/avg_smooth_loss', avg_smooth_loss,
                global_step=self.current_epoch)
            self.logger.experiment.add_scalar('valid/avg_offreg_loss', avg_offreg_loss,
                global_step=self.current_epoch)

            self.print('========= Epoch {} Validation =========='.format(self.current_epoch))
            self.print('total loss: {:.8f}; sdf loss: {:.8f}; sdf geo loss: {:.8f}; smooth loss: {:.8f}; offreg loss: {:.8f}'\
                .format(avg_loss, avg_sdf_loss_global, avg_sdf_loss_global_geo, avg_smooth_loss, avg_offreg_loss))

        self.log('sdf_loss', avg_sdf_loss_global)

    def forward_mis(self, input_points):
        # encode local point cloud
        input_points = input_points.transpose(1, 2).contiguous()
        anchor_pos_vis, anchor_embed_vis = self.local_encoder(input_points, None, trans_type="coord_only")
        anchor_pos_vis = anchor_pos_vis.transpose(1, 2).contiguous()
        anchor_embed_vis = anchor_embed_vis.transpose(1, 2).contiguous()
        # anchor shape codes
        anchor_sigma_vis = self.e_sigma_head(anchor_embed_vis)
        anchor_rotate_vis = self.e_rotate_head(anchor_embed_vis)
        
        # drop anchors
        to_drop_feas_list = [
            anchor_pos_vis, anchor_embed_vis
        ]
        fea_sizes = [item.shape[-1] for item in to_drop_feas_list]
        to_drop_feas = torch.cat(to_drop_feas_list, dim=-1)
        dropped_feas = drop_anchors(anchor_pos_vis, anchor_sigma_vis, anchor_rotate_vis, self.config.local_encoder.drop_count, to_drop_feas)
        (anchor_pos_vis, anchor_embed_vis) = \
            torch.split(dropped_feas, fea_sizes, dim=-1)
            
        ######################### Global Anchors ######################
        # fetch global features
        mid_embed = self.transformer_global(anchor_pos_vis, anchor_embed_vis.detach())
        pred_mis_points = self.globalpoints_head(mid_embed)

        # global shape code
        anchor_embed_g = self.transformer_local(anchor_pos_vis, anchor_embed_vis, pred_mis_points)

        anchor_pos_g = torch.cat([anchor_pos_vis, pred_mis_points], dim=1)
        anchor_code_g = self.d_shapecode_head(anchor_embed_g)
        anchor_sigma_g = self.d_sigma_head(anchor_embed_g)
        anchor_rotate_g = self.d_rotate_head(anchor_embed_g)
        anchor_offset_g = self.d_offset_head(anchor_embed_g)

        anchor_pos_g = anchor_pos_g + anchor_offset_g

        return anchor_pos_g, anchor_code_g, anchor_sigma_g, anchor_rotate_g
    