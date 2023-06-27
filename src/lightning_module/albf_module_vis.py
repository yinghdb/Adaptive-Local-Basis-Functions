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
from src.net.head_nets import ShapeCodeHead, GlobalPointsHead, SigmaHead, RotateHead
from src.net.dsdf_decoder import Decoder

class VisModule(pl.LightningModule):

    def __init__(self, config, pretrained_ckpt=None):
        super().__init__()

        self.config = config

        # init model
        embed_dim = config.local_encoder.embed_dim
        self.local_encoder = Local_Encoder(self.config.local_encoder, 3)
        self.shapecode_head = ShapeCodeHead(embed_dim, embed_dim)
        self.transformer_global = Transformer_Global(config.transformer_global, embed_dim)
        self.globalpoints_head = GlobalPointsHead(embed_dim, config.transformer_global.num_query_per_obj)
        self.local_decoder = Decoder(config.local_decoder, embed_dim)
        self.sigma_head = SigmaHead(embed_dim)
        self.rotate_head = RotateHead(embed_dim)
        
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
            self.local_encoder.load_state_dict(state_dict['local_encoder'])
            self.local_decoder.load_state_dict(state_dict['local_decoder'])
            self.shapecode_head.load_state_dict(state_dict['shapecode_head'])
            self.sigma_head.load_state_dict(state_dict['sigma_head'])
            self.rotate_head.load_state_dict(state_dict['rotate_head'])
            self.transformer_global.load_state_dict(state_dict['transformer_global'])
            self.globalpoints_head.load_state_dict(state_dict['globalpoints_head'])
            print(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = {
            "local_encoder": self.local_encoder.state_dict(),
            "local_decoder": self.local_decoder.state_dict(),
            "shapecode_head": self.shapecode_head.state_dict(),
            "sigma_head": self.sigma_head.state_dict(),
            "rotate_head": self.rotate_head.state_dict(),
            "transformer_global": self.transformer_global.state_dict(),
            "globalpoints_head": self.globalpoints_head.state_dict(),
        }

    def configure_optimizers(self):
        optim_config = self.config.optimizer
        optim_type = optim_config.type
        param_list = [
            dict(params=self.local_encoder.parameters(), lr=optim_config.lr),
            dict(params=self.local_decoder.parameters(), lr=optim_config.lr),
            dict(params=self.shapecode_head.parameters(), lr=optim_config.lr),
            dict(params=self.sigma_head.parameters(), lr=optim_config.lr),
            dict(params=self.rotate_head.parameters(), lr=optim_config.lr),
            dict(params=self.transformer_global.parameters(), lr=optim_config.lr),
            dict(params=self.globalpoints_head.parameters(), lr=optim_config.lr),
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
        
    def _fetch_chamfer_targets(self, anchor_pos_vis, pred_mis_points, target_mis_points):
        r"""
        input:
        anchor_pos_vis: (B, nk, 3)
        pred_mis_points: (B, mk, 3)
        target_mis_points: (B, tk, 3)
        
        output:
        min_anchor_dist: (B, mk) 
        min_point_dist: (B, mk) 
        """
        fps_num = pred_mis_points.shape[1]
        fps_idx = furthest_point_sample(target_mis_points, fps_num, anchor_pos_vis.contiguous())
        sub_pc = pointnet2_utils.gather_operation(target_mis_points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()

        square_dist = (sub_pc[:, :, None, :] - pred_mis_points[:, None, :, :]).pow(2).sum(dim=-1)  # B, mk, mk
        min_anchor_dist = square_dist.min(dim=1)[0]
        min_point_dist = square_dist.min(dim=2)[0]

        batch_size = pred_mis_points.shape[0]
        anchor_weight = torch.ones_like(min_anchor_dist) / min_anchor_dist.shape[1] / batch_size
        point_weight = torch.ones_like(min_point_dist) / min_point_dist.shape[1] / batch_size

        return min_anchor_dist, min_point_dist, anchor_weight, point_weight

    def training_step(self, batch, batch_idx):
        ######################### Visible Anchors Encoding ######################
        # get input infos
        input_points = batch["input_points"].transpose(1,2).contiguous()

        # encode local point cloud
        anchor_pos_vis, anchor_embed_vis = self.local_encoder(input_points, None, trans_type="coord_only")
        anchor_pos_vis = anchor_pos_vis.transpose(1, 2).contiguous()
        anchor_embed_vis = anchor_embed_vis.transpose(1, 2).contiguous()
        # anchor shape codes
        anchor_code_vis = self.shapecode_head(anchor_embed_vis)
        anchor_sigma_vis = self.sigma_head(anchor_embed_vis)
        anchor_rotate_vis = self.rotate_head(anchor_embed_vis)
        
        # drop anchors
        to_drop_feas_list = [
            anchor_pos_vis, anchor_embed_vis, anchor_code_vis, anchor_sigma_vis, anchor_rotate_vis
        ]
        fea_sizes = [item.shape[-1] for item in to_drop_feas_list]
        to_drop_feas = torch.cat(to_drop_feas_list, dim=-1)
        dropped_feas = drop_anchors(anchor_pos_vis, anchor_sigma_vis, anchor_rotate_vis, self.config.local_encoder.drop_count, to_drop_feas)
        (anchor_pos_vis, anchor_embed_vis, anchor_code_vis, anchor_sigma_vis, anchor_rotate_vis) = \
            torch.split(dropped_feas, fea_sizes, dim=-1)
            
        ######################### Global Anchors Position Prediction ######################
        # fetch global features
        mid_embed = self.transformer_global(anchor_pos_vis, anchor_embed_vis.detach())
        pred_mis_points = self.globalpoints_head(mid_embed)

        ######################### Training Targets for Visible Anchors ######################
        ## prepare for knn query loss
        knn_points = batch['nonface_sdfs_vis'][:, :, :3]
        knn_sdfs = batch['nonface_sdfs_vis'][:, :, 3:]

        # knn query
        num_k = self.config.sdf.global_query_k
        mdist = m_dist(anchor_pos_vis, anchor_sigma_vis, anchor_rotate_vis, knn_points) # B, N, ng
        mdist_top, mdist_ids_top = torch.topk(mdist, num_k, dim=-1, largest=True, sorted=True) # B, N, 2

        _B, _N, _k = mdist_top.shape
        gather_ids = torch.flatten(mdist_ids_top, start_dim=1, end_dim=2)
        batch_ids = torch.arange(0, _B, device=gather_ids.device).view(-1, 1).repeat(1, _N*_k).contiguous()
        gather_ids = gather_ids.view(-1)
        batch_ids = batch_ids.view(-1)
        
        ref_pos = anchor_pos_vis[batch_ids, gather_ids].view(_B, _N, _k, -1)
        ref_code = anchor_code_vis[batch_ids, gather_ids].view(_B, _N, _k, -1)
        input_vis_knn, target_vis_knn = self._to_knn_training_samples(ref_pos, ref_code, knn_points, knn_sdfs)
        mdist_top = mdist_top.view(_B*_N, _k, 1) + 1e-12
        combine_weight = mdist_top / mdist_top.sum(dim=1, keepdim=True)

        ######################### Training Targets for Missing Anchors ######################
        ## prepare for charmer loss
        min_anchor_dist, min_point_dist, anchor_weight, point_weight = \
            self._fetch_chamfer_targets(anchor_pos_vis, pred_mis_points, batch["missing_points"])

        ######################### Forward ######################
        pred_vis_knn = self.local_decoder(input_vis_knn)

        ######################### Loss ######################
        # sdf loss
        sdf_loss_vis = (combine_weight[:, 0, :] * (pred_vis_knn[:, 0, :] - target_vis_knn).abs()).mean() + \
            (combine_weight[:, 1, :] * (pred_vis_knn[:, 1, :] - target_vis_knn).abs()).mean()

        # charmfer loss
        chamfer_loss = (min_anchor_dist.sqrt() * anchor_weight).sum() + (min_point_dist.sqrt() * point_weight).sum()

        loss =  sdf_loss_vis*self.config.loss.sdf_vis_lambda + \
                chamfer_loss*self.config.loss.chamfer_lambda

        loss_dict = {
            'loss': loss,
            'sdf_loss_vis': sdf_loss_vis.detach(),
            'chamfer_loss': chamfer_loss.detach(),
        }
        self.log_dict(loss_dict, prog_bar=True, logger=False, sync_dist=True)

        return loss_dict

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_sdf_loss_vis = torch.stack([x['sdf_loss_vis'] for x in outputs]).mean()
        avg_chamfer_loss = torch.stack([x['chamfer_loss'] for x in outputs]).mean()

        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar('train/avg_loss', avg_loss,
                global_step=self.current_epoch)
            self.logger.experiment.add_scalar('train/avg_sdf_loss_vis', avg_sdf_loss_vis,
                global_step=self.current_epoch)
            self.logger.experiment.add_scalar('train/avg_chamfer_loss', avg_chamfer_loss,
                global_step=self.current_epoch)

            self.print('========= Epoch {} Training =========='.format(self.current_epoch))
            self.print('total loss: {:.8f}; sdf loss: {:.8f}; chamfer loss: {:.8f}'\
                .format(avg_loss, avg_sdf_loss_vis, avg_chamfer_loss))

    def validation_step(self, batch, batch_idx):
        ######################### Visible Anchors Encoding ######################
        # get input infos
        input_points = batch["input_points"].transpose(1,2).contiguous()

        # encode local point cloud
        anchor_pos_vis, anchor_embed_vis = self.local_encoder(input_points, None, trans_type="coord_only")
        anchor_pos_vis = anchor_pos_vis.transpose(1, 2).contiguous()
        anchor_embed_vis = anchor_embed_vis.transpose(1, 2).contiguous()
        # anchor shape codes
        anchor_code_vis = self.shapecode_head(anchor_embed_vis)
        anchor_sigma_vis = self.sigma_head(anchor_embed_vis)
        anchor_rotate_vis = self.rotate_head(anchor_embed_vis)
        
        # drop anchors
        to_drop_feas_list = [
            anchor_pos_vis, anchor_embed_vis, anchor_code_vis, anchor_sigma_vis, anchor_rotate_vis
        ]
        fea_sizes = [item.shape[-1] for item in to_drop_feas_list]
        to_drop_feas = torch.cat(to_drop_feas_list, dim=-1)
        dropped_feas = drop_anchors(anchor_pos_vis, anchor_sigma_vis, anchor_rotate_vis, self.config.local_encoder.drop_count, to_drop_feas)
        (anchor_pos_vis, anchor_embed_vis, anchor_code_vis, anchor_sigma_vis, anchor_rotate_vis) = \
            torch.split(dropped_feas, fea_sizes, dim=-1)
            
        ######################### Global Anchors Position Prediction ######################
        # fetch global features
        mid_embed = self.transformer_global(anchor_pos_vis, anchor_embed_vis.detach())
        pred_mis_points = self.globalpoints_head(mid_embed)

        ######################### Training Targets for Visible Anchors ######################
        ## prepare for knn query loss
        knn_points = batch['nonface_sdfs_vis'][:, :, :3]
        knn_sdfs = batch['nonface_sdfs_vis'][:, :, 3:]

        # knn query
        num_k = self.config.sdf.global_query_k
        mdist = m_dist(anchor_pos_vis, anchor_sigma_vis, anchor_rotate_vis, knn_points) # B, N, ng
        mdist_top, mdist_ids_top = torch.topk(mdist, num_k, dim=-1, largest=True, sorted=True) # B, N, num_k

        _B, _N, _k = mdist_top.shape
        gather_ids = torch.flatten(mdist_ids_top, start_dim=1, end_dim=2)
        batch_ids = torch.arange(0, _B, device=gather_ids.device).view(-1, 1).repeat(1, _N*_k).contiguous()
        gather_ids = gather_ids.view(-1)
        batch_ids = batch_ids.view(-1)
        
        ref_pos = anchor_pos_vis[batch_ids, gather_ids].view(_B, _N, _k, -1)
        ref_code = anchor_code_vis[batch_ids, gather_ids].view(_B, _N, _k, -1)
        input_vis_knn, target_vis_knn = self._to_knn_training_samples(ref_pos, ref_code, knn_points, knn_sdfs)
        mdist_top = mdist_top.view(_B*_N, _k, 1) + 1e-12
        combine_weight = mdist_top / mdist_top.sum(dim=1, keepdim=True)

        ######################### Training Targets for Missing Anchors ######################
        ## prepare for charmer loss
        min_anchor_dist, min_point_dist, anchor_weight, point_weight = \
            self._fetch_chamfer_targets(anchor_pos_vis, pred_mis_points, batch["missing_points"])

        ######################### Forward ######################
        pred_vis_knn = self.local_decoder(input_vis_knn)

        ######################### Loss ######################
        # sdf loss
        sdf_loss_vis = (combine_weight[:, 0, :] * (pred_vis_knn[:, 0, :] - target_vis_knn).abs()).mean() + \
            (combine_weight[:, 1, :] * (pred_vis_knn[:, 1, :] - target_vis_knn).abs()).mean()

        # charmfer loss
        chamfer_loss = (min_anchor_dist.sqrt() * anchor_weight).sum() + (min_point_dist.sqrt() * point_weight).sum()

        loss =  sdf_loss_vis*self.config.loss.sdf_vis_lambda + \
                chamfer_loss*self.config.loss.chamfer_lambda

        loss_dict = {
            'loss': loss,
            'sdf_loss_vis': sdf_loss_vis.detach(),
            'chamfer_loss': chamfer_loss.detach(),
        }
        self.log_dict(loss_dict, prog_bar=True, logger=False, sync_dist=True)

        return loss_dict

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_sdf_loss_vis = torch.stack([x['sdf_loss_vis'] for x in outputs]).mean()
        avg_chamfer_loss = torch.stack([x['chamfer_loss'] for x in outputs]).mean()

        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar('valid/avg_loss', avg_loss,
                global_step=self.current_epoch)
            self.logger.experiment.add_scalar('valid/avg_sdf_loss_vis', avg_sdf_loss_vis,
                global_step=self.current_epoch)
            self.logger.experiment.add_scalar('valid/avg_chamfer_loss', avg_chamfer_loss,
                global_step=self.current_epoch)

            self.print('========= Epoch {} Validation =========='.format(self.current_epoch))

            self.print('total loss: {:.8f}; sdf loss: {:.8f}; chamfer loss: {:.8f}'\
                .format(avg_loss, avg_sdf_loss_vis, avg_chamfer_loss))
        self.log('sdf_loss', avg_sdf_loss_vis)

    def forward_vis(self, input_points):
        # encode local point cloud
        input_points = input_points.transpose(1, 2).contiguous()
        anchor_pos_vis, anchor_embed_vis = self.local_encoder(input_points, None, trans_type="coord_only")
        anchor_pos_vis = anchor_pos_vis.transpose(1, 2).contiguous()
        anchor_embed_vis = anchor_embed_vis.transpose(1, 2).contiguous()
        # anchor shape codes
        anchor_code_vis = self.shapecode_head(anchor_embed_vis)
        anchor_sigma_vis = self.sigma_head(anchor_embed_vis)
        anchor_rotate_vis = self.rotate_head(anchor_embed_vis)
        
        # drop anchors
        to_drop_feas_list = [
            anchor_pos_vis, anchor_embed_vis, anchor_code_vis, anchor_sigma_vis, anchor_rotate_vis
        ]
        fea_sizes = [item.shape[-1] for item in to_drop_feas_list]
        to_drop_feas = torch.cat(to_drop_feas_list, dim=-1)
        dropped_feas = drop_anchors(anchor_pos_vis, anchor_sigma_vis, anchor_rotate_vis, self.config.local_encoder.drop_count, to_drop_feas)
        (anchor_pos_vis, anchor_embed_vis, anchor_code_vis, anchor_sigma_vis, anchor_rotate_vis) = \
            torch.split(dropped_feas, fea_sizes, dim=-1)
            
        ######################### Global Anchors Position Prediction ######################
        # fetch global features
        mid_embed = self.transformer_global(anchor_pos_vis, anchor_embed_vis.detach())
        pred_mis_points = self.globalpoints_head(mid_embed)

        return anchor_pos_vis.contiguous(), anchor_code_vis.contiguous(), anchor_sigma_vis.contiguous(), anchor_rotate_vis.contiguous(), pred_mis_points.contiguous()
    