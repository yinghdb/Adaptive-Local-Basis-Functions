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
from src.utils.lr_scheduler import WarmupMultiStepLR

from src.net.local_encoder import Local_Encoder
from src.net.head_nets import ShapeCodeHead, GlobalPointsHead, SigmaHead, RotateHead
from src.net.dsdf_decoder import Decoder

class RecModule(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.config = config

        # init model
        embed_dim = config.local_encoder.embed_dim
        self.local_encoder = Local_Encoder(self.config.local_encoder, 3)
        self.shapecode_head = ShapeCodeHead(embed_dim, embed_dim)
        self.dsdf_decoder = Decoder(config.local_decoder, embed_dim)
        self.sigma_head = SigmaHead(embed_dim)
        self.rotate_head = RotateHead(embed_dim)

    def configure_optimizers(self):
        optim_config = self.config.optimizer
        optim_type = optim_config.type
        param_list = [
            dict(params=self.local_encoder.parameters(), lr=optim_config.lr),
            dict(params=self.dsdf_decoder.parameters(), lr=optim_config.lr),
            dict(params=self.shapecode_head.parameters(), lr=optim_config.lr),
            dict(params=self.sigma_head.parameters(), lr=optim_config.lr),
            dict(params=self.rotate_head.parameters(), lr=optim_config.lr),
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
        elif sched_type == "WarmupMultiStepLR":
            scheduler.update(
                {'scheduler': WarmupMultiStepLR(
                            optimizer, 
                            sche_config.mslr_milestones, 
                            sche_config.gamma,
                            warmup_factor=sche_config.warmup_factor,
                            warmup_iters=sche_config.warmup_iters,
                            warmup_method=sche_config.warmup_method,
                )})
        else:
            raise NotImplementedError()

        return [optimizer], [scheduler]

    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_idx,
        optimizer_closure, on_tpu, using_native_amp, using_lbfgs):


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
        
    def prepare_training(self, anchor_code, anchor_pos, anchor_sigma, anchor_rotate, query_points, query_sdfs):
        num_k = self.config.sdf.global_query_k
        mdist = m_dist(anchor_pos, anchor_sigma, anchor_rotate, query_points) # B, N, ng
        mdist_top, mdist_ids_top = torch.topk(mdist, num_k, dim=-1, largest=True, sorted=True) # B, N, 2

        _B, _N, _k = mdist_top.shape
        gather_ids = torch.flatten(mdist_ids_top, start_dim=1, end_dim=2)
        batch_ids = torch.arange(0, _B, device=gather_ids.device).view(-1, 1).repeat(1, _N*_k).contiguous()
        gather_ids = gather_ids.view(-1)
        batch_ids = batch_ids.view(-1)
        
        ref_pos = anchor_pos[batch_ids, gather_ids].view(_B, _N, _k, -1)
        ref_code = anchor_code[batch_ids, gather_ids].view(_B, _N, _k, -1)
        training_input, training_target = self._to_knn_training_samples(ref_pos, ref_code, query_points, query_sdfs)
        mdist_top = mdist_top + 1e-12
        combine_weight = mdist_top / mdist_top.sum(dim=2, keepdim=True) # B, N, 2
        combine_weight = combine_weight.view(_B*_N, _k, 1)

        return training_input, training_target, combine_weight

    def training_step(self, batch, batch_idx):
        ######################### Visible Anchors Encoding ######################
        # get input infos
        input_points = batch["face_pts"].transpose(1,2).contiguous()

        # encode local point cloud
        anchor_pos, anchor_embed = self.local_encoder(input_points, None, trans_type="coord_only")
        anchor_pos = anchor_pos.transpose(1, 2).contiguous()
        anchor_embed = anchor_embed.transpose(1, 2).contiguous()
        # anchor shape codes
        anchor_code = self.shapecode_head(anchor_embed)
        anchor_sigma = self.sigma_head(anchor_embed)
        anchor_rotate = self.rotate_head(anchor_embed)
        
        # drop anchors
        to_drop_feas_list = [
            anchor_pos, anchor_embed, anchor_code, anchor_sigma, anchor_rotate
        ]
        fea_sizes = [item.shape[-1] for item in to_drop_feas_list]
        to_drop_feas = torch.cat(to_drop_feas_list, dim=-1)
        dropped_feas = drop_anchors(anchor_pos, anchor_sigma, anchor_rotate, self.config.local_encoder.drop_count, to_drop_feas)
        (anchor_pos, anchor_embed, anchor_code, anchor_sigma, anchor_rotate) = \
            torch.split(dropped_feas, fea_sizes, dim=-1)

        ######################### Training Targets for Visible Anchors ######################
        ## prepare for knn query loss
        uniform_points = batch['uniform_pts']
        uniform_sdfs = batch['uniform_sdfs']
        nearface_points = batch['nearface_pts']
        nearface_sdfs = batch['nearface_sdfs']

        uniform_input, uniform_target, uniform_weights = self.prepare_training(anchor_code, anchor_pos, anchor_sigma, anchor_rotate, uniform_points, uniform_sdfs)
        nearface_input, nearface_target, nearface_weights = self.prepare_training(anchor_code, anchor_pos, anchor_sigma, anchor_rotate, nearface_points, nearface_sdfs)

        ######################### Forward ######################
        uniform_pred = self.dsdf_decoder(uniform_input)
        nearface_pred = self.dsdf_decoder(nearface_input)

        # clamp dist
        clamp_dist = self.config.sdf.clamp_dist
        uniform_target = torch.clamp(uniform_target, -clamp_dist, clamp_dist)
        uniform_pred = torch.clamp(uniform_pred, -clamp_dist, clamp_dist)

        ######################### Loss ######################

        # sdf loss
        sdf_loss = (nearface_weights[:, 0, :] * (nearface_pred[:, 0, :] - nearface_target).abs()).mean() + \
            (nearface_weights[:, 1, :] * (nearface_pred[:, 1, :] - nearface_target).abs()).mean()
        uniform_sdf_loss = (uniform_weights[:, 0, :] * (uniform_pred[:, 0, :] - uniform_target).abs()).mean() + \
            (uniform_weights[:, 1, :] * (uniform_pred[:, 1, :] - uniform_target).abs()).mean()
        sdf_loss = sdf_loss + uniform_sdf_loss

        # smooth loss
        smooth_loss = ((nearface_pred[:, 0, :].detach() - nearface_pred[:, 1, :]).abs()).mean()

        loss = sdf_loss + smooth_loss * self.config.loss.smooth_lambda

        loss_dict = {
            'loss': loss,
            'sdf_loss': sdf_loss,
            'smooth_loss': smooth_loss
        }
        self.log_dict(loss_dict, prog_bar=True, logger=False, sync_dist=True)

        return loss_dict

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_sdf_loss = torch.stack([x['sdf_loss'] for x in outputs]).mean()
        avg_smooth_loss = torch.stack([x['smooth_loss'] for x in outputs]).mean()

        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar('train/avg_loss', avg_loss,
                global_step=self.current_epoch)
            self.logger.experiment.add_scalar('train/avg_sdf_loss', avg_sdf_loss,
                global_step=self.current_epoch)
            self.logger.experiment.add_scalar('train/avg_smooth_loss', avg_smooth_loss,
                global_step=self.current_epoch)

            self.print('========= Epoch {} Training =========='.format(self.current_epoch))
            self.print('sdf loss: {:.8f}; smooth loss: {:.8f};'\
                .format(avg_sdf_loss, avg_smooth_loss))

    def validation_step(self, batch, batch_idx):
         ######################### Visible Anchors Encoding ######################
        # get input infos
        input_points = batch["face_pts"].transpose(1, 2).contiguous()

        # encode local point cloud
        anchor_pos, anchor_embed = self.local_encoder(input_points, None, trans_type="coord_only")
        anchor_pos = anchor_pos.transpose(1, 2).contiguous()
        anchor_embed = anchor_embed.transpose(1, 2).contiguous()
        # anchor shape codes
        anchor_code = self.shapecode_head(anchor_embed)
        anchor_sigma = self.sigma_head(anchor_embed)
        anchor_rotate = self.rotate_head(anchor_embed)

        # drop anchors
        to_drop_feas_list = [
            anchor_pos, anchor_embed, anchor_code, anchor_sigma, anchor_rotate
        ]
        fea_sizes = [item.shape[-1] for item in to_drop_feas_list]
        to_drop_feas = torch.cat(to_drop_feas_list, dim=-1)
        dropped_feas = drop_anchors(anchor_pos, anchor_sigma, anchor_rotate, self.config.local_encoder.drop_count, to_drop_feas)
        (anchor_pos, anchor_embed, anchor_code, anchor_sigma, anchor_rotate) = \
            torch.split(dropped_feas, fea_sizes, dim=-1)

        ######################### Training Targets for Visible Anchors ######################
        ## prepare for knn query loss
        uniform_points = batch['uniform_pts']
        uniform_sdfs = batch['uniform_sdfs']
        nearface_points = batch['nearface_pts']
        nearface_sdfs = batch['nearface_sdfs']

        uniform_input, uniform_target, uniform_weights = self.prepare_training(anchor_code, anchor_pos, anchor_sigma, anchor_rotate, uniform_points, uniform_sdfs)
        nearface_input, nearface_target, nearface_weights = self.prepare_training(anchor_code, anchor_pos, anchor_sigma, anchor_rotate, nearface_points, nearface_sdfs)

        ######################### Forward ######################
        uniform_pred = self.dsdf_decoder(uniform_input)
        nearface_pred = self.dsdf_decoder(nearface_input)

        # clamp dist
        clamp_dist = self.config.sdf.clamp_dist
        uniform_target = torch.clamp(uniform_target, -clamp_dist, clamp_dist)
        uniform_pred = torch.clamp(uniform_pred, -clamp_dist, clamp_dist)

        ######################### Loss ######################

        # sdf loss
        sdf_loss = (nearface_weights[:, 0, :] * (nearface_pred[:, 0, :] - nearface_target).abs()).mean() + \
            (nearface_weights[:, 1, :] * (nearface_pred[:, 1, :] - nearface_target).abs()).mean()
        uniform_sdf_loss = (uniform_weights[:, 0, :] * (uniform_pred[:, 0, :] - uniform_target).abs()).mean() + \
            (uniform_weights[:, 1, :] * (uniform_pred[:, 1, :] - uniform_target).abs()).mean()
        sdf_loss = sdf_loss + uniform_sdf_loss

        # smooth loss
        smooth_loss = ((nearface_pred[:, 0, :].detach() - nearface_pred[:, 1, :]).abs()).mean()

        loss = sdf_loss + smooth_loss * self.config.loss.smooth_lambda

        loss_dict = {
            'loss': loss,
            'sdf_loss': sdf_loss,
            'smooth_loss': smooth_loss
        }
        self.log_dict(loss_dict, prog_bar=True, logger=False, sync_dist=True)

        return loss_dict


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_sdf_loss = torch.stack([x['sdf_loss'] for x in outputs]).mean()
        avg_smooth_loss = torch.stack([x['smooth_loss'] for x in outputs]).mean()

        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar('valid/avg_loss', avg_loss,
                global_step=self.current_epoch)
            self.logger.experiment.add_scalar('valid/avg_sdf_loss', avg_sdf_loss,
                global_step=self.current_epoch)
            self.logger.experiment.add_scalar('valid/avg_smooth_loss', avg_smooth_loss,
                global_step=self.current_epoch)

            self.print('========= Epoch {} Validation =========='.format(self.current_epoch))
            self.print('sdf loss: {:.8f};  smooth loss: {:.8f};'\
                    .format(avg_sdf_loss, avg_smooth_loss))
        self.log('sdf_loss', avg_loss)

    def forward_enc(self, input_points):
        input_points = input_points.transpose(1,2).contiguous()
        # encode local point cloud
        anchor_pos, anchor_embed = self.local_encoder(input_points, None, trans_type="coord_only")
        anchor_pos = anchor_pos.transpose(1, 2).contiguous()
        anchor_embed = anchor_embed.transpose(1, 2).contiguous()
        # anchor shape codes
        anchor_code = self.shapecode_head(anchor_embed)
        anchor_sigma = self.sigma_head(anchor_embed)
        anchor_rotate = self.rotate_head(anchor_embed)

        # drop anchors
        to_drop_feas_list = [
            anchor_pos, anchor_code, anchor_sigma, anchor_rotate
        ]
        fea_sizes = [item.shape[-1] for item in to_drop_feas_list]
        to_drop_feas = torch.cat(to_drop_feas_list, dim=-1)
        dropped_feas = drop_anchors(anchor_pos, anchor_sigma, anchor_rotate, self.config.local_encoder.drop_count, to_drop_feas)
        (anchor_pos, anchor_code, anchor_sigma, anchor_rotate) = \
            torch.split(dropped_feas, fea_sizes, dim=-1)

        return anchor_pos.contiguous(), anchor_code.contiguous(), anchor_sigma.contiguous(), anchor_rotate.contiguous()
    