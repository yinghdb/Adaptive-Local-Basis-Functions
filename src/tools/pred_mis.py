import argparse
import os
import numpy as np
import torch
import sys
import math
import matplotlib
import seaborn as sns
import pytorch_lightning as pl
sys.path.append("./")

from src.config.default import get_cfg_defaults
from src.utils.data import compute_sdf_grid, compute_sdf_grid_sep, m_dist
from src.utils.data_export import save_ply, save_mesh, sdf2mesh
from src.lightning_module.albf_module_mis import MisModule

parser = argparse.ArgumentParser()
parser.add_argument('cfg_path', type=str, help='config path')
parser.add_argument('--ckpt_path', type=str, default=None, help='pretrained checkpoint path')
parser.add_argument('--pred_list', type=str, default=None, help='to predict list')
parser.add_argument('--output_root_dir', type=str, default="./samples/")
parser.add_argument('--step_size', type=float, default=0.01)
parser.add_argument('--lr_code', type=float, default=1e-3)
parser.add_argument('--lr_offset', type=float, default=1e-3)
parser.add_argument('--iteration', type=int, default=1000)
args = parser.parse_args()
cmap = matplotlib.cm.get_cmap('jet')
color_list = sns.color_palette("hls", 128)
color_list = color_list[0::4] + color_list[2::4] + color_list[1::4] + color_list[3::4]

def load_input_data(data_root_dir, class_id, obj_id, render_id, observation_dir):
    data_path = os.path.join(data_root_dir, observation_dir, class_id, obj_id, render_id+".npz")
    npz_data = np.load(data_path)
    input_points = torch.FloatTensor(npz_data['input_points']).cuda()

    return input_points
    
def forward_knn_sdf(local_decoder, ref_pos, ref_code, points):
    # to local coord
    points_local = points[:, :, None, :] - ref_pos
    points_local = torch.flatten(points_local, start_dim=0, end_dim=1)
    # prepare anchor code
    ref_code = torch.flatten(ref_code, start_dim=0, end_dim=1)

    input_data = torch.cat([ref_code, points_local], dim=-1)
    pred_sdf = local_decoder(input_data)

    return pred_sdf

if __name__ == "__main__":
    # read data list
    with open(args.pred_list, 'r') as f:
        lines = f.readlines()
    file_list = []
    for line in lines:
        file_list += [line.strip().split(" ")]

    config = get_cfg_defaults()
    config.merge_from_file(args.cfg_path)
    pl.seed_everything(config.trainer.seed) 

    ######################### Init Module ######################
    model = MisModule(config, pretrained_ckpt=args.ckpt_path).cuda()
    model.eval()
    
    for i in range(len(file_list)):
        class_id = file_list[i][0]
        obj_id = file_list[i][1]
        render_id = file_list[i][2]

        print("predicting %s %s %s" % (class_id, obj_id, render_id))    

        ######################### Get Data ######################
        input_points = load_input_data(config.dataset.sample_data_root, class_id, obj_id, render_id, config.dataset.observation_dir)
        input_points.unsqueeze_(0)
    
        ######################### Fetch Local bases ######################
        with torch.no_grad():
            anchor_pos, anchor_code, anchor_sigma, anchor_rotate = model.forward_mis(input_points)

        ######################### Save Input Points ######################
        input_pts = input_points[0].cpu().numpy()
        ouput_file = os.path.join(args.output_root_dir, "input_points", class_id, obj_id, render_id+".ply")
        save_ply(input_pts, ouput_file)

        ######################### Save Predicted Mesh ######################
        # compute sdf grid
        with torch.no_grad():
            sdf_grid, origin, step_size = compute_sdf_grid( \
                model.local_decoder, anchor_pos[0], anchor_code[0], anchor_sigma[0], anchor_rotate[0], step_size=args.step_size, query_k=config.sdf.global_query_k)

        # if (sdf_grid*mask < 0).sum() < 8 or (sdf_grid*mask > 0).sum() < 8:
        #     continue
        # to mesh
        verts, faces = sdf2mesh(sdf_grid, origin, step_size)
        # save
        ouput_file = os.path.join(args.output_root_dir, "mesh_decode", class_id, obj_id, render_id+".ply")
        save_mesh(verts, faces, ouput_file)

        ######################### Save Predicted Local Bases ######################
        # sample sdfs
        with torch.no_grad():
            sdf_grid_list, mask_list, origin, step_size = compute_sdf_grid_sep( \
                model.local_decoder, anchor_pos[0], anchor_code[0], anchor_sigma[0], anchor_rotate[0])
        # to mesh
        for grid_id in range(sdf_grid_list.shape[0]):
            # if (sdf_grid_list[grid_id]*mask_list[grid_id] < 0).sum() < 8 or (sdf_grid_list[grid_id]*mask_list[grid_id] > 0).sum() < 8:
            #     continue
            try:
                verts, faces = sdf2mesh(sdf_grid_list[grid_id], origin, step_size, mask_list[grid_id])
            except:
                continue

            # center = verts.mean(axis=0)
            # offset = center * 0.1
            # verts = verts + offset

            color = color_list[grid_id]
            colors = np.zeros_like(verts) + color
            # save
            ouput_file = os.path.join(args.output_root_dir, "mesh_decode_sep", class_id, obj_id, render_id, str(grid_id)+".ply")
            save_mesh(verts, faces, ouput_file, colors)


        ######################### Start Optimization ######################
        def sample_near_anchors(num_sample_per_anchor, radius):
            anchor_num = anchor_pos.shape[1]
            samples = (torch.rand(anchor_num, num_sample_per_anchor, 3) - 0.5) * 2 * radius
            samples = anchor_pos[0][:, None, :] + samples.to(anchor_pos.device)

            return samples.view(-1, 3)
        anchor_offset = torch.zeros_like(anchor_pos)
        anchor_code_ori = anchor_code.clone()
        anchor_offset.requires_grad = True
        anchor_code.requires_grad = True
        param_list = [
            dict(params=anchor_code, lr=args.lr_code),
            dict(params=anchor_offset, lr=args.lr_offset),
        ]
        optimizer = torch.optim.Adam(param_list)

        # sample points
        sampled_pts = sample_near_anchors(128, 0.1).unsqueeze(0)

        for it in range(args.iteration):
            optimizer.zero_grad()

            anchor_pos_offset = anchor_pos + anchor_offset
            ## for surface loss 
            mdist = m_dist(anchor_pos, anchor_sigma, anchor_rotate, input_points) # 1, N, ng
            mdist_top, mdist_ids_top = torch.topk(mdist, 2, dim=-1, largest=True, sorted=True) # 1, N, 2
            N = mdist_ids_top.shape[1]

            gather_ids = torch.flatten(mdist_ids_top, start_dim=1, end_dim=2)  # 1, N*2
            ref_pos = anchor_pos_offset[0, gather_ids[0]].view(1, N, 2, 3)
            ref_code = anchor_code[0, gather_ids[0]].view(1, N, 2, -1)
            
            mdist_top = mdist_top[0] + 1e-12
            combine_weight = mdist_top / mdist_top.sum(dim=1, keepdim=True)

            sdf_surface = forward_knn_sdf(model.local_decoder, ref_pos, ref_code, input_points)
            sdf_surface = (sdf_surface * combine_weight[:, :, None]).sum(dim=1)
            surface_loss = torch.clamp_min(sdf_surface.abs() - 0.01, 0.0).pow(2).mean()

            ## for smooth loss 
            mdist = m_dist(anchor_pos, anchor_sigma, anchor_rotate, sampled_pts) # 1, N, ng
            mdist_top, mdist_ids_top = torch.topk(mdist, 2, dim=-1, largest=True, sorted=True) # 1, N, 2
            N = mdist_ids_top.shape[1]

            gather_ids = torch.flatten(mdist_ids_top, start_dim=1, end_dim=2)  # 1, N*2
            ref_pos = anchor_pos_offset[0, gather_ids[0]].view(1, N, 2, 3)
            ref_code = anchor_code[0, gather_ids[0]].view(1, N, 2, -1)

            sdf_sampled = forward_knn_sdf(model.local_decoder, ref_pos, ref_code, sampled_pts)
            dist_weight = sdf_sampled.abs().min(dim=1)[0]
            dist_weight = torch.exp(-dist_weight**2 * 10000)
            mdist_top = mdist_top[0] + 1e-12
            combine_weight = mdist_top / mdist_top.sum(dim=1, keepdim=True)
            smooth_weight = combine_weight[:, 0] - combine_weight[:, 1]
            smooth_weight = torch.exp(-smooth_weight**2 * 1000)
            smooth_weight = dist_weight * smooth_weight
            smooth_loss = (smooth_weight * ((sdf_sampled[:, 0] - sdf_sampled[:, 1]).pow(2))).mean()

            ## for maintain loss 
            maintain_code_loss = ((anchor_code - anchor_code_ori) ** 2).mean()
            maintain_offset_loss = (anchor_offset ** 2).mean()

            loss = surface_loss * 1.0 + smooth_loss * 10.0 + maintain_code_loss * 1e-1 + maintain_offset_loss * 1e-1
            loss.backward()
            optimizer.step()

            if it % 100 == 0:
                print("loss: ", loss.detach().cpu().numpy(),
                    "| surface loss: ", surface_loss.detach().cpu().numpy(), 
                    "| smooth loss: ", smooth_loss.detach().cpu().numpy(), 
                    "| maintain code loss: ", maintain_code_loss.detach().cpu().numpy(),
                    "| maintain offset loss: ", maintain_offset_loss.detach().cpu().numpy(),
                )

        ######################### Save Optimized Mesh ######################
        # compute sdf grid
        with torch.no_grad():
            sdf_grid, origin, step_size = compute_sdf_grid( \
                model.local_decoder, anchor_pos_offset[0], anchor_code[0], anchor_sigma[0], anchor_rotate[0], step_size=args.step_size, query_k=config.sdf.global_query_k)
        # to mesh
        verts, faces = sdf2mesh(sdf_grid, origin, step_size)
        # save
        ouput_file = os.path.join(args.output_root_dir, "mesh_optimized", class_id, obj_id, render_id+".ply")
        save_mesh(verts, faces, ouput_file)