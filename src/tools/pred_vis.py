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
from src.utils.data import compute_sdf_grid_vis, compute_sdf_grid_vis_sep, m_dist
from src.utils.data_export import save_ply, save_mesh, sdf2mesh
from src.lightning_module.albf_module_vis import VisModule

parser = argparse.ArgumentParser()
parser.add_argument('cfg_path', type=str, help='config path')
parser.add_argument('--ckpt_path', type=str, default=None, help='pretrained checkpoint path')
parser.add_argument('--pred_list', type=str, default=None, help='to predict list')
parser.add_argument('--output_root_dir', type=str, default="./samples/")
parser.add_argument('--step_size', type=float, default=0.01)
args = parser.parse_args()
cmap = matplotlib.cm.get_cmap('jet')
color_list = sns.color_palette("hls", 128)
color_list = color_list[0::4] + color_list[2::4] + color_list[1::4] + color_list[3::4]

def load_input_data(data_root_dir, class_id, obj_id, render_id, observation_dir):
    data_path = os.path.join(data_root_dir, observation_dir, class_id, obj_id, render_id+".npz")
    npz_data = np.load(data_path)
    input_points = torch.FloatTensor(npz_data['input_points']).cuda()

    return input_points

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
    model = VisModule(config, pretrained_ckpt=args.ckpt_path).cuda()
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
            anchor_pos, anchor_code, anchor_sigma, anchor_rotate, miss_pts = model.forward_vis(input_points)

        ######################### Save Input Points ######################
        input_pts = input_points[0].cpu().numpy()
        ouput_file = os.path.join(args.output_root_dir, "input_points", class_id, obj_id, render_id+".ply")
        save_ply(input_pts, ouput_file)

        ######################### Save Predicted Missing Points ######################
        miss_pts = miss_pts[0].cpu().numpy()  # ng, 3
        ouput_file = os.path.join(args.output_root_dir, "missing_points", class_id, obj_id, render_id+".ply")
        save_ply(miss_pts, ouput_file)

        ######################### Save Predicted Mesh ######################
        # compute sdf grid
        with torch.no_grad():
            sdf_grid, sdf_mask, origin, step_size = compute_sdf_grid_vis( \
                model.local_decoder, anchor_pos[0], anchor_code[0], anchor_sigma[0], anchor_rotate[0], input_points[0], radius=0.05, query_k=2, step_size=args.step_size)
        # to mesh
        verts, faces = sdf2mesh(sdf_grid, origin, step_size, sdf_mask)
        # save
        ouput_file = os.path.join(args.output_root_dir, "mesh_encode", class_id, obj_id, render_id+".ply")
        save_mesh(verts, faces, ouput_file)

        ######################### Save Predicted Local Bases ######################
        # sample sdfs
        with torch.no_grad():
            sdf_grid_list, mask_list, origin, step_size = compute_sdf_grid_vis_sep( \
                model.local_decoder, anchor_pos[0], anchor_code[0], anchor_sigma[0], anchor_rotate[0], input_points[0], radius=0.05, step_size=args.step_size)
        # to mesh
        for grid_id in range(sdf_grid_list.shape[0]):
            # if (sdf_grid_list[grid_id]*mask_list[grid_id] < 0).sum() < 4 or (sdf_grid_list[grid_id]*mask_list[grid_id] > 0).sum() < 4:
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
            ouput_file = os.path.join(args.output_root_dir, "mesh_encode_sep", class_id, obj_id, render_id, str(grid_id)+".ply")
            save_mesh(verts, faces, ouput_file, colors)
