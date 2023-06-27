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
from src.utils.data import compute_sdf_grid, compute_sdf_grid_sep
from src.utils.data_export import save_ply, save_mesh, sdf2mesh
from src.lightning_module.albf_module_rec import RecModule

parser = argparse.ArgumentParser()
parser.add_argument('cfg_path', type=str, help='config path')
parser.add_argument('--ckpt_path', type=str, default=None, help='pretrained checkpoint path')
parser.add_argument('--pred_list', type=str, default=None, help='to predict list')
parser.add_argument('--output_root_dir', type=str, default="./samples/")
args = parser.parse_args()
cmap = matplotlib.cm.get_cmap('jet')
color_list = sns.color_palette("hls", 128)
color_list = color_list[0::4] + color_list[2::4] + color_list[1::4] + color_list[3::4]

def load_entity_data(data_root_dir, class_id, obj_id):
    data_path = os.path.join(data_root_dir, "entity", class_id, obj_id+".npz")
    npz_data = np.load(data_path)

    face_points = npz_data["face_points"]

    face_points = torch.FloatTensor(face_points).cuda()

    return face_points

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
    model = RecModule.load_from_checkpoint(args.ckpt_path, config=config)
    model.cuda().eval()
    
    for i in range(len(file_list)):
        class_id = file_list[i][0]
        obj_id = file_list[i][1]

        print("predicting %s %s" % (class_id, obj_id))

        ######################### Get Data ######################
        input_points = load_entity_data(config.dataset.sample_data_root, class_id, obj_id)
        input_points.unsqueeze_(0)
    
        ######################### Fetch Local bases ######################
        with torch.no_grad():
            anchor_pos, anchor_code, anchor_sigma, anchor_rotate = model.forward_enc(input_points)

        ######################### Save Input Points ######################
        input_pts_numpy = input_points[0].cpu().numpy()
        ouput_file = os.path.join(args.output_root_dir, class_id, obj_id, "input_points.ply")
        save_ply(input_pts_numpy, ouput_file)
        
        ######################### Save Predicted Mesh ######################
        # compute sdf grid
        with torch.no_grad():
            sdf_grid, origin, step_size = compute_sdf_grid( \
                model.dsdf_decoder, anchor_pos[0], anchor_code[0], anchor_sigma[0], anchor_rotate[0])
        # to mesh
        verts, faces = sdf2mesh(sdf_grid, origin, step_size)
        # save
        ouput_file = os.path.join(args.output_root_dir, class_id, obj_id+".ply")
        save_mesh(verts, faces, ouput_file)

        ######################### Save Predicted Local Bases ######################
        # sample sdfs
        with torch.no_grad():
            sdf_grid_list, mask_list, origin, step_size = compute_sdf_grid_sep( \
                model.dsdf_decoder, anchor_pos[0], anchor_code[0], anchor_sigma[0], anchor_rotate[0])
        # to mesh
        for grid_id in range(sdf_grid_list.shape[0]):
            # if (sdf_grid_list[grid_id]*mask_list[grid_id] < 0).sum() < 4 or (sdf_grid_list[grid_id]*mask_list[grid_id] > 0).sum() < 4:
            #     continue
            try:
                verts, faces = sdf2mesh(sdf_grid_list[grid_id], origin, step_size, mask_list[grid_id])
            except:
                continue

            center = verts.mean(axis=0)
            offset = center * 0.1
            verts = verts + offset

            color = color_list[grid_id]
            colors = np.zeros_like(verts) + color
            # save
            ouput_file = os.path.join(args.output_root_dir, class_id, obj_id, "sep_parts", "sep_"+str(grid_id)+".ply")
            save_mesh(verts, faces, ouput_file, colors)