import sys; sys.path.append("./")
import os
import argparse
import numpy as np
import trimesh
import torch
from mesh_to_sdf import mesh_to_sdf, get_surface_point_cloud
from src.utils.data import depth2pc
from src.utils.ops import furthest_point_sample

os.environ['PYOPENGL_PLATFORM'] = 'egl'

parser = argparse.ArgumentParser()
parser.add_argument('--class_ids', type=str, default="")
parser.add_argument('--depth_root_dir', type=str, default="")
parser.add_argument('--observe_point_root_dir', type=str, default="")
parser.add_argument('--data_build_dir', type=str, default="")
parser.add_argument('--n_samples_input', type=int, default=2048, help='Number of sampling input points')
parser.add_argument('--n_samples_missing', type=int, default=128, help='Number of sampling missing points')

args = parser.parse_args()

def load_depth_data(depth_root_dir, class_id, obj_id, render_id):
    data_path = os.path.join(depth_root_dir, class_id, obj_id, render_id+".npz")
    npz_data = np.load(data_path)
    
    depth = npz_data['depth']
    valid = depth > 1e-6
    cam_ex = npz_data['cam2world']
    cam_in = npz_data['projection']

    depth = torch.FloatTensor(depth).cuda()
    cam_ex = torch.FloatTensor(cam_ex).cuda()
    cam_in = torch.FloatTensor(cam_in).cuda()
    valid = torch.BoolTensor(valid).cuda()

    return depth, cam_ex, cam_in, valid

def sample_input_points(depth_root_dir, class_id, obj_id, render_id):
    depth, cam_ex, cam_in, valid = load_depth_data(depth_root_dir, class_id, obj_id, render_id)
    pt_pos, _ = depth2pc(depth, cam_ex, cam_in, valid)
    fps_idx = furthest_point_sample(pt_pos.unsqueeze(0), args.n_samples_input)[0]
    input_points = pt_pos[fps_idx.long()]

    return input_points

def sample_missing_points(surface_points, input_points):
    fps_idx = furthest_point_sample(surface_points.unsqueeze(0), args.n_samples_missing, input_points.unsqueeze(0))[0]
    missing_points = surface_points[fps_idx.long()]

    return missing_points


if __name__ == "__main__":
    class_ids = args.class_ids.split(',')
    # data list 
    for class_id in class_ids:
        class_dir = os.path.join(args.data_build_dir, class_id, "4_watertight_scaled")
        for obj_id in os.listdir(class_dir):
            obj_id = obj_id.split('.')[0]
            input_mesh = os.path.join(class_dir, obj_id+".off")
            mesh = trimesh.load(input_mesh)
            points_surface, _ = trimesh.sample.sample_surface(mesh, 2048)

            # for observation data
            depth_obj_dir = os.path.join(args.depth_root_dir, class_id, obj_id)
            output_dir = os.path.join(args.observe_point_root_dir, class_id, obj_id)
            os.makedirs(output_dir, exist_ok=True)
            for render_id in os.listdir(depth_obj_dir):
                render_id = render_id.split('.')[0]
                print(f"Sample observe points {class_id} {obj_id} {render_id}")
                input_points = sample_input_points(args.depth_root_dir, class_id, obj_id, render_id)
                missing_points = sample_missing_points(torch.FloatTensor(points_surface).cuda(), input_points)

                output_file = os.path.join(output_dir, render_id+".npz")
                np.savez(output_file, input_points=input_points.cpu().numpy(), missing_points=missing_points.cpu().numpy())