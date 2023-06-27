import sys; sys.path.append("./")
import os
import argparse
import numpy as np
import trimesh
import torch
from mesh_to_sdf import mesh_to_sdf, get_surface_point_cloud

os.environ['PYOPENGL_PLATFORM'] = 'egl'

parser = argparse.ArgumentParser()
parser.add_argument('--class_ids', type=str, default="")
parser.add_argument('--observe_point_root_dir', type=str, default="")
parser.add_argument('--data_build_dir', type=str, default="")
parser.add_argument('--dataset_root_dir', type=str, default="")
parser.add_argument('--n_samples_target', type=int, default=50000, help='Number of sampling target points')
parser.add_argument('--points_uniform_ratio', type=float, default=0.2, help='Ratio of points to sample uniformly')
parser.add_argument('--bbox_padding', type=float, default=0.2, help='Padding for bounding box')
parser.add_argument('--max_dist', type=float, default=0.02, help='Max distance for sampling points near surface')

args = parser.parse_args()

def sample_nonsurface(mesh, count, uniform_ratio, box_padding, max_dist):
    n_points_uniform = int(count * uniform_ratio)
    n_points_surface = count - n_points_uniform

    boxsize = 1 + box_padding
    points_uniform = np.random.rand(n_points_uniform, 3)
    points_uniform = boxsize * (points_uniform - 0.5)

    points_surface, _ = trimesh.sample.sample_surface(mesh, n_points_surface)
    points_surface += max_dist * (np.random.rand(n_points_surface, 3)*2 - 1.0)
    points = np.concatenate([points_uniform, points_surface], axis=0)

    return points, n_points_uniform

def select_by_radius(nonsurface_points, nonsurface_sdfs, input_points, radius):
    nonsurface_points_torch = torch.FloatTensor(nonsurface_points).cuda()
    input_points_torch = torch.FloatTensor(input_points).cuda()
    dist = (nonsurface_points_torch[:, None] - input_points_torch[None, :])
    dist = dist.pow(2).sum(dim=-1)
    min_dist = torch.min(dist, axis=-1)[0]
    if_select = (min_dist < (radius**2)).cpu().numpy()
    
    nonsurface_sdfs = np.concatenate([nonsurface_points, nonsurface_sdfs[:, np.newaxis]], axis=-1)
    selected_nonface_sdfs = nonsurface_sdfs[if_select]

    return selected_nonface_sdfs

if __name__ == "__main__":
    # data list 
    class_ids = args.class_ids.split(',')
    for class_id in class_ids:
        class_dir = os.path.join(args.data_build_dir, class_id, "4_watertight_scaled")
        for obj_id in os.listdir(class_dir):
            obj_id = obj_id.split('.')[0]
            input_mesh = os.path.join(class_dir, obj_id+".off")
            mesh = trimesh.load(input_mesh)
            print(f"Sample target sdfs {input_mesh}")
            
            # target sdfs for mesh
            output_dir = os.path.join(args.dataset_root_dir, "entity", class_id)
            output_npz = os.path.join(output_dir, obj_id+".npz")
            os.makedirs(output_dir, exist_ok=True)

            points, n_points_uniform = sample_nonsurface(mesh, args.n_samples_target, args.points_uniform_ratio, args.bbox_padding, args.max_dist)
            sdfs = mesh_to_sdf(mesh, points, 
                surface_point_method='sample',
                sign_method='normal',
                bounding_radius=None,
                sample_point_count=10000000,
                normal_sample_count=11)
            uniform_points = points[:n_points_uniform]
            uniform_sdfs = sdfs[:n_points_uniform][:, np.newaxis]
            near_surface_points = points[n_points_uniform:]
            near_surface_sdfs = sdfs[n_points_uniform:][:, np.newaxis]

            uniform_sdfs = np.concatenate([uniform_points, uniform_sdfs], axis=-1)
            nsurface_sdfs = np.concatenate([near_surface_points, near_surface_sdfs], axis=-1)

            face_points, face_ids = trimesh.sample.sample_surface(mesh, 2048)
            face_normals = mesh.face_normals[face_ids]

            np.savez(output_npz, uniform_sdfs=uniform_sdfs, nsurface_sdfs=nsurface_sdfs, \
                    face_points=face_points, face_normals=face_normals)
            
            # for observation data
            observe_obj_dir = os.path.join(args.observe_point_root_dir, class_id, obj_id)
            nonsurface_vis_obj_dir = os.path.join(args.dataset_root_dir, "observation", class_id, obj_id)
            os.makedirs(nonsurface_vis_obj_dir, exist_ok=True)
            for render_id in os.listdir(observe_obj_dir):
                render_id = render_id.split('.')[0]
                observe_data = np.load(os.path.join(observe_obj_dir, render_id+".npz"))
                input_points = observe_data["input_points"]
                missing_points = observe_data["missing_points"]

                selected_nonface_sdfs = select_by_radius(points, sdfs, input_points, args.max_dist)
                output_file = os.path.join(nonsurface_vis_obj_dir, render_id+".npz")
                np.savez(output_file, input_points=input_points, missing_points=missing_points, nonface_sdfs_vis=selected_nonface_sdfs)