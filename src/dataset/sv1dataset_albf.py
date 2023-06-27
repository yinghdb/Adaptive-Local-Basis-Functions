import torch
import torch.utils.data
import numpy as np
import os

def sample_points(points, num):
    np.random.shuffle(points)
    while (points.shape[0] < num):
        points = np.tile(points, (2, 1))
    points = points[:num]
    return points

def load_data(data_root_dir, class_id, obj_id, render_id, uniform_num, nsurface_num, is_encode, observation_dir):
    data_path_observ = os.path.join(data_root_dir, observation_dir, class_id, obj_id, render_id+".npz")
    data_path_entity = os.path.join(data_root_dir, "entity", class_id, obj_id+".npz")
    npz_data_observ = np.load(data_path_observ)
    if is_encode:
        nonface_sdfs_vis = sample_points(npz_data_observ['nonface_sdfs_vis'], 4096)
        missing_points = npz_data_observ['missing_points']
        nonface_sdfs = None
    else:
        npz_data_entity = np.load(data_path_entity)
        nonface_sdfs_vis = None
        missing_points = None
        uniform_sdfs = sample_points(npz_data_entity["uniform_sdfs"], uniform_num)
        nsurface_sdfs = sample_points(npz_data_entity["nsurface_sdfs"], nsurface_num)
        nonface_sdfs = np.concatenate([uniform_sdfs, nsurface_sdfs], axis=0)
        nonface_sdfs = sample_points(nonface_sdfs, uniform_num+nsurface_num)
    input_points = npz_data_observ['input_points']

    return input_points, missing_points, nonface_sdfs_vis, nonface_sdfs

class SV1Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, observe_list, is_encode, observation_dir="observation", transforms=None):
        self.sample_data_root = cfg.sample_data_root
        self.observe_list = observe_list
        self.uniform_samples = cfg.uniform_samples
        self.nsurface_samples = cfg.nsurface_samples
        self.is_encode = is_encode
        self.transforms = transforms
        self.observation_dir = observation_dir

    def __len__(self):
        return len(self.observe_list)

    def __getitem__(self, index):
        class_id = self.observe_list[index][0]
        obj_id = self.observe_list[index][1]
        render_id = self.observe_list[index][2]

        input_points, missing_points, nonface_sdfs_vis, nonface_sdfs = \
            load_data(self.sample_data_root, class_id, obj_id, render_id, self.uniform_samples, self.nsurface_samples, self.is_encode, self.observation_dir)

        name = f"{class_id}_{obj_id}_{render_id}"

        if (self.is_encode):
            if self.transforms is not None:
                input_points_num = input_points.shape[0]
                face_points = np.concatenate([input_points, missing_points], axis=0)
                sdf_points = nonface_sdfs_vis[:, :3]
                sdfs = nonface_sdfs_vis[:, 3:]
                face_points, sdf_points, _, sdfs = self.transforms(face_points, sdf_points, None, sdfs)

                input_points = face_points[:input_points_num]
                missing_points = face_points[input_points_num:]
                nonface_sdfs_vis = np.concatenate([sdf_points, sdfs], axis=1)
            input_points = torch.FloatTensor(input_points)
            missing_points = torch.FloatTensor(missing_points)
            nonface_sdfs_vis = torch.FloatTensor(nonface_sdfs_vis)
            data = {
                'input_points': input_points,
                'missing_points': missing_points,
                'nonface_sdfs_vis': nonface_sdfs_vis,
                'name': name
            }
        else:
            if self.transforms is not None:
                sdf_points = nonface_sdfs[:, :3]
                sdfs = nonface_sdfs[:, 3:]
                input_points, sdf_points, _, sdfs = self.transforms(input_points, sdf_points, None, sdfs)

                nonface_sdfs = np.concatenate([sdf_points, sdfs], axis=1)
            input_points = torch.FloatTensor(input_points)
            nonface_sdfs = torch.FloatTensor(nonface_sdfs)
            data = {
                'input_points': input_points,
                'nonface_sdfs': nonface_sdfs,
                'name': name
            }

        return data
