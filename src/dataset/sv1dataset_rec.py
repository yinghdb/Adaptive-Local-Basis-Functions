import torch
import torch.utils.data
import numpy as np
import os
from .data_utils import Compose, PointCloudRotate, PointCloudScale, PointCloudTranslate

def sample_points(points, num):
    np.random.shuffle(points)
    while (points.shape[0] < num):
        points = np.tile(points, (2, 1))
    points = points[:num]
    return points

def load_entity_data(data_root_dir, class_id, obj_id, uniform_num, nsurface_num, transforms):
    data_path = os.path.join(data_root_dir, "entity", class_id, obj_id+".npz")
    npz_data = np.load(data_path)


    face_points = npz_data["face_points"]
    face_normals = npz_data["face_normals"]
    
    uniform_data = npz_data["uniform_sdfs"]
    nsurface_data = npz_data["nsurface_sdfs"]
    uniform_data = sample_points(uniform_data, uniform_num)
    nsurface_data = sample_points(nsurface_data, nsurface_num)

    uniform_points = uniform_data[:, :3]
    uniform_sdfs = uniform_data[:, 3:]
    nsurface_points = nsurface_data[:, :3]
    nsurface_sdfs = nsurface_data[:, 3:]

    points_data = np.concatenate([uniform_points, nsurface_points], axis=0)
    sdfs_data = np.concatenate([uniform_sdfs, nsurface_sdfs], axis=0)
    if transforms is not None:
        face_points, points_data, face_normals, sdfs_data = transforms(face_points, points_data, face_normals, sdfs_data)

    face_points = torch.FloatTensor(face_points)
    face_normals = torch.FloatTensor(face_normals)
    uniform_points = torch.FloatTensor(points_data[:uniform_num])
    uniform_sdfs = torch.FloatTensor(sdfs_data[:uniform_num])
    nsurface_points = torch.FloatTensor(points_data[uniform_num:uniform_num+nsurface_num])
    nsurface_sdfs = torch.FloatTensor(sdfs_data[uniform_num:])

    return face_points, face_normals, uniform_points, uniform_sdfs, nsurface_points, nsurface_sdfs

class Dataset_Rec(torch.utils.data.Dataset):
    def __init__(self, sample_data_root, uniform_samples, nsurface_samples, entity_list, transforms=None):
        self.sample_data_root = sample_data_root
        self.uniform_samples = uniform_samples
        self.nsurface_samples = nsurface_samples
        self.entity_list = entity_list
        self.transforms = transforms

    def __len__(self):
        return len(self.entity_list)

    def __getitem__(self, index):
        class_id = self.entity_list[index][0]
        obj_id = self.entity_list[index][1]

        face_pts, face_norms, uniform_pts, uniform_sdfs, nearface_pts, nearface_sdfs = \
            load_entity_data(self.sample_data_root, class_id, obj_id, self.uniform_samples, self.nsurface_samples, self.transforms)

        name = f"{class_id}_{obj_id}"

        data = {
            'face_pts': face_pts,
            'face_norms': face_norms,
            'uniform_pts': uniform_pts,
            'uniform_sdfs': uniform_sdfs,
            'nearface_pts': nearface_pts,
            'nearface_sdfs': nearface_sdfs,
            'name': name
        }

        return data


if __name__ == "__main__":
    sample_data_root = "~/Datasets/ShapeNetESample.v1"
    uniform_samples = 10
    nsurface_samples = 10
    entity_list = [
        ["03001627", "1c3f1a9cea91359c4c3e19c2c67c262f"],
        ["03001627", "34ed902dfa2d8eec2cafe1b125cab8fb"]
    ]
    transforms = Compose(
        [
            PointCloudRotate(),
            PointCloudScale(), 
            PointCloudTranslate(),
        ]
    )
    dset = Dataset_Rec(sample_data_root, uniform_samples, nsurface_samples, entity_list, transforms)
    print(dset[0])
    print(dset[1])

    dloader = torch.utils.data.DataLoader(dset, batch_size=2, shuffle=False)
    print("end")