import numpy as np
import os
import logging

import pytorch_lightning as pl
from torch import distributed as dist
from torch.utils.data import (
    DataLoader,
)

from src.dataset.sv1dataset_albf import SV1Dataset
from .data_utils import Compose, PointCloudRotate, PointCloudScale, PointCloudTranslate, PointCloudJitter

def get_observe_list(root_dir, class_list, stage, render_num):
    observe_list = []
    for class_id in class_list:
        list_path = os.path.join(root_dir, class_id+"_"+stage+".lst")
        if os.path.exists(list_path):
            with open(list_path, 'r') as f:
                lines = f.read().splitlines()
                for line in lines:
                    for render_id in range(render_num):
                        observe_list += [(class_id, line.strip(), "%02d" % (render_id))]
        else:
            logging.getLogger("DataModule").error("File Not Found: %s!", list_path)
    return observe_list

def get_val_observe_list(val_path):
    observe_list = []
    with open(val_path, 'r') as f:
        if os.path.exists(val_path):
            lines = f.read().splitlines()
            for line in lines:
                observe_list += [line.strip().split(" ")]
        else:
            logging.getLogger("DataModule").error("File Not Found: %s!", val_path)
    return observe_list

class DataModule(pl.LightningDataModule):
    def __init__(self, args, cfg, is_encode):
        super().__init__()
        logging.getLogger("DataModule").setLevel(logging.INFO)

        self.cfg = cfg
        self.is_encode = is_encode
        self.train_loader_params = {
            'batch_size': args.batch_size,
            'shuffle': True,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True),
            'prefetch_factor': args.num_workers * 2
        }
        self.val_loader_params = {
            'batch_size': args.batch_size,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True),
            'prefetch_factor': args.num_workers * 2
        }
        self.test_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 1,
            # 'pin_memory': True
        }
        self.pred_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 1,
            # 'pin_memory': True
        }

    def setup(self, stage=None):
        """
        Setup train / val / test dataset. This method will be called by PL automatically.
        Args:
            stage (str): 'fit' in training phase, and 'test' in testing phase.
        """

        assert stage in ['fit', 'test', 'predict'], "stage must be either fit or test"

        try:
            self.rank = dist.get_rank()
        except AssertionError as ae:
            self.rank = 0
        except RuntimeError as re:
            self.rank = 0

        class_list = self.cfg.class_list.split(',')
        if stage == 'fit':
            transforms = Compose(
                [
                    PointCloudRotate(),
                    PointCloudScale(), 
                    PointCloudTranslate(),
                    PointCloudJitter()
                ]
            )
            observe_list_train = get_observe_list(self.cfg.filelist_root, class_list, "train", self.cfg.render_num)
            # random.shuffle(observe_list_train)
            observe_list_val = get_val_observe_list(self.cfg.val_file_path)
            self.train_dataset = SV1Dataset(
                                        self.cfg, 
                                        observe_list_train, 
                                        self.is_encode,
                                        transforms=transforms if self.cfg.data_aug else None)
            self.val_dataset = SV1Dataset(self.cfg, observe_list_val, self.is_encode)
            logging.getLogger("DataModule").info('[rank: %d] Train & Val Dataset loaded!', self.rank)
        elif stage == 'predict':
            observe_list_pred = get_observe_list(self.cfg.filelist_root, class_list, "pred", 8)
            self.pred_dataset = SV1Dataset(self.cfg, observe_list_pred, self.is_encode)
            logging.getLogger("DataModule").info('[rank: %d] Predict Dataset loaded!', self.rank)
        else:  # stage == 'test'
            observe_list_test = get_val_observe_list(self.cfg.val_file_path)
            self.test_dataset = SV1Dataset(self.cfg, observe_list_test, self.is_encode)
            logging.getLogger("DataModule").info('[rank: %d] Test Dataset loaded!', self.rank)

    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset, **self.train_loader_params)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.val_dataset, **self.val_loader_params)
        return dataloader

    def test_dataloader(self, *args, **kwargs):
        dataloader = DataLoader(self.test_dataset, **self.test_loader_params)
        return dataloader

    def predict_dataloader(self, *args, **kwargs):
        dataloader = DataLoader(self.pred_dataset, **self.pred_loader_params)
        return dataloader
