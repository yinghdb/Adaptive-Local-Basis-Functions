import numpy as np
import os
import logging

import pytorch_lightning as pl
from torch import distributed as dist
from torch.utils.data import (
    DataLoader,
)
from .data_utils import Compose, PointCloudRotate, PointCloudScale, PointCloudTranslate, PointCloudJitter
from .sv1dataset_rec import Dataset_Rec

def get_entity_list(root_dir, class_list, stage):
    entity = []
    for class_id in class_list:
        list_path = os.path.join(root_dir, class_id+"_"+stage+".lst")
        if os.path.exists(list_path):
            with open(list_path, 'r') as f:
                lines = f.read().splitlines()
                for line in lines:
                    entity += [(class_id, line.strip())]
        else:
            logging.getLogger("DataModule").error("File Not Found: %s!", list_path)
    return entity

class DataModule(pl.LightningDataModule):
    def __init__(self, args, cfg):
        super().__init__()
        logging.getLogger("DataModule").setLevel(logging.INFO)

        self.cfg = cfg
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
        Setup train / val / test / pred dataset. This method will be called by PL automatically.
        Args:
            stage (str): 'fit' in training phase, and 'test' in testing phase.
        """

        assert stage in ['fit', 'test', 'predict'], "stage must be either fit, test and predict"

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
            train_list = get_entity_list(self.cfg.filelist_root, class_list, "train")
            val_list = get_entity_list(self.cfg.filelist_root, class_list, "test")
            self.train_dataset = Dataset_Rec(
                                    self.cfg.sample_data_root, 
                                    self.cfg.uniform_samples,
                                    self.cfg.nsurface_samples,
                                    train_list,
                                    transforms=transforms if self.cfg.data_aug else None)
            self.val_dataset = Dataset_Rec(
                                    self.cfg.sample_data_root, 
                                    self.cfg.uniform_samples,
                                    self.cfg.nsurface_samples,
                                    val_list)
            logging.getLogger("DataModule").info('[rank: %d] Train & Val Dataset loaded!', self.rank)
        elif stage == 'predict':
            pred_list = get_entity_list(self.cfg.filelist_root, class_list, "test")
            self.pred_dataset = Dataset_Rec(
                                    self.cfg.sample_data_root, 
                                    self.cfg.uniform_samples,
                                    self.cfg.nsurface_samples,
                                    pred_list)
            logging.getLogger("DataModule").info('[rank: %d] Predict Dataset loaded!', self.rank)
        else:  # stage == 'test'
            test_list = get_entity_list(self.cfg.filelist_root, class_list, "test")
            self.test_dataset = Dataset_Rec(
                                    self.cfg.sample_data_root, 
                                    self.cfg.uniform_samples,
                                    self.cfg.nsurface_samples,
                                    test_list)
            logging.getLogger("DataModule").info('[rank: %d] Test Dataset loaded!', self.rank)

    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset, **self.train_loader_params)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.val_dataset, **self.val_loader_params)
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(self.test_dataset, **self.test_loader_params)
        return dataloader

    def predict_dataloader(self):
        dataloader = DataLoader(self.pred_dataset, **self.pred_loader_params)
        return dataloader
