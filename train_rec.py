# import sys; sys.path.append("./")
import os
import torch
import argparse
import pprint
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
# torch.multiprocessing.set_start_method('spawn')

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

from src.config.default import get_cfg_defaults
from src.lightning_module.albf_module_rec import RecModule
from src.dataset.sv1datamodule_rec import DataModule

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' 


def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'cfg_path', type=str, help='config path')
    parser.add_argument(
        '--exp_name', type=str, default='default_exp_name')
    parser.add_argument(
        '--batch_size', type=int, default=8, help='batch_size per gpu')
    parser.add_argument(
        '--lr', type=float, default=0.0, help='learning rate')
    parser.add_argument(
        '--num_workers', type=int, default=8)
    parser.add_argument(
        '--ckpt_path', type=str, default=None,
        help='pretrained checkpoint path')
    parser.add_argument(
        '--debug', action='store_true'
    )
    parser.add_argument(
        '--pin_memory', action='store_true')

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()

def main():
    # parse arguments
    args = parse_args()
    
    config = get_cfg_defaults()
    config.merge_from_file(args.cfg_path)
    if args.lr != 0:
        config.optimizer.lr = args.lr
    pl.seed_everything(config.trainer.seed)  # reproducibility
    args.max_epochs = config.trainer.max_epochs

    rank_zero_only(pprint.pprint)(vars(args))

    # lightning data
    data_module = DataModule(args, config.dataset)

    # lightning module
    model = RecModule(config)

    # TensorBoard Logger
    logger = TensorBoardLogger(save_dir='logs/train_logs', name=args.exp_name, default_hp_metric=False)
    # model._setup_figdir(logger.log_dir)
    ckpt_dir = os.path.join(logger.log_dir, 'checkpoints')

    # Callbacks
    ckpt_callback = ModelCheckpoint(save_last=True, dirpath=str(ckpt_dir), monitor='sdf_loss', save_top_k=1, mode='min', filename='{epoch}-{sdf_loss:.8f}')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor, ckpt_callback]

    # Lightning Trainer
    if args.debug or args.devices == 1:
        trainer = pl.Trainer.from_argparse_args(
            args,
            callbacks=callbacks,
            logger=logger,
            accelerator="gpu",
            devices=1
        )
    else:
        trainer = pl.Trainer.from_argparse_args(
            args,
            callbacks=callbacks,
            logger=logger,
            accelerator="gpu",
            strategy=DDPStrategy(find_unused_parameters=True),
        )
    trainer.fit(model, datamodule=data_module)

if __name__ == '__main__':
    main()
