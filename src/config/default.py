from yacs.config import CfgNode as CN
_CN = CN()

_CN.name = "ALBF"

_CN.dataset = CN()
_CN.dataset.sample_data_root = "../../Datasets/ShapeNetALBF.v1"
_CN.dataset.filelist_root = "./data/filelists"
_CN.dataset.train_file_path = ""
_CN.dataset.val_file_path = ""
_CN.dataset.observation_dir = "observation"
_CN.dataset.render_num = 16
_CN.dataset.class_list = "04256520,03001627"
_CN.dataset.samples_per_obj = 2048
_CN.dataset.uniform_samples = 4096
_CN.dataset.nsurface_samples = 16384
_CN.dataset.data_aug = False

_CN.local_encoder = CN()
_CN.local_encoder.embed_dim = 256
_CN.local_encoder.sample_stages = [512, 128]
_CN.local_encoder.dropout = 0.0
_CN.local_encoder.norm = "GroupNorm"
_CN.local_encoder.knn = 16
_CN.local_encoder.drop_count = 0
_CN.local_encoder.trans_type = "coord_only"

_CN.transformer_global = CN()
_CN.transformer_global.num_blocks = 3
_CN.transformer_global.num_heads = 8
_CN.transformer_global.attn_pdrop = 0.0
_CN.transformer_global.resid_pdrop = 0.0
_CN.transformer_global.num_query_per_obj = 32
_CN.transformer_global.wave_pos_embed = True

_CN.transformer_local = CN()
_CN.transformer_local.num_blocks = 6
_CN.transformer_local.num_heads = 8
_CN.transformer_local.attn_pdrop = 0.0
_CN.transformer_local.resid_pdrop = 0.0
_CN.transformer_local.wave_pos_embed = True

_CN.local_decoder = CN()
_CN.local_decoder.dims = [ 512, 512, 512, 512, 512, 512, 512, 512 ]
_CN.local_decoder.dropouts = None
_CN.local_decoder.dropout_prob = 0.0
_CN.local_decoder.skip_in = [4]
_CN.local_decoder.geometric_init = True
_CN.local_decoder.radius_init = 1.0
_CN.local_decoder.beta = 100

_CN.sdf = CN()
_CN.sdf.samples_per_anchor = 128
_CN.sdf.sample_radius = 0.1
_CN.sdf.global_query_k = 2
_CN.sdf.clamp_dist = 0.05

_CN.loss = CN()
_CN.loss.sdf_vis_lambda = 1.0
_CN.loss.chamfer_lambda = 0.1
_CN.loss.sdf_global_lambda = 1.0
_CN.loss.sdf_global_geo_lambda = 1.0
_CN.loss.smooth_lambda = 1.0
_CN.loss.offreg_lambda = 0.01

_CN.optimizer = CN()
_CN.optimizer.type = "adam"
_CN.optimizer.lr = 0.005
_CN.optimizer.weight_decay = 0.0
_CN.optimizer.lr_rate = 1.0

_CN.scheduler = CN()
_CN.scheduler.interval = "epoch"
_CN.scheduler.type = "MultiStepLR"
_CN.scheduler.step_size = 15
_CN.scheduler.gamma = 0.5
_CN.scheduler.lowest_gamma = 0.02
_CN.scheduler.mslr_milestones = [15, 22, 27]
_CN.scheduler.cosa_tmax = None
_CN.scheduler.warmup_step = 0

_CN.trainer = CN()
_CN.trainer.max_epochs = 30
_CN.trainer.world_size = 1

# reproducibility
_CN.trainer.seed = 66

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()