## Environment

The code was tested with python 3.8 and pytorch 1.9 on linux. 

If you use a system pip installation, run `pip install -r requirements.txt`

Then build pytorch extensions
```
# PointNet++
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
# Customized Extensions
python setup.py build_ext --inplace
```

## Datasets

We use ShapeNet as our training and testing dataset.

To build the dataset, you need to follow [OccNet repository](https://github.com/autonomousvision/occupancy_networks) to fetch watertight meshes first.

And then, run `bash scripts/data_preprocessing/data_prepare.sh`. Inside the script file, you need to fill in your own data path.

## Training

There are config files in `configs` folder, with which you can customize your own settings. An important note is that you need to set `_CN.dataset.sample_data_root` in the config file as your own dataset path.

After setting up the config file, you can start training. Here are some sample scripts, note that inside the `train_mis_chair_128-32-16.sh`, you will need to set `vis_ckpt_path` as the path to the trained checkpoint file of the visible training phase.:

For shape completion:
```
# Training for visible regions
bash scripts/completion/train_vis_chair_128-32-16.sh
# Training for missing regions
bash scripts/completion/train_mis_chair_128-32-16.sh
```


For shape reconstruction:



## Prediction

We offer basic scripts for running prediction. Within the scripts, you need to set `ckpt_path` and `pred_list` to provide the path to the trained checkpoint and the path to the file of predicting list.

Run `bash scripts/completion/pred_vis_chair_128-64-32.sh` for encoding phase predicting

Run `bash scripts/completion/pred_mis_chair_128-64-32.sh` for decoding phase predicting
