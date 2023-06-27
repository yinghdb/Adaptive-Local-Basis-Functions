# class_ids="03001627,02691156,02933112,03636649,04256520,04379243,04530566,02958343"
class_ids="03001627"
data_build_dir="/mnt/data1/yinghui/Datasets/ShapeNetBuild.v1"
dataset_root_dir="/mnt/data2/yinghui/Datasets/ShapeNetALBF.test"
depth_root_dir=$dataset_root_dir+"/depth"
observe_point_root_dir=$dataset_root_dir+"/observed_points"
render_num=16

python data_preprocessing/render_depth_images.py \
    --class_ids=$class_ids \
    --data_build_dir=$data_build_dir \
    --depth_root_dir=$depth_root_dir \
    --n_images_per_mesh=$render_num

python data_preprocessing/sample_observation_points.py \
    --class_ids=$class_ids \
    --depth_root_dir=$depth_root_dir \
    --observe_point_root_dir=$observe_point_root_dir \
    --data_build_dir=$data_build_dir

python data_preprocessing/build_dataset.py \
    --class_ids=$class_ids \
    --observe_point_root_dir=$observe_point_root_dir \
    --data_build_dir=$data_build_dir \
    --dataset_root_dir=$dataset_root_dir