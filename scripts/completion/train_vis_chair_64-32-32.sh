cfg_path="configs/completion/chair_64-32-32.py"
devices=2
batch_size=8
lr=0.0005
exp_name="vis_chair_64-32-32-bs=$(($batch_size * $devices))-lr=$lr"

# LOG="./logs/echo_logs/${exp_name}.txt"
# exec &> >(tee "$LOG")

CUDA_VISIBLE_DEVICES=2,3 \
python -u ./train_vis.py \
    ${cfg_path} \
    --exp_name=${exp_name} \
    --devices=${devices} \
    --batch_size=${batch_size} \
    --lr $lr \
    --check_val_every_n_epoch=1 \
    --log_every_n_steps=1000