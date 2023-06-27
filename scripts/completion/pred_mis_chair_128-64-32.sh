cfg_path="configs/completion/chair_128-64-32.py"

ckpt_path="./logs/train_logs/mis_chair_128-64-32-bs=16-lr=0.0005/version_1/checkpoints/last.ckpt"
pred_list="./data/vis/chair.lst"
output_root_dir="./experiments/complete_chair_128-64-32"

CUDA_VISIBLE_DEVICES=0 \
python -u ./src/tools/pred_mis.py \
    ${cfg_path} \
    --ckpt_path=${ckpt_path} \
    --pred_list=${pred_list} \
    --output_root_dir=${output_root_dir}
