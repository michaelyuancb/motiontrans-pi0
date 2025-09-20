# bash scripts_exp/eval.sh

exp_name="0712_pi0_eval_lrsc"                          # the description of the experiment target
repo_id="0703_pi_cotrain"
dataset_path="zarr_data/zarr_data_human|zarr_data/zarr_data_robot"  # link different folders with |
policy_dir="Put YOUR_CKPT_PATH Here"
max_token_len=100

logging_time=$(date "+%d-%H.%M.%S")
now_seconds="${logging_time: -8}"
now_date=$(date "+%Y.%m.%d")
num_devices=1
single_val_batch_size=24
val_batch_size=$((num_devices * single_val_batch_size))
echo val_batch_size $val_batch_size

checkpoint_base_dir="checkpoints_pi0/pretrained_ckpts"
assets_base_dir="checkpoints_pi0/assets"
export HF_HOME="/cephfs/shared/yuanchengbo/hub/huggingface"
export OPENPI_DATA_HOME="checkpoints_pi0/openpi"
export LEROBOT_HOME="checkpoints_pi0/lerobot"                                             
# export WANDB_BASE_URL=https://api.bandw.top

uv run scripts/eval.py pi0_droid_motiontrans \
--exp-name=${exp_name} \
--single_arm \
--checkpoint_base_dir=${checkpoint_base_dir} \
--policy_dir=${policy_dir} \
--assets_base_dir=${assets_base_dir} \
--repo_id=${repo_id} \
--dataset_path=${dataset_path} \
--state_down_sample_steps 2 \
--action_down_sample_steps 2 \
--proprioception_rep "relative" \
--action_rep "relative" \
--use_val_dataset \
--val_batch_size=$val_batch_size \
--model.max_token_len ${max_token_len} \
