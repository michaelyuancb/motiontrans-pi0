# bash scripts_exp/train_cotrain.sh

exp_name="default"                           # the description of the experiment target
repo_id="0703_pi_cotrain"                    # the repo used for dataset and norm-stat
dataset_path="zarr_data/zarr_data_human|zarr_data/zarr_data_robot"  # link different folders with |

checkpoint_base_dir="checkpoints_pi0/pretrained_ckpts"
assets_base_dir="checkpoints_pi0/assets"
export HF_HOME="/cephfs/shared/yuanchengbo/hub/huggingface"
export OPENPI_DATA_HOME="checkpoints_pi0/openpi"
export LEROBOT_HOME="checkpoints_pi0/lerobot"                                             

logging_time=$(date "+%d-%H.%M.%S")
now_seconds="${logging_time: -8}"
now_date=$(date "+%Y.%m.%d")

alpha=0.5
proprioception_droprate=0.0
num_devices=8
single_batch_size=24
batch_size=$((num_devices * single_batch_size))
echo batch_size $batch_size

num_train_steps=150001
keep_period=75000
log_interval=250
save_interval=75000
val_interval=5000
max_token_len=150

single_val_batch_size=24
val_batch_size=$((num_devices * single_val_batch_size))
echo val_batch_size $val_batch_size

# we downsample data-obs and action from 20 Hz to 10 Hz, since pi0 inference only support for 10Hz inference speed. 

# ======== pi0 cocktail  =====
# WANDB_DISABLED=True 
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi0_droid_motiontrans \
--exp-name="${now_date}_${now_seconds}_${repo_id}_${exp_name}" \
--alpha=${alpha} \
--single_arm \
--checkpoint_base_dir=${checkpoint_base_dir} \
--assets_base_dir=${assets_base_dir} \
--batch-size=$batch_size \
--repo_id=${repo_id} \
--dataset_path=${dataset_path} \
--state_down_sample_steps 2 \
--action_down_sample_steps 2 \
--proprioception_rep "relative" \
--action_rep "relative" \
--proprioception_droprate ${proprioception_droprate} \
--use_val_dataset \
--val_batch_size=$val_batch_size \
--num_train_steps ${num_train_steps} \
--keep_period ${keep_period} \
--log_interval ${log_interval} \
--save_interval ${save_interval} \
--val_interval ${val_interval} \
--model.max_token_len ${max_token_len} \
