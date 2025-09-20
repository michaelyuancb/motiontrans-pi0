# bash scripts_exp/get_normalize_cotrain.sh

repo_id="0703_pi_cotrain"
dataset_path="zarr_data/zarr_data_human|zarr_data/zarr_data_robot"  # link different folders with |

checkpoint_base_dir="checkpoints_pi0/pretrained_ckpts"
assets_base_dir="checkpoints_pi0/assets"
export HF_HOME="/cephfs/shared/yuanchengbo/hub/huggingface"
export OPENPI_DATA_HOME="checkpoints_pi0/openpi"
export LEROBOT_HOME="checkpoints_pi0/lerobot"                                             
export CUDA_VISIBLE_DEVICES=0
exp_name="default"               # not used

# we downsample data-obs and action from 20 Hz to 10 Hz, since pi0 inference only support for 10Hz inference speed. 

uv run scripts/compute_norm_stats.py pi0_droid_motiontrans \
--exp_name=${exp_name} \
--single_arm \
--checkpoint_base_dir=${checkpoint_base_dir} \
--assets_base_dir=${assets_base_dir} \
--repo_id=${repo_id} \
--dataset_path=${dataset_path} \
--state_down_sample_steps 2 \
--action_down_sample_steps 2 \
--proprioception_rep "relative" \
--action_rep "relative" \
--use_val_dataset --create_train_val_split --val_ratio=0.025 \
--compute_norm_stats \
--no_wandb_enabled \

# image_down_sample_steps / state_down_sample_steps: list, get idx - steps[i] for i in down_sample_steps.
# action_down_sample_steps: int, get idx + i * action_down_sample_steps for i in range(action_horizon)


