import etils.epath as epath
import jax
import jax.experimental
import jax.numpy as jnp
import copy
import numpy as np
import tqdm
import cv2
import datetime

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms
import openpi.models.model as _model
from serve_policy import Args, Checkpoint, Default, EnvMode, create_eval_policy


def get_tp_fn_fp_tn(to_act, gt_to_act):
    tp = np.logical_and(to_act, gt_to_act).sum()
    fn = np.logical_and(~to_act, gt_to_act).sum()
    fp = np.logical_and(to_act, ~gt_to_act).sum()
    tn = np.logical_and(~to_act, ~gt_to_act).sum()
    return tp, fn, fp, tn


def shift_and_pad(tokenized_prompt, first_one_indices):
    B, T = tokenized_prompt.shape
    batch_indices = jnp.arange(B)[:, None]  # (B, 1)
    positions = jnp.arange(T)[None, :]       # (1, T)
    f = first_one_indices[:, None]           # (B, 1)
    L = T - f                                # (B, 1)
    valid_mask = positions < L               # (B, T)
    source_indices = f + positions           # (B, T)
    shifted = jnp.where(valid_mask, tokenized_prompt[batch_indices, source_indices], 0)
    return shifted


def create_args(config, policy_dir) -> Args:
    # return Args(
    #     env=EnvMode.FAST_BASE,
    #     policy=Default(),
    # )
    return Args(
        policy=Checkpoint(config=config, dir=policy_dir),
    )


def calc_mse(predictions, targets, mask):
    squared_error = np.square(predictions - targets)
    mse = np.sum(squared_error * mask[:, None, None]) / (np.sum(mask) * targets.shape[1] * targets.shape[2])
    return mse


def compute_error(predictions,
                  targets,
                  mask,
                  ):
    pos0_mse = calc_mse(predictions[:,:,0:3], targets[:,:,0:3], mask)
    rot0_mse = calc_mse(predictions[:,:,3:9], targets[:,:,3:9], mask)
    gripper0_mse = calc_mse(predictions[:,:,9:15], targets[:,:,9:15], mask)
    pos1_mse = calc_mse(predictions[:,:,15:18], targets[:,:,15:18], mask)
    rot1_mse = calc_mse(predictions[:,:,18:24], targets[:,:,18:24], mask)
    gripper1_mse = calc_mse(predictions[:,:,24:30], targets[:,:,24:30], mask)

    results = {
        "pos0": pos0_mse,
        "rot0": rot0_mse,
        "gripper0": gripper0_mse,
        "pos1": pos1_mse,
        "rot1": rot1_mse,
        "gripper1": gripper1_mse,
    }
    
    return results

def main(config: _config.TrainConfig):
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    _, val_data_loader = _data_loader.create_data_loader(
        config,
        num_workers=config.num_workers,
        shuffle=False,
    )
    assert val_data_loader is not None, "Validation data loader is None"

    total_error = 0.0
    num_batches = 0
    num_action_loss_fraction = []
    
    args = create_args(config.name, config.policy_dir)
    policy = create_eval_policy(args, config)
    _output_transform_list = copy.deepcopy(policy._output_transform.transforms)
    # we need to filter out the ExtractFASTActions transform
    # since for data loader, the 'action' key is real action, not tokens
    _output_transform_list = [x for x in _output_transform_list if not isinstance(x, transforms.ExtractFASTActions)]
    _output_transform = transforms.CompositeTransform(_output_transform_list)
            

    def target_transform(obs: _model.Observation,
                         targets: _model.Actions) -> dict: 
        _inputs = jax.tree.map(lambda x: x, obs)
        _targets = jax.tree.map(lambda x: x, targets)
        _target_outputs = {
            "state": _inputs.state,
            "actions": _targets,
        }
        _target_outputs['tokenized_suffix'] = _inputs.tokenized_prompt
        _target_outputs = jax.tree.map(np.asarray, _target_outputs)
        action_list = []
        thought_list = []
        for i in range(_target_outputs['actions'].shape[0]):
            this_output = jax.tree.map(lambda x: np.asarray(x[i]), _target_outputs)
            this_transformed = _output_transform(this_output)
            action_list.append(this_transformed['actions'])
        ret_dict = {'actions': np.asarray(action_list)}
        return ret_dict
    
    total_error_dict = None
    
    for batch in tqdm.tqdm(val_data_loader, dynamic_ncols=True):
        obs, targets = batch
        policy_out = policy.infer(obs)
        actions = policy_out['actions']
        transformed_targets = target_transform(obs, targets)
        transformed_target_actions = transformed_targets['actions']
        action_mask = np.ones(actions.shape[0], dtype=np.bool_)
        results = compute_error(actions, transformed_target_actions, action_mask)
        if total_error_dict is None:
            total_error_dict = dict()
            for key, value in results.items():
                total_error_dict[key] = value
        else:
            for key, value in results.items():
                total_error_dict[key] = total_error_dict[key] + value
        num_batches += 1
        num_action_loss_fraction.append(
            action_mask.sum() / action_mask.shape[0])

    print(f"Policy: {config.policy_dir}")
    for key in total_error_dict.keys():
        average_error = total_error_dict[key] / num_batches
        average_action_fraction = np.mean(num_action_loss_fraction)
        print(f'{key}:  average_error={average_error}  num_action_loss_fraction={average_action_fraction}')

if __name__ == "__main__":
    main(_config.cli())