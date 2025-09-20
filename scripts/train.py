import dataclasses
import functools
import logging
import platform
from typing import Any
import requests
import socket
import time

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import optax
import tqdm_loggable.auto as tqdm
import wandb
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        # wandb.init(id=run_id, resume="must", project=config.project_name)
        # we may not resume from the same wandb run
        # as the loaded step from checkpoint might be earlier than wandb's step
        # and wandb only supports monotonically increasing step
        wandb_name = config.exp_name + '-resumed'
        wandb.init(
            name=wandb_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def _create_output_transform(config: _config.TrainConfig) -> tuple[_transforms.DataTransformFn, _transforms.DataTransformFn]:
    """Creates the output transforms to be applied to model outputs and targets for validation."""
    # copied from src/openpi/policies/policy_config.py
    data_config = config.data.create(config.assets_dirs, config.model)
    norm_stats = data_config.norm_stats
    output_transform = _transforms.CompositeTransform([
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ])
    
    # we need to filter out the ExtractFASTActions transform
    # since for data loader, the 'action' key is real action, not tokens
    _output_transform_list = output_transform.transforms
    _target_transform_list = [x for x in _output_transform_list if not isinstance(x, _transforms.ExtractFASTActions)]
    target_transform = _transforms.CompositeTransform(_target_transform_list)
    return output_transform, target_transform


@at.typecheck
def infer_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> dict[str, at.Array]:
    """
    Neural network inference for validation.
    The output transforms are not jittable, so they should be applied outside this function.

    Returns:
        outputs: A dictionary to compute MSE: 
        ```
        {
            "state": Array,
            "actions": Array,
            "targets": Array,
            "action_mask": Array,
            "other informatioin from the `model.sample_actions` function"
        }
        ```
    """
    if state.ema_decay is None:
        model = nnx.merge(state.model_def, state.params)
    else:
        model = nnx.merge(state.model_def, state.ema_params)
    model.eval()

    sample_actions = nnx_utils.module_jit(model.sample_actions)

    infer_rng = jax.random.fold_in(rng, state.step)

    observation, targets = batch
    actions = sample_actions(rng=infer_rng, observation=observation)

    # deep copy to avoid inplace modification
    _inputs = jax.tree.map(lambda x: x, observation)
    _targets = jax.tree.map(lambda x: x, targets)
    _action_mask = jnp.ones(actions.shape[0], dtype=jnp.bool_)

    outputs = {
        "state": _inputs.state,
        "actions": actions,
        "targets": _targets,
        "action_mask": _action_mask
    }
    return outputs


@at.typecheck
def compute_mse(
        state: at.Float[at.Array, 'b s'],
        actions: at.Float[at.Array, 'b ah ad'],
        targets: at.Float[at.Array, 'b ah ad'],
        action_mask: at.Bool[at.Array, 'b'],
        output_transform: _transforms.DataTransformFn,
        target_transform: _transforms.DataTransformFn,
        ) -> dict[str, at.ArrayLike]:
    batch_size = state.shape[0]
    errors = []
    for i in range(batch_size):
        state_i = np.asarray(state[i])
        action_i = np.asarray(actions[i])
        target_i = np.asarray(targets[i])
        
        transformed_action_i = output_transform({
            "state": state_i,
            "actions": action_i
        })['actions']
        transformed_target_i = target_transform({
            "state": state_i,
            "actions": target_i
        })['actions']
        errors.append(transformed_target_i - transformed_action_i)
    
    errors = np.asanyarray(errors)
    broadcasted_action_mask = np.broadcast_to(
        action_mask[:, None, None], errors.shape)
    mse = np.mean(errors[broadcasted_action_mask] ** 2)
    return {'action_mse': mse,
            'num_action_loss_fraction': np.sum(action_mask) / batch_size}


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    jax.config.update('jax_threefry_partitionable', False)

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    if config.use_val_dataset:
        val_rng, _ = jax.random.split(train_rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader, val_data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        num_workers=config.num_workers,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    if config.use_val_dataset:
        val_data_iter = iter(val_data_loader)
        val_batch = next(val_data_iter)
        logging.info(f"Initialized validation data loader:\n{training_utils.array_tree_to_info(val_batch)}")
    else:
        val_data_iter = None

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    pval_inference = lambda : None
    if config.use_val_dataset:
        pval_inference = jax.jit(
            functools.partial(infer_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )
        output_transform, target_transform = _create_output_transform(config)

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )
    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Train at step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        if (step % config.val_interval == 0 or step == start_step) and config.use_val_dataset:
            val_infos = []
            for val_batch in tqdm.tqdm(val_data_loader, dynamic_ncols=True, desc='Validation', leave=False):
                with sharding.set_mesh(mesh):
                    val_outputs = pval_inference(val_rng, train_state, val_batch)
                val_info = compute_mse(state=val_outputs['state'],
                                       actions=val_outputs['actions'],
                                       targets=val_outputs['targets'],
                                       action_mask=val_outputs['action_mask'],
                                       output_transform=output_transform,
                                       target_transform=target_transform)
                if 'text_loss' in val_outputs:
                    val_info['text_loss'] = val_outputs['text_loss']
                val_infos.append(val_info)
            stacked_val_infos = common_utils.stack_forest(val_infos)
            reduced_val_info = jax.device_get(jax.tree.map(jnp.nanmean, stacked_val_infos))
            val_info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_val_info.items())
            pbar.write(f"Validation at step {step}: {val_info_str}")
            log_val_info = {f'val/{k}': v for k, v in reduced_val_info.items()}
            wandb.log(log_val_info, step=step)

        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


def notify_task_completion(name, password, vmids):
    url = "http://k8svmgr-main.devops.svc.cluster.local:8000/api/task_finished"
    data = {
        "name": name,
        "password": password,
        "vmids": vmids
    }
    response = requests.post(url, json=data)
    return response


if __name__ == "__main__":
    main(_config.cli())
    time.sleep(30)
    hostname = socket.gethostname()
    if "zhaojunmin" in hostname:
        name = "zhaojunmin"
        password = "1234qwer"
    else:
        exit()
        
    vmids = [socket.gethostname()]
    response = notify_task_completion(name, password, vmids)
    print("Response from server:", response.text)