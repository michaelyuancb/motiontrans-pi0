import random
import os
from typing import Protocol, SupportsIndex, TypeVar, List, Dict

import jax.numpy as jnp
import numpy as np
import torch
import torchvision
import json
import copy
from tqdm import tqdm
import shutil 
import pathlib
from datetime import datetime
import scipy.interpolate as si
import scipy.spatial.transform as st

from openpi.policies.pose_util import pose_to_mat, mat_to_pose10d, mat_to_pose
from openpi.policies.pose_repr_util import convert_pose_mat_rep
import zarr
from filelock import FileLock
from openpi.replay_buffer import ReplayBuffer
from openpi.imagecodecs_numcodecs import register_codecs


register_codecs()

def get_replay_buffer(dataset_path, cache_dir):
    if dataset_path is None:
        return None
    if cache_dir is None:
        replay_buffer = ReplayBuffer.create_from_path(zarr_path=dataset_path, mode='r')      
    else:
        # TODO: refactor into a stand alone function?
        # determine path name
        mod_time = os.path.getmtime(dataset_path)
        stamp = datetime.fromtimestamp(mod_time).isoformat()
        stem_name = os.path.basename(dataset_path).split('.')[0]
        cache_name = '_'.join([stem_name, stamp])
        cache_dir = pathlib.Path(os.path.expanduser(cache_dir))
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir.joinpath(cache_name + '.zarr.mdb')
        lock_path = cache_dir.joinpath(cache_name + '.lock')
        
        # load cached file
        print('Acquiring lock on cache.')
        with FileLock(lock_path):
            # cache does not exist
            if not cache_path.exists():
                try:
                    with zarr.LMDBStore(str(cache_path),     
                        writemap=True, metasync=False, sync=False, map_async=True, lock=False
                    ) as lmdb_store:
                        print(f"Copying data to {str(cache_path)}")
                        ReplayBuffer.copy_from_path(zarr_path=dataset_path, store=lmdb_store, compressors='disk')
                    print("Cache written to disk!")
                except Exception as e:
                    shutil.rmtree(cache_path)
                    raise e
            
        # open read-only lmdb store
        store = zarr.LMDBStore(str(cache_path), readonly=True, lock=False)
        replay_buffer = ReplayBuffer.create_from_group(
            group=zarr.group(store)
        )
    return replay_buffer



def get_replay_buffer_list(dataset_path, cache_dir):
    dataset_path_list_tmp = dataset_path.split("|")
    dataset_path_list_tmp = [data_p for data_p in dataset_path_list_tmp if data_p is not None and len(data_p) > 0]
    dataset_path_list = []
    for data_p in dataset_path_list_tmp:
        if data_p.endswith(".json"):
            continue  # skip json files
        if data_p.endswith('.zarr'):
            dataset_path_list.append(data_p)
        else:
            data_p_list = os.listdir(data_p)
            for data_fp in data_p_list:
                if data_fp.endswith('.zarr'):
                    dataset_path_list.append(os.path.join(data_p, data_fp))
                elif data_fp.endswith(".json"):
                    continue  # skip json files
                else:
                    raise ValueError(f'Unsupported dataset path {data_fp} from auto-folder file finding, only support .zarr files, please check the dataset path.')
    replay_buffer_list = []
    for data_p in dataset_path_list:
        # print(data_p)
        replay_buffer_list.append(get_replay_buffer(data_p, cache_dir))
    return replay_buffer_list, dataset_path_list


def get_instruction_from_filename_list(filename_list):
    instruction_list = []
    for filename in filename_list:
        # instruction位于两个+中间，例如XXXX+instruction+XXXX
        if '+' in filename and filename.find('+') > 0:
            instruction = filename[filename.find('+') + 1:filename.rfind('+')]
            instruction = instruction.strip()
            instruction = instruction.replace('_', ' ')
            if not instruction.endswith('.'):
                instruction += '.'
            if '+' in instruction:
                raise ValueError(f'Filename {filename} contains multiple instructions between + signs.')
            if len(instruction) > 0:
                instruction_list.append(instruction)
            else:
                raise ValueError(f'Filename {filename} contains empty instruction between + signs.')
        else:
            raise ValueError(f'Filename {filename} does not contain instruction in [ ] format.')
    return instruction_list


T_co = TypeVar("T_co", covariant=True)

class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")
    

class ZarrDataset(Dataset):
    def __init__(self, data_config, action_horizon: int, split='train'):

        replay_buffer_list, dataset_path_list = get_replay_buffer_list(dataset_path=data_config.dataset_path, cache_dir=None)
        self.dataset_path_list = dataset_path_list
        self.dataset_name_instructions_list = get_instruction_from_filename_list(dataset_path_list)
        self.single_arm = data_config.single_arm

        self.data_config = data_config
        self.alpha = data_config.alpha
        self.image_hisory_length = len(data_config.image_down_sample_steps) + 1
        self.image_down_sample_steps = data_config.image_down_sample_steps
        self.state_hisory_length = len(data_config.state_down_sample_steps) + 1
        self.state_down_sample_steps = data_config.state_down_sample_steps
        self.action_horizon = action_horizon
        self.action_down_sample_steps = data_config.action_down_sample_steps
        self.compute_norm_stats = data_config.compute_norm_stats
        self.proprioception_rep = data_config.proprioception_rep
        self.action_rep = data_config.action_rep
        self.proprioception_droprate = data_config.proprioception_droprate
        
        self._get_all_episodes(replay_buffer_list)

        n_robot = sum(self.embodiments)
        n_human = len(self.embodiments) - n_robot
        if n_human == 0:
            self.alpha = 1.0
        elif n_robot == 0:
            self.alpha = 0.0
        else:
            alpha_robot = self.alpha / n_robot
            alpha_human = (1 - self.alpha) / n_human
            adjust_alpha = alpha_robot / (alpha_robot + alpha_human)
            self.alpha = adjust_alpha

        self.datas = [
            replay_buffer.data for replay_buffer in replay_buffer_list
        ]
        self.action_horizon = action_horizon

        if data_config.create_train_val_split:
            assert data_config.use_val_dataset
        if data_config.create_train_val_split and split == 'train':
            self.create_train_val_split()
        if data_config.use_val_dataset:
            self.indices = self.get_indices(split)
        else:
            self.indices = list(range(self.episode_idx_range[0], self.episode_idx_range[-1]))

        del replay_buffer_list
        

    def _get_all_episodes(self, replay_buffer_list) -> List:
        """获取所有episode的数据"""
        episodes = []
        data_idxs = []
        ep_to_zarr = []
        episode_idx_range = [0]
        starts = []
        ends = []
        embodiments = [] 
        n_episodes = 0
        for idx_ep, replay_buffer in enumerate(tqdm(replay_buffer_list)):
            num_episodes = len(replay_buffer.episode_ends)
            dataset_name = self.dataset_path_list[idx_ep].split('/')[-1]
            embodiment = int('robot' in dataset_name)         # 1: robot, 0: human
            for idx in range(num_episodes):

                eps_start = 0 if idx == 0 else replay_buffer.episode_ends[idx - 1]
                eps_end = replay_buffer.episode_ends[idx]

                data_idxs.extend(list(range(eps_start, eps_end)))
                episodes.extend([n_episodes] * (eps_end - eps_start))
                n_episodes += 1
                starts.extend([eps_start] * (eps_end - eps_start))
                ends.extend([eps_end] * (eps_end - eps_start))
                ep_to_zarr.extend([idx_ep] * (eps_end - eps_start))
                embodiments.extend([embodiment] * (eps_end - eps_start))
                episode_idx_range.append(episode_idx_range[-1] + (eps_end - eps_start))

        self.start_frames = starts         # the detailed start idx inside zarr for data[i]
        self.end_frames = ends             # the detailed end idx inside zarr for data[i]
        self.zarr_idxs = ep_to_zarr        # the idxs of zarr for data[i]
        self.episodes_idxs = episodes      # the (global) episode idx for data[i]
        self.data_idxs = data_idxs         # the detailed idx inside zarr for data[i]
        self.embodiments = embodiments     # 1: robot, 0: human
        self.n_episodes = len(set(episodes))
        self.episode_idx_range = episode_idx_range  # cumulative sum of episode lengths, used for indexing
        

    def create_train_val_split(self):
        episode_num = len(set(self.episodes_idxs))
        val_num = int(episode_num * self.data_config.val_ratio)
        np.random.seed(self.data_config.seed)
        train_episode_idx = np.random.choice(episode_num, episode_num - val_num, replace=False)
        train_episode_idx = np.sort(train_episode_idx)
        val_episode_idx = np.setdiff1d(np.arange(episode_num), train_episode_idx)
        os.makedirs(self.data_config.norm_stats_dir, exist_ok=True)
        with open(os.path.join(self.data_config.norm_stats_dir, 'train_val_split.json'), 'w') as f:
            json.dump({'train_episode_idx': train_episode_idx.tolist(), 'val_episode_idx': val_episode_idx.tolist()}, f)


    def get_indices(self, split):
        with open(os.path.join(self.data_config.norm_stats_dir, 'train_val_split.json'), 'r') as f:
            split_idx = json.load(f)[f'{split}_episode_idx']
        indices = []
        for episode_idx in split_idx:
            start_idx = self.episode_idx_range[episode_idx]
            end_idx = self.episode_idx_range[episode_idx + 1]
            indices.extend(list(range(start_idx, end_idx)))
        return indices


    def get_val_dataset(self):
        val_set = copy.copy(self)
        val_set.indices = self.get_indices('val')
        return val_set


    def set_sample_ratio(self, sample_ratio):
        interval_size = int(1.0 / sample_ratio)
        self.indices = self.indices[::interval_size]


    def __len__(self):
        return len(self.indices)
    
    
    def get_prob(self, start_step, end_step, now_step, start_prob=0.8, end_prob=0.4):
        # from start_prob -> end_prob linearly
        assert start_step <= now_step < end_step
        return start_prob - (start_prob - end_prob) * (now_step - start_step) / (end_step - start_step)


    def _get_single_arm_data(self, data, robot_idx, state_target_idx, action_idx_slice, cam_proj, interpolation_start, interpolation_end):

        if f'robot{robot_idx}_eef_pos' not in data.keys():
            return None, None, None, None

        rot_preprocess = st.Rotation.from_rotvec
        rot_postprocess = st.Rotation.as_rotvec
        slerp = st.Slerp(
            times=np.arange(interpolation_start, interpolation_end),
            rotations=rot_preprocess(data[f'robot{robot_idx}_eef_rot_axis_angle'][interpolation_start: interpolation_end])
        )
        output_rot = rot_postprocess(slerp(state_target_idx))
        interp = si.interp1d(
            x=np.arange(interpolation_start, interpolation_end),
            y=data[f'robot{robot_idx}_eef_pos'][interpolation_start: interpolation_end],
            axis=0, assume_sorted=True)
        output_pos = interp(state_target_idx)
        obs_pose = np.concatenate([output_pos, output_rot], axis=-1)

        cam_obs_pose_mat = pose_to_mat(obs_pose)
        cam_obs_pose_mat = cam_proj @ cam_obs_pose_mat 
        relative_pose_base = cam_obs_pose_mat[-1].copy()

        cam_obs_pose_mat = convert_pose_mat_rep(
            cam_obs_pose_mat, 
            base_pose_mat=relative_pose_base,
            pose_rep=self.proprioception_rep,
            backward=False)
        
        # for robot eef proprioception, we ignore identity relative action for eef-pose.
        cam_obs_pose = mat_to_pose10d(cam_obs_pose_mat[:-1])
        # for hand / gripper proprioception, we ignore the earliest propriception to make the number of timestamps for eef & gripper the same.
        interp = si.interp1d(
            x=np.arange(interpolation_start, interpolation_end),
            y=data[f'gripper{robot_idx}_gripper_pose'][interpolation_start: interpolation_end],
            axis=0, assume_sorted=True)
        gripper_obs_pose = interp(state_target_idx)[1:]

        action_pose = data['action'][action_idx_slice, 12*robot_idx:12*robot_idx+6]
        cam_action_pose_mat = pose_to_mat(action_pose)
        cam_action_pose_mat = cam_proj @ cam_action_pose_mat 

        cam_action_pose_mat = convert_pose_mat_rep(
            cam_action_pose_mat, 
            base_pose_mat=relative_pose_base,
            pose_rep=self.action_rep,
            backward=False)
        
        cam_action_pose = mat_to_pose10d(cam_action_pose_mat)
        gripper_action_pose = data['action'][action_idx_slice, 12*robot_idx+6:12*robot_idx+12]

        return cam_obs_pose, gripper_obs_pose, cam_action_pose, gripper_action_pose


    def __getitem__(self, idx: SupportsIndex) -> Dict:
        idx = self.indices[idx]
        start_idx = self.start_frames[idx]     # inside_buffer start_idx
        end_idx = self.end_frames[idx]         # inside_buffer end_idx
        zarr_idx = self.zarr_idxs[idx]      
        data = self.datas[zarr_idx]
        episode_idx = self.episodes_idxs[idx]  # inside_buffer episode_idx
        embodiment = self.embodiments[idx]
        idx = self.data_idxs[idx]              # inside_buffer data_idx
        if 'camera0_pose' in data.keys():
            a = np.array(pose_to_mat(data['camera0_pose'][idx]))
            b = np.array(pose_to_mat(data['camera0_pose'][start_idx]))
            cam_proj = np.linalg.inv(a) @ b
        else:
            cam_proj = np.eye(4)

        # ============================== add image data ==============================
        rgbs = {}
        image_target_idx = np.array([idx] + [idx - self.image_down_sample_steps[history_idx] for history_idx in range(self.image_hisory_length - 1)])
        image_target_idx = np.clip(image_target_idx[::-1], start_idx, end_idx - 1)
        for i in range(self.image_hisory_length):

            img = torch.from_numpy(data['camera0_rgb'][int(image_target_idx[i])].astype(np.float32)) / 255.0 * 2.0 - 1.0
            rgbs['image_{}'.format(i + 1)] = img

        # print(type(rgbs['image_{}'.format(i + 1)]))
        # print(rgbs['image_{}'.format(i + 1)].dtype)
        # print(rgbs['image_{}'.format(i + 1)].max())

        # ============================== add proprioception ==============================
        state_target_idx = np.array([idx] + [idx - self.state_down_sample_steps[history_idx] for history_idx in range(self.state_hisory_length - 1)])
        state_target_idx = np.clip(state_target_idx[::-1], start_idx, end_idx - 1)  #  history->now
        interpolation_start = max(int(state_target_idx[0]) - 5, start_idx)
        interpolation_end = min(int(state_target_idx[-1]) + 2 + 5, end_idx)

        slice_end = min(end_idx, idx + (self.action_horizon - 1) * self.action_down_sample_steps + 1)
        action_idx_slice = slice(idx, slice_end, self.action_down_sample_steps)          #(idx: slice_end: self.action_down_sample_steps)

        obs_pose_right, gripper_pose_right, action_pose_right, action_gripper_right = self._get_single_arm_data(
            data, 0, state_target_idx, action_idx_slice, cam_proj, interpolation_start, interpolation_end)

        if self.single_arm is False:
            obs_pose_left, gripper_pose_left, action_pose_left, action_gripper_left = self._get_single_arm_data(
                data, 1, state_target_idx, action_idx_slice, cam_proj, interpolation_start, interpolation_end)

            if obs_pose_right is None and obs_pose_left is None:
                raise ValueError(f"Missing data for episode {episode_idx} at index {idx}. Please check the dataset.")
            if obs_pose_right is None:
                obs_pose_right = np.zeros_like(obs_pose_left)
                gripper_pose_right = np.zeros_like(gripper_pose_left)
                action_pose_right = np.zeros_like(action_pose_left)
                action_gripper_right = np.zeros_like(action_gripper_left)
                action_real_horizon = action_pose_left.shape[0]
            elif obs_pose_left is None:
                obs_pose_left = np.zeros_like(obs_pose_right)
                gripper_pose_left = np.zeros_like(gripper_pose_right)
                action_pose_left = np.zeros_like(action_pose_right)
                action_gripper_left = np.zeros_like(action_gripper_right)
                action_real_horizon = action_pose_right.shape[0]
            else:
                action_real_horizon = min(action_pose_left.shape[0], action_pose_right.shape[0])
            
            actions = np.concatenate([
                action_pose_right[:action_real_horizon], action_gripper_right[:action_real_horizon], 
                action_pose_left[:action_real_horizon], action_gripper_left[:action_real_horizon],
            ], axis=-1)
        else:
            action_real_horizon = action_pose_right.shape[0]
            actions = np.concatenate([
                action_pose_right[:action_real_horizon], action_gripper_right[:action_real_horizon], 
            ], axis=-1)

        padding = np.repeat(actions[-1:], self.action_horizon - action_real_horizon, axis=0)
        actions = np.concatenate([actions, padding], axis=0)
        action_is_pad = torch.tensor([False] * actions.shape[0] + [True] * (self.action_horizon - actions.shape[0]))
        
        if self.single_arm is False:
            states = np.concatenate([
                obs_pose_right.flatten(), gripper_pose_right.flatten(), obs_pose_left.flatten(), gripper_pose_left.flatten(),
            ], axis=-1)
        else:
            states = np.concatenate([
                obs_pose_right.flatten(), gripper_pose_right.flatten()
            ], axis=-1)

        if random.random() <= self.proprioception_droprate:
            states = np.zeros_like(states,dtype=states.dtype)
        
        if 'instruction' in data.keys():
            prompt = str(data['instruction'][idx])
        else:
            prompt = self.dataset_name_instructions_list[zarr_idx]
        prompt = prompt.strip()
        if not prompt.endswith('.'):
            prompt += '.'

        if embodiment == 0:
            alpha = 1 - self.alpha
        else:
            alpha = self.alpha

        # print(f"states.shape: {states.shape}")
        # print(f"actions.shape: {actions.shape}")
        # print(f"prompt: {prompt}")
        # print(f"alpha: {alpha}")

        sample = {
            **rgbs,
            "state": torch.from_numpy(states.astype(np.float32)),
            'actions': torch.from_numpy(actions.astype(np.float32)),
            'action_is_pad': action_is_pad,
            'prompt': prompt,
            'alpha': alpha,
        }
        
        return sample