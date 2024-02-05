import pickle
import os
import copy
from typing import Any
import numpy as np
from typing import List, Dict, Any
import collections
from omegaconf import OmegaConf
import pandas as pd



from data import residue_constants
from data import upload_data

import io
from torch.utils import data
import torch




move_to_np = lambda x: x.cpu().detach().numpy()


class CPU_Unpickler(pickle.Unpickler):
    """Pytorch pickle loading workaround.

    https://github.com/pytorch/pytorch/issues/16797
    """
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def write_pkl(
        save_path: str, pkl_data: Any, create_dir: bool = False, use_torch=False):
    """Serialize data into a pickle file."""
    if create_dir:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if use_torch:
        torch.save(pkl_data, save_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(save_path, 'wb') as handle:
            pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pkl(read_path: str, verbose=True, use_torch=False, map_location=None):
    """Read data from a pickle file."""
    try:
        if use_torch:
            return torch.load(read_path, map_location=map_location)
        else:
            with open(read_path, 'rb') as handle:
                return pickle.load(handle)
    except Exception as e:
        try:
            with open(read_path, 'rb') as handle:
                return CPU_Unpickler(handle).load()
        except Exception as e2:
            if verbose:
                print(f'Failed to read {read_path}. First error: {e}\n Second error: {e2}')
            raise(e)

def compare_conf(conf1, conf2):
    return OmegaConf.to_yaml(conf1) == OmegaConf.to_yaml(conf2)




def encode_res(res_list, ribose=False):
    encodes = []
    for res in res_list:
        if ribose:
            restype_idx = residue_constants.restypes_order.get(res+'R', residue_constants.restypes_num)
        else:
            restype_idx = residue_constants.restypes_order.get(res, residue_constants.restypes_num)

        encodes.append(restype_idx)

    encodes = np.array(encodes)
    return encodes

def linear_to_4x4(lin_arr):
    arr_4x4 = []
    for vec in lin_arr:
        t = vec[:3]
        e1 = vec[3:6]
        e2 = vec[6:9]
        e3 = vec[9:12]
        mat = np.hstack((e1.reshape(-1,1),e2.reshape(-1,1),e3.reshape(-1,1),t.reshape(-1,1)))
        mat = np.vstack((mat, np.array([0,0,0,1])))
        arr_4x4.append(mat)
    return np.array(arr_4x4)





def write_checkpoint(
        ckpt_path: str,
        model,
        conf,
        optimizer,
        epoch,
        step,
        logger=None,
        use_torch=True,
    ):
    """Serialize experiment state and stats to a pickle file.

    Args:
        ckpt_path: Path to save checkpoint.
        conf: Experiment configuration.
        optimizer: Optimizer state dict.
        epoch: Training epoch at time of checkpoint.
        step: Training steps at time of checkpoint.
        exp_state: Experiment state to be written to pickle.
        preds: Model predictions to be written as part of checkpoint.
    """
    ckpt_dir = os.path.dirname(ckpt_path)
    '''for fname in os.listdir(ckpt_dir):
        if '.pkl' in fname or '.pth' in fname:
            os.remove(os.path.join(ckpt_dir, fname))'''
    if logger is not None:
        logger.info(f'Serializing experiment state to {ckpt_path}')
    else:
        print(f'Serializing experiment state to {ckpt_path}')
    write_pkl(
        ckpt_path,
        {
            'model': model,
            'conf': conf,
            'optimizer': optimizer,
            'epoch': epoch,
            'step': step
        },
        use_torch=use_torch)

def concat_np_features(
        np_dicts: List[Dict[str, np.ndarray]], add_batch_dim: bool):
    """Performs a nested concatenation of feature dicts.

    Args:
        np_dicts: list of dicts with the same structure.
            Each dict must have the same keys and numpy arrays as the values.
        add_batch_dim: whether to add a batch dimension to each feature.

    Returns:
        A single dict with all the features concatenated.
    """
    combined_dict = collections.defaultdict(list)
    for chain_dict in np_dicts:
        for feat_name, feat_val in chain_dict.items():
            if add_batch_dim:
                feat_val = feat_val[None]
            combined_dict[feat_name].append(feat_val)
    # Concatenate each feature
    for feat_name, feat_vals in combined_dict.items():
        combined_dict[feat_name] = np.concatenate(feat_vals, axis=0)
    return combined_dict



def create_data_loader(
        torch_dataset: data.Dataset,
        batch_size,
        shuffle,
        sampler=None,
        num_workers=0,
        np_collate=False,
        max_squared_res=1e6,
        length_batch=False,
        drop_last=False,
        prefetch_factor=2):
    """Creates a data loader with jax compatible data structures."""
    if np_collate:
        collate_fn = lambda x: concat_np_features(x, add_batch_dim=True)

    else:
        collate_fn = None
    persistent_workers = True if num_workers > 0 else False
    prefetch_factor = 2 if num_workers == 0 else prefetch_factor
    return data.DataLoader(
        torch_dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
        # Need fork https://github.com/facebookresearch/hydra/issues/964
        multiprocessing_context='fork' if num_workers != 0 else None,
        )









def calc_distogram(pos1, pos2, min_bin, max_bin, num_bins):
    dists_2d = torch.linalg.norm(
        pos1[:, :, None, :] - pos2[:, None, :, :], axis=-1)[..., None]
    lower = torch.linspace(
        min_bin,
        max_bin,
        num_bins,
        device=pos1.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d >= lower) * (dists_2d < upper)).type(pos1.dtype)
    return dgram

def quat_to_rotvec(quat, eps=1e-6):
    # w > 0 to ensure 0 <= angle <= pi
    flip = (quat[..., :1] < 0).float()
    quat = (-1 * quat) * flip + (1 - flip) * quat

    angle = 2 * torch.atan2(
        torch.linalg.norm(quat[..., 1:], dim=-1),
        quat[..., 0]
    )

    angle2 = angle * angle
    small_angle_scales = 2 + angle2 / 12 + 7 * angle2 * angle2 / 2880
    large_angle_scales = angle / torch.sin(angle / 2 + eps)

    small_angles = (angle <= 1e-3).float()
    rot_vec_scale = small_angle_scales * small_angles + (1 - small_angles) * large_angle_scales
    rot_vec = rot_vec_scale[..., None] * quat[..., 1:]
    return rot_vec






def process_data(out_dir, names):
    valid_set = pd.DataFrame(columns=['pdb_name', 'A_path', 'B_path'])
    for name in names:
        A_path, B_path = upload_data.create_data(out_dir, name)
        valid_set.loc[len(valid_set)] = {'pdb_name': name, 'A_path':A_path, 'B_path':B_path}

    return valid_set


def extract_true_positions(T_r_traj, com):
    com = copy.deepcopy(com)
    out_of_com = lambda x: x + com
    all_rigids = [move_to_np(copy.deepcopy(T_r_t).apply_trans_fn(out_of_com).to_tensor_7().squeeze(0)) for T_r_t in T_r_traj]
    return np.array(all_rigids)

