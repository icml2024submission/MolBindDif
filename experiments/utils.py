"""Utility functions for experiments."""
import os
import numpy as np
import pandas as pd
import torch
import random
import torch.distributed as dist
from openfold.utils import rigid_utils
from data import utils as du

Rigid = rigid_utils.Rigid

def zero_diagonal_blocks(matrix, block_shapes):
    current_row = 0
    for block_size in block_shapes:
        matrix[..., current_row:current_row + block_size, current_row:current_row + block_size] = 0
        current_row += block_size

def prepare_input(A_file_path, B_file_path, diffuser):
      
    featsA = pd.read_csv(A_file_path)
    featsB = pd.read_csv(B_file_path)
    
    A_res_idx = featsA['id'].values
    B_res_idx = featsB['id'].values
    
    new_res_idx_A = A_res_idx - np.min(A_res_idx)
    new_res_idx_B = B_res_idx - np.min(B_res_idx)
    
    ridx_r = np.hstack((new_res_idx_B, new_res_idx_B))
    ridx_p = new_res_idx_A

    #one_hot = torch.nn.functional.one_hot(target, num_classes=)
    A_res_type = du.encode_res(featsA['residue'].values, ribose=False)
    Bb_res_type = du.encode_res(featsB['residue'].values, ribose=False)
    Br_res_type = du.encode_res(featsB['residue'].values, ribose=True)

    rtype_r = np.concatenate((Bb_res_type,Br_res_type), axis=0)
    rtype_p = A_res_type

    A_res_bind = featsA['binded'].values
    B_res_bind = featsB['binded'].values

    rbind_r = np.hstack((B_res_bind, B_res_bind))
    rbind_p = A_res_bind

    A_linears = featsA.iloc[:,3:].values
    Bb_linears = featsB.iloc[:,3:15].values
    Br_linears = featsB.iloc[:,15:].values

    com = np.sum(A_linears, axis=0)[:3]/len(A_linears)
    A_linears[:,:3] -= com
    Bb_linears[:,:3] -= com
    Br_linears[:,:3] -= com

    A_frames = du.linear_to_4x4(A_linears)                                          
    Bb_frames = du.linear_to_4x4(Bb_linears)
    Br_frames = du.linear_to_4x4(Br_linears)

    
    T_r = torch.tensor(np.concatenate((Bb_frames, Br_frames), axis=0)).float()
    T_p = torch.tensor(A_frames).float()

    T_r = rigid_utils.Rigid.from_tensor_4x4(T_r.unsqueeze(1))[:, 0]
    T_p = rigid_utils.Rigid.from_tensor_4x4(T_p.unsqueeze(1))[:, 0]

    final_feats = {
        'rtype_r': torch.tensor(rtype_r).to(torch.int),
        'rtype_p': torch.tensor(rtype_p).to(torch.int),
        'ridx_r': torch.tensor(ridx_r).to(torch.int16),
        'ridx_p': torch.tensor(ridx_p).to(torch.int16),
        'bind_r': torch.tensor(rbind_r).to(torch.int),
        'bind_p': torch.tensor(rbind_p).to(torch.int),
        'T_r_0': T_r,
        'T_p_0': T_p,
        'com': com
    }
    
    t=torch.tensor(1.0)
    diff_feats_t = diffuser.sample_ref(n_samples=final_feats['T_r_0'].shape[0])
    final_feats['sc_r'] = torch.zeros_like(final_feats['T_r_0'].get_trans())
    final_feats['sc_p'] = torch.clone(final_feats['T_p_0'].get_trans())
    final_feats.update(diff_feats_t)
    final_feats['t'] = t

    final_feats['T_r_0'] = final_feats['T_r_0'].to_tensor_7()
    final_feats['T_r_t'] = final_feats['T_r_t'].to_tensor_7()
    final_feats['T_p_0'] = final_feats['T_p_0'].to_tensor_7()
        
    return final_feats




def get_ddp_info():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    node_id = rank // world_size
    return {"node_id": node_id, "local_rank": local_rank, "rank": rank, "world_size": world_size}


def flatten_dict(raw_dict):
    """Flattens a nested dict."""
    flattened = []
    for k, v in raw_dict.items():
        if isinstance(v, dict):
            flattened.extend([
                (f'{k}:{i}', j) for i, j in flatten_dict(v)
            ])
        else:
            flattened.append((k, v))
    return flattened


def t_stratified_loss(batch_t, batch_loss, num_bins=5, loss_name=None):
    """Stratify loss by binning t."""
    flat_losses = batch_loss.flatten()
    flat_t = batch_t.flatten()
    bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins+1)
    bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
    t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
    t_binned_n = np.bincount(bin_idx)
    stratified_losses = {}
    if loss_name is None:
        loss_name = 'loss'
    for t_bin in np.unique(bin_idx).tolist():
        bin_start = bin_edges[t_bin]
        bin_end = bin_edges[t_bin+1]
        t_range = f'{loss_name} t=[{bin_start:.2f},{bin_end:.2f})'
        range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
        stratified_losses[t_range] = range_loss
    return stratified_losses


def get_sampled_mask(contigs, length, rng=None, num_tries=1000000):
    '''
    Parses contig and length argument to sample scaffolds and motifs.

    Taken from rosettafold codebase.
    '''
    length_compatible=False
    count = 0
    while length_compatible is False:
        inpaint_chains=0
        contig_list = contigs.strip().split()
        sampled_mask = []
        sampled_mask_length = 0
        #allow receptor chain to be last in contig string
        if all([i[0].isalpha() for i in contig_list[-1].split(",")]):
            contig_list[-1] = f'{contig_list[-1]},0'
        for con in contig_list:
            if (all([i[0].isalpha() for i in con.split(",")[:-1]]) and con.split(",")[-1] == '0'):
                #receptor chain
                sampled_mask.append(con)
            else:
                inpaint_chains += 1
                #chain to be inpainted. These are the only chains that count towards the length of the contig
                subcons = con.split(",")
                subcon_out = []
                for subcon in subcons:
                    if subcon[0].isalpha():
                        subcon_out.append(subcon)
                        if '-' in subcon:
                            sampled_mask_length += (int(subcon.split("-")[1])-int(subcon.split("-")[0][1:])+1)
                        else:
                            sampled_mask_length += 1

                    else:
                        if '-' in subcon:
                            if rng is not None:
                                length_inpaint = rng.integers(int(subcon.split("-")[0]),int(subcon.split("-")[1]))
                            else:
                                length_inpaint=random.randint(int(subcon.split("-")[0]),int(subcon.split("-")[1]))
                            subcon_out.append(f'{length_inpaint}-{length_inpaint}')
                            sampled_mask_length += length_inpaint
                        elif subcon == '0':
                            subcon_out.append('0')
                        else:
                            length_inpaint=int(subcon)
                            subcon_out.append(f'{length_inpaint}-{length_inpaint}')
                            sampled_mask_length += int(subcon)
                sampled_mask.append(','.join(subcon_out))
        #check length is compatible 
        if length is not None:
            if sampled_mask_length >= length[0] and sampled_mask_length < length[1]:
                length_compatible = True
        else:
            length_compatible = True
        count+=1
        if count == num_tries: #contig string incompatible with this length
            raise ValueError("Contig string incompatible with --length range")
    return sampled_mask, sampled_mask_length, inpaint_chains
