"""Utilities for calculating all atom representations."""
import torch
from data import residue_constants
from openfold.utils import rigid_utils as ru

### THIS WHOLE THING SHOULD BE REWRITTEN AND RESTRUCTURED

Rigid = ru.Rigid
Rotation = ru.Rotation


IDEALIZED_POS_R = torch.tensor(residue_constants.restype_atom_rigid_positions_r)
ATOM_MASK_R = torch.tensor(residue_constants.restype_atom_mask_r)
N_ATOMS_R = residue_constants.atom_types_num_r


IDEALIZED_POS_P = torch.tensor(residue_constants.restype_atom_rigid_positions_p)
ATOM_MASK_P = torch.tensor(residue_constants.restype_atom_mask_p)
N_ATOMS_P = residue_constants.atom_types_num_p



def compute_backbone_r(
        r: torch.Tensor,
        restype: torch.Tensor,
    ):
   

    t_atoms_to_global = r.unsqueeze(2).repeat(1, 1, N_ATOMS_R, 1)
    t_atoms_to_global = Rigid.from_tensor_7(t_atoms_to_global.type(torch.float32))
    
    res_num = restype.cpu().long()
    
    frame_atom_mask = ATOM_MASK_R[res_num, ...].to(r.device)

    
    frame_null_pos = IDEALIZED_POS_R[res_num, ...].to(r.device)
    pred_positions = t_atoms_to_global.apply(frame_null_pos)

    return pred_positions[frame_atom_mask.bool(),:], frame_atom_mask.sum(-1)[0]



def compute_backbone_p(
        r: torch.Tensor,
        restype: torch.Tensor,
    ):


    t_atoms_to_global = r.unsqueeze(2).repeat(1, 1, N_ATOMS_P, 1)
    t_atoms_to_global = Rigid.from_tensor_7(t_atoms_to_global.type(torch.float32))
    
    res_num = restype.cpu().long()-8
    
    frame_atom_mask = ATOM_MASK_P[res_num, ...].to(r.device)

    
    frame_null_pos = IDEALIZED_POS_P[res_num, ...].to(r.device)
    pred_positions = t_atoms_to_global.apply(frame_null_pos)

    return pred_positions[frame_atom_mask.bool(),:]



def compute_sparse_backbone_r(
        r: torch.Tensor,
        restype: torch.Tensor,
    ):


    t_atoms_to_global = r.unsqueeze(2).repeat(1, 1, N_ATOMS_R, 1)
    t_atoms_to_global = Rigid.from_tensor_7(t_atoms_to_global.type(torch.float32))
    
    res_num = restype.cpu().long()
    
    frame_atom_mask = ATOM_MASK_R[res_num, ...].to(r.device).unsqueeze(-1)

    
    frame_null_pos = IDEALIZED_POS_R[res_num, ...].to(r.device)
    pred_positions = t_atoms_to_global.apply(frame_null_pos)

    return pred_positions*frame_atom_mask, frame_atom_mask
