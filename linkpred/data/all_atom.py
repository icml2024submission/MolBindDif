"""Utilities for calculating all atom representations."""
import torch
import torch.nn as nn
from linkpred.data import residue_constants
from openfold.utils import rigid_utils as ru

Rigid = ru.Rigid
Rotation = ru.Rotation

FRAMES = torch.tensor(residue_constants.restype_rigid_group_default_frame)
GROUP_IDX = torch.tensor(residue_constants.restype_atom_to_rigid_group)
IDEALIZED_POS = torch.tensor(residue_constants.restype_atom_rigid_group_positions)
ATOM_MASK = torch.tensor(residue_constants.restype_atom_mask)
N_ATOMS = residue_constants.atom_types_num




def torsion_angles_to_frames(
    r: Rigid,
    alpha: torch.Tensor,
    aatype: torch.Tensor
):
    r = Rigid.from_tensor_7(r)

    res_num = aatype.cpu().long()
    default_4x4 = FRAMES[res_num, ...].to(r.device)

    default_r = r.from_tensor_4x4(default_4x4)

    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 0] = 1

    alpha = torch.cat([bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2)
    
    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape).to(r.device)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 0]
    all_rots[..., 1, 2] = -alpha[..., 1]
    all_rots[..., 2, 1] = alpha[..., 1]
    all_rots[..., 2, 2] = alpha[..., 0]

    all_rots = Rigid(Rotation(rot_mats=all_rots), None)

    all_frames = default_r.compose(all_rots)

    nu2_frame_to_bb = all_frames[..., 1]
    nu3_frame_to_bb = all_frames[..., 2]
    alpha_frame_to_frame = all_frames[..., 3]
    beta_frame_to_frame = all_frames[..., 4]
    gamma_frame_to_frame = all_frames[..., 5]
    delta_frame_to_bb = all_frames[..., 6]

    gamma_frame_to_bb = delta_frame_to_bb.compose(gamma_frame_to_frame)
    beta_frame_to_bb = gamma_frame_to_bb.compose(beta_frame_to_frame)
    alpha_frame_to_bb = beta_frame_to_bb.compose(alpha_frame_to_frame)

    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., 0].unsqueeze(-1),
            delta_frame_to_bb.unsqueeze(-1),
            gamma_frame_to_bb.unsqueeze(-1),
            beta_frame_to_bb.unsqueeze(-1),
            alpha_frame_to_bb.unsqueeze(-1),
            nu2_frame_to_bb.unsqueeze(-1),
            nu3_frame_to_bb.unsqueeze(-1)
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)

    return all_frames_to_global


def frames_and_literature_positions_to_atom_pos(
    r: Rigid,
    aatype: torch.Tensor,
    default_frames=FRAMES,
    group_idx=GROUP_IDX,
    lit_positions=IDEALIZED_POS
):
    res_num = aatype.cpu().long()
    group_mask = group_idx[res_num, ...].to(r.device)

    group_mask = nn.functional.one_hot(
        group_mask,
        num_classes=default_frames.shape[-3],
    )

    t_atoms_to_global = r[..., None, :] * group_mask

    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(lambda x: torch.sum(x, dim=-1))

    lit_positions = lit_positions[res_num, ...].to(r.device)
    pred_positions = t_atoms_to_global.apply(lit_positions)

    return pred_positions
