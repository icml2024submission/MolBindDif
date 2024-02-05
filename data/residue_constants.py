from openfold.utils import rigid_utils

import numpy as np
import torch 

#_chem_comp_atom.pdbx_model_Cartn_x_ideal 
atom_positions_r = {
    'U': [
        ['N1', (0.028, 0.464, 2.451)],
        ['C2', (-0.690, -0.671, 2.486)],
        ['O2', (-0.587, -1.474, 1.580)],
        ['N3', (-1.515, -0.936, 3.517)],
        ['C4', (-1.641, -0.055, 4.530)],
        ['O4', (-2.391, -0.292, 5.460)],
        ['C5', (-0.894, 1.146, 4.502)],
        ['C6', (-0.070, 1.384, 3.459)]
    ],
    'G': [
        ['N9', (-0.297, 0.162, -1.534)],
        ['C8', (-1.440, 0.880, -1.334)],
        ['N7', (-2.066, 1.037, -2.464)],
        ['C5', (-1.364, 0.431, -3.453)],
        ['C6', (-1.556, 0.279, -4.846)],
        ['O6', (-2.534, 0.755, -5.397)],
        ['N1', (-0.626, -0.401, -5.551)],
        ['C2', (0.459, -0.934, -4.923)],
        ['N2', (1.384, -1.626, -5.664)],
        ['N3', (0.649, -0.800, -3.630)],
        ['C4', (-0.226, -0.134, -2.868)]
    ],
    'A': [
        ['N9', (0.158, 0.029, 1.803)],
        ['C8', (1.265, 0.813, 1.672)],
        ['N7', (1.843, 0.963, 2.828)],
        ['C5', (1.143, 0.292, 3.773)],
        ['C6', (1.290, 0.091, 5.156)],
        ['N6', (2.344, 0.664, 5.846)],
        ['N1', (0.391, -0.656, 5.787)],
        ['C2', (-0.617, -1.206, 5.136)],
        ['N3', (-0.792, -1.051, 3.841)],
        ['C4', (0.056, -0.320, 3.126)]
    ],
    'C': [
        ['N1', (-0.036, -0.470, 2.453)],
        ['C2', (0.652, 0.683, 2.514)],
        ['O2', (0.529, 1.504, 1.620)],
        ['N3', (1.467, 0.945, 3.535)],
        ['C4', (1.620, 0.070, 4.520)],
        ['N4', (2.464, 0.350, 5.569)],
        ['C5', (0.916, -1.151, 4.483)],
        ['C6', (0.087, -1.399, 3.442)]
    ],
    'UR': [
        ["C4'", (-0.309, 1.476, -0.0)],
        ["C3'", (0.0, 0.0, -0.0)],
        ["O3'", (-0.567, -0.684, -1.096)],
        ["C2'", (1.519, 0.0, -0.0)],
    ],
    'GR': [
        ["C4'", (-0.33, 1.484, 0.0)],
        ["C3'", (0.0, -0.0, 0.0)],
        ["O3'", (-0.578, -0.746, -1.06)],
        ["C2'", (1.522, -0.0, -0.0)],
    ],
    'AR': [
        ["C4'", (-0.35, 1.466, -0.0)],
        ["C3'", (0.0, 0.0, 0.0)],
        ["O3'", (-0.544, -0.649, -1.112)],
        ["C2'", (1.507, 0.0, -0.0)],
    ],
    'CR': [
        ["C4'", (-0.313, 1.485, -0.0)],
        ["C3'", (0.0, 0.0, 0.0)],
        ["O3'", (-0.515, -0.654, -1.141)],
        ["C2'", (1.518, 0.0, 0.0)],
    ]
}

atom_positions_p = {
    'ALA': [
        ['N', (-0.525, 1.363, 0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.526, -0.000, -0.000)],
        ['CB', (-0.529, -0.774, -1.205)]
    ],
    'ARG': [
        ['N', (-0.524, 1.362, -0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.525, -0.000, -0.000)],
        ['CB', (-0.524, -0.778, -1.209)]
    ],
    'ASN': [
        ['N', (-0.536, 1.357, 0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.526, -0.000, -0.000)],
        ['CB', (-0.531, -0.787, -1.200)]
    ],
    'ASP': [
        ['N', (-0.525, 1.362, -0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.527, 0.000, -0.000)],
        ['CB', (-0.526, -0.778, -1.208)]
    ],
    'CYS': [
        ['N', (-0.522, 1.362, -0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.524, 0.000, 0.000)],
        ['CB', (-0.519, -0.773, -1.212)]
    ],
    'GLN': [
        ['N', (-0.526, 1.361, -0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.526, 0.000, 0.000)],
        ['CB', (-0.525, -0.779, -1.207)]
    ],
    'GLU': [
        ['N', (-0.528, 1.361, 0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.526, -0.000, -0.000)],
        ['CB', (-0.526, -0.781, -1.207)]
    ],
    'GLY': [
        ['N', (-0.572, 1.337, 0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.517, -0.000, -0.000)]
    ],
    'HIS': [
        ['N', (-0.527, 1.360, 0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.525, 0.000, 0.000)],
        ['CB', (-0.525, -0.778, -1.208)]
    ],
    'ILE': [
        ['N', (-0.493, 1.373, -0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.527, -0.000, -0.000)],
        ['CB', (-0.536, -0.793, -1.213)]
    ],
    'LEU': [
        ['N', (-0.520, 1.363, 0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.525, -0.000, -0.000)],
        ['CB', (-0.522, -0.773, -1.214)]
    ],
    'LYS': [
        ['N', (-0.526, 1.362, -0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.526, 0.000, 0.000)],
        ['CB', (-0.524, -0.778, -1.208)]
    ],
    'MET': [
        ['N', (-0.521, 1.364, -0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.525, 0.000, 0.000)],
        ['CB', (-0.523, -0.776, -1.210)]
    ],
    'PHE': [
        ['N', (-0.518, 1.363, 0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.524, 0.000, -0.000)],
        ['CB', (-0.525, -0.776, -1.212)]
    ],
    'PRO': [
        ['N', (-0.566, 1.351, -0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.527, -0.000, 0.000)],
        ['CB', (-0.546, -0.611, -1.293)]
    ],
    'SER': [
        ['N', (-0.529, 1.360, -0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.525, -0.000, -0.000)],
        ['CB', (-0.518, -0.777, -1.211)]
    ],
    'THR': [
        ['N', (-0.517, 1.364, 0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.526, 0.000, -0.000)],
        ['CB', (-0.516, -0.793, -1.215)]
    ],
    'TRP': [
        ['N', (-0.521, 1.363, 0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.525, -0.000, 0.000)],
        ['CB', (-0.523, -0.776, -1.212)]
    ],
    'TYR': [
        ['N', (-0.522, 1.362, 0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.524, -0.000, -0.000)],
        ['CB', (-0.522, -0.776, -1.213)]
    ],
    'VAL': [
        ['N', (-0.494, 1.373, -0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.527, -0.000, -0.000)],
        ['CB', (-0.533, -0.795, -1.213)]
    ],
}


restypes_r = [
    'U', 'G', 'A', 'C', 'UR', 'GR', 'AR', 'CR'
]
restypes_order_r = {restype: i for i, restype in enumerate(restypes_r)}
restypes_num_r = len(restypes_r)


restypes_p = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
    ]
restypes_order_p = {restype: i for i, restype in enumerate(restypes_p)}
restypes_num_p = len(restypes_p)

restypes = restypes_r + restypes_p
restypes_order = {restype: i for i, restype in enumerate(restypes)}
restypes_num = len(restypes)


atom_types_r = []
for key in restypes_r:
    for atom, pos in atom_positions_r[key]:
        atom_types_r.append(atom)
atom_types_r = list(set(atom_types_r))
atom_order_r = {atom_type: i for i, atom_type in enumerate(atom_types_r)}
atom_types_num_r = len(atom_types_r)

atom_types_p = []
for key in restypes_p:
    for atom, pos in atom_positions_p[key]:
        atom_types_p.append(atom)
atom_types_p = list(set(atom_types_p))
atom_order_p = {atom_type: i for i, atom_type in enumerate(atom_types_p)}
atom_types_num_p = len(atom_types_p)

atom_types = atom_types_r + atom_types_p
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_types_num = len(atom_types)



def _make_rigid_transformation_4x4(ex, ey, translation):
  """Create a rigid 4x4 transformation matrix from two axes and transl."""
  # Normalize ex.
  ex_normalized = ex / np.linalg.norm(ex)

  # make ey perpendicular to ex
  ey_normalized = ey - np.dot(ey, ex_normalized) * ex_normalized
  ey_normalized /= np.linalg.norm(ey_normalized)

  # compute ez as cross product
  eznorm = np.cross(ex_normalized, ey_normalized)
  m = np.stack([ex_normalized, ey_normalized, eznorm, translation]).transpose()
  m = np.concatenate([m, [[0., 0., 0., 1.]]], axis=0)
  return m


restype_rigid_frame_r = np.zeros([restypes_num_r, 4, 4], dtype=np.float32)
restype_atom_mask_r = np.zeros([restypes_num_r, atom_types_num_r], dtype=np.float32)
restype_atom_rigid_positions_r = np.zeros([restypes_num_r, atom_types_num_r, 3], dtype=np.float32)


restype_rigid_frame_p = np.zeros([restypes_num_p, 4, 4], dtype=np.float32)
restype_atom_mask_p = np.zeros([restypes_num_p, atom_types_num_p], dtype=np.float32)
restype_atom_rigid_positions_p = np.zeros([restypes_num_p, atom_types_num_p, 3], dtype=np.float32)



def _make_rigid_group_constants_r():
    """Fill the arrays above."""
    

    for restype, restype_letter in enumerate(restypes_r):

        atom_positions = {name: np.array(pos) for name, pos in atom_positions_r[restype_letter]}

        if restype_letter in ['G','A']:
            mat = _make_rigid_transformation_4x4(
                ex=atom_positions['N3'] - atom_positions['N9'],
                ey=atom_positions['N7'] - atom_positions['N9'],
                translation=atom_positions['N9'])
            restype_rigid_frame_r[restype, :, :] = mat
        
        elif restype_letter in ['U','C']:
            mat = _make_rigid_transformation_4x4(
                ex=atom_positions['N3'] - atom_positions['N1'],
                ey=atom_positions['O2'] - atom_positions['N1'],
                translation=atom_positions['N1'])
            restype_rigid_frame_r[restype, :, :] = mat

        elif restype_letter in ['UR', 'GR', 'AR', 'CR']:
            mat = _make_rigid_transformation_4x4(
                ex=atom_positions["C2'"] - atom_positions["C3'"],
                ey=atom_positions["C4'"] - atom_positions["C3'"],
                translation=atom_positions["C3'"])
            restype_rigid_frame_r[restype, :, :] = mat

        else:
            restype_rigid_frame_r[restype, :, :] = np.eye(4)

    for restype, restype_letter in enumerate(restypes_r):
        for atomname, atom_position in atom_positions_r[restype_letter]:
            atomtype = atom_order_r[atomname]
            restype_atom_mask_r[restype, atomtype] = 1
            restype_atom_rigid_positions_r[restype, atomtype, :] = rigid_utils.Rigid.from_tensor_4x4(torch.tensor(restype_rigid_frame_r[restype])).invert_apply(torch.tensor(atom_position))



def _make_rigid_group_constants_p():
    """Fill the arrays above."""
    

    for restype, restype_letter in enumerate(restypes_p):

        restype_rigid_frame_p[restype, :, :] = np.eye(4)

    for restype, restype_letter in enumerate(restypes_p):
        for atomname, atom_position in atom_positions_p[restype_letter]:
            atomtype = atom_order_p[atomname]
            restype_atom_mask_p[restype, atomtype] = 1
            restype_atom_rigid_positions_p[restype, atomtype, :] = rigid_utils.Rigid.from_tensor_4x4(torch.tensor(restype_rigid_frame_p[restype])).invert_apply(torch.tensor(atom_position))



_make_rigid_group_constants_r()
_make_rigid_group_constants_p()


Rigid = rigid_utils.Rigid
Rotation = rigid_utils.Rotation


IDEALIZED_POS_R = torch.tensor(restype_atom_rigid_positions_r)
ATOM_MASK_R = torch.tensor(restype_atom_mask_r)
N_ATOMS_R = atom_types_num_r


IDEALIZED_POS_P = torch.tensor(restype_atom_rigid_positions_p)
ATOM_MASK_P = torch.tensor(restype_atom_mask_p)
N_ATOMS_P = atom_types_num_p


def compute_backbone_r(
        r: torch.Tensor,
        restype: torch.Tensor,
    ):
    """Convert frames to their idealized all atom representation.

    Args:
        r: All rigid groups. [..., N, 8, 3]
        aatype: Residue types. [..., N]

    Returns:

    """

    t_atoms_to_global = r.unsqueeze(2).repeat(1, 1, N_ATOMS_R, 1)
    t_atoms_to_global = Rigid.from_tensor_7(t_atoms_to_global.type(torch.float32))
    
    res_num = restype.cpu().long()
    
    frame_atom_mask = ATOM_MASK_R[res_num, ...].to(r.device)

    
    frame_null_pos = IDEALIZED_POS_R[res_num, ...].to(r.device)
    pred_positions = t_atoms_to_global.apply(frame_null_pos)

    return pred_positions[frame_atom_mask.bool(),:]



def compute_backbone_p(
        r: torch.Tensor,
        restype: torch.Tensor,
    ):
    """Convert frames to their idealized all atom representation.

    Args:
        r: All rigid groups. [..., N, 8, 3]
        aatype: Residue types. [..., N]

    Returns:

    """

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
    """Convert frames to their idealized all atom representation.

    Args:
        r: All rigid groups. [..., N, 8, 3]
        aatype: Residue types. [..., N]

    Returns:

    """

    t_atoms_to_global = r.unsqueeze(2).repeat(1, 1, N_ATOMS_R, 1)
    t_atoms_to_global = Rigid.from_tensor_7(t_atoms_to_global.type(torch.float32))
    
    res_num = restype.cpu().long()
    
    frame_atom_mask = ATOM_MASK_R[res_num, ...].to(r.device).unsqueeze(-1)

    
    frame_null_pos = IDEALIZED_POS_R[res_num, ...].to(r.device)
    pred_positions = t_atoms_to_global.apply(frame_null_pos)

    return pred_positions*frame_atom_mask, frame_atom_mask






'''
atom_positions_r = {
    'U': [
        ['N1', (0.028, 0.464, 2.451)],
        ['C2', (-0.690, -0.671, 2.486)],
        ['O2', (-0.587, -1.474, 1.580)],
        ['N3', (-1.515, -0.936, 3.517)],
        ['C4', (-1.641, -0.055, 4.530)],
        ['O4', (-2.391, -0.292, 5.460)],
        ['C5', (-0.894, 1.146, 4.502)],
        ['C6', (-0.070, 1.384, 3.459)]
    ],
    'G': [
        ['N9', (-0.297, 0.162, -1.534)],
        ['C8', (-1.440, 0.880, -1.334)],
        ['N7', (-2.066, 1.037, -2.464)],
        ['C5', (-1.364, 0.431, -3.453)],
        ['C6', (-1.556, 0.279, -4.846)],
        ['O6', (-2.534, 0.755, -5.397)],
        ['N1', (-0.626, -0.401, -5.551)],
        ['C2', (0.459, -0.934, -4.923)],
        ['N2', (1.384, -1.626, -5.664)],
        ['N3', (0.649, -0.800, -3.630)],
        ['C4', (-0.226, -0.134, -2.868)]
    ],
    'A': [
        ['N9', (0.158, 0.029, 1.803)],
        ['C8', (1.265, 0.813, 1.672)],
        ['N7', (1.843, 0.963, 2.828)],
        ['C5', (1.143, 0.292, 3.773)],
        ['C6', (1.290, 0.091, 5.156)],
        ['N6', (2.344, 0.664, 5.846)],
        ['N1', (0.391, -0.656, 5.787)],
        ['C2', (-0.617, -1.206, 5.136)],
        ['N3', (-0.792, -1.051, 3.841)],
        ['C4', (0.056, -0.320, 3.126)]
    ],
    'C': [
        ['N1', (-0.036, -0.470, 2.453)],
        ['C2', (0.652, 0.683, 2.514)],
        ['O2', (0.529, 1.504, 1.620)],
        ['N3', (1.467, 0.945, 3.535)],
        ['C4', (1.620, 0.070, 4.520)],
        ['N4', (2.464, 0.350, 5.569)],
        ['C5', (0.916, -1.151, 4.483)],
        ['C6', (0.087, -1.399, 3.442)]
    ],
    'UR': [
        ["OP3", (-2.122, 1.033, -4.690)],
        ["P", (-1.030, 0.047, -4.037)],
        ["OP1", (-1.679, -1.228, -3.660)],
        ["OP2", (0.138, -0.241, -5.107)],
        ["O5'", (-0.399, 0.736, -2.726)],
        ["C5'", (0.557, -0.182, -2.196)],
        ["C4'", (1.197, 0.415, -0.942)],
        ["O4'", (0.194, 0.645, 0.074)],
        ["C3'", (2.181, -0.588, -0.301)],
        ["O3'", (3.524, -0.288, -0.686)],
        ["C2'", (1.995, -0.383, 1.218)],
        ["O2'", (3.219, 0.046, 1.819)],
        ["C1'", (0.922, 0.723, 1.319)]
    ],
    'GR': [
        ["OP3", (-1.945, -1.360, 5.599)],
        ["P", (-0.911, -0.277, 5.008)],
        ["OP1", (-1.598, 1.022,  4.844)],
        ["OP2", (0.325, -0.105, 6.025)],
        ["O5'", (-0.365, -0.780, 3.580)],
        ["C5'", (0.542, 0.217, 3.109)],
        ["C4'", (1.100, -0.200, 1.748)],
        ["O4'", (0.033, -0.318, 0.782)],
        ["C3'", (2.025, 0.898, 1.182)],
        ["O3'", (3.395, 0.582, 1.439)],
        ["C2'", (1.741, 0.884, -0.338)],
        ["O2'", (2.927, 0.560, -1.066)],
        ["C1'", (0.675, -0.220, -0.507)]
    ],
    'AR': [
        ["OP3", (2.135, -1.141, -5.313)],
        ["P", (1.024, -0.137, -4.723)],
        ["OP1", (1.633, 1.190, -4.488)],
        ["OP2", (-0.183, 0.005, -5.778)],
        ["O5'", (0.456, -0.720, -3.334)],
        ["C5'", (-0.520, 0.209, -2.863)],
        ["C4'", (-1.101, -0.287, -1.538)],
        ["O4'", (-0.064, -0.383, -0.538)],
        ["C3'", (-2.105, 0.739, -0.969)],
        ["O3'", (-3.445, 0.360, -1.287)],
        ["C2'", (-1.874, 0.684, 0.558)],
        ["O2'", (-3.065, 0.271, 1.231)],
        ["C1'", (-0.755, -0.367, 0.729)]
    ],
    'CR': [
        ["OP3", (2.147, -1.021, -4.678)],
        ["P", (1.049, -0.039, -4.028)],
        ["OP1", (1.692, 1.237, -3.646)],
        ["OP2", (-0.116, 0.246, -5.102)],
        ["O5'", (0.415, -0.733, -2.721)],
        ["C5'", (-0.546, 0.181, -2.193)],
        ["C4'", (-1.189, -0.419, -0.942)],
        ["O4'", (-0.190, -0.648, 0.076)],
        ["C3'", (-2.178, 0.583, -0.307)],
        ["O3'", (-3.518, 0.283, -0.703)],
        ["C2'", (-2.001, 0.373, 1.215)],
        ["O2'", (-3.228, -0.059, 1.806)],
        ["C1'", (-0.924, -0.729, 1.317)]
    ]
}

atom_positions_p = {
        'ALA': [
        ['N', (-0.525, 1.363, 0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.526, -0.000, -0.000)],
        ['CB', (-0.529, -0.774, -1.205)]
    ],
    'ARG': [
        ['N', (-0.524, 1.362, -0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.525, -0.000, -0.000)],
        ['CB', (-0.524, -0.778, -1.209)]
    ],
    'ASN': [
        ['N', (-0.536, 1.357, 0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.526, -0.000, -0.000)],
        ['CB', (-0.531, -0.787, -1.200)]
    ],
    'ASP': [
        ['N', (-0.525, 1.362, -0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.527, 0.000, -0.000)],
        ['CB', (-0.526, -0.778, -1.208)]
    ],
    'CYS': [
        ['N', (-0.522, 1.362, -0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.524, 0.000, 0.000)],
        ['CB', (-0.519, -0.773, -1.212)]
    ],
    'GLN': [
        ['N', (-0.526, 1.361, -0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.526, 0.000, 0.000)],
        ['CB', (-0.525, -0.779, -1.207)]
    ],
    'GLU': [
        ['N', (-0.528, 1.361, 0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.526, -0.000, -0.000)],
        ['CB', (-0.526, -0.781, -1.207)]
    ],
    'GLY': [
        ['N', (-0.572, 1.337, 0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.517, -0.000, -0.000)]
    ],
    'HIS': [
        ['N', (-0.527, 1.360, 0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.525, 0.000, 0.000)],
        ['CB', (-0.525, -0.778, -1.208)]
    ],
    'ILE': [
        ['N', (-0.493, 1.373, -0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.527, -0.000, -0.000)],
        ['CB', (-0.536, -0.793, -1.213)]
    ],
    'LEU': [
        ['N', (-0.520, 1.363, 0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.525, -0.000, -0.000)],
        ['CB', (-0.522, -0.773, -1.214)]
    ],
    'LYS': [
        ['N', (-0.526, 1.362, -0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.526, 0.000, 0.000)],
        ['CB', (-0.524, -0.778, -1.208)]
    ],
    'MET': [
        ['N', (-0.521, 1.364, -0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.525, 0.000, 0.000)],
        ['CB', (-0.523, -0.776, -1.210)]
    ],
    'PHE': [
        ['N', (-0.518, 1.363, 0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.524, 0.000, -0.000)],
        ['CB', (-0.525, -0.776, -1.212)]
    ],
    'PRO': [
        ['N', (-0.566, 1.351, -0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.527, -0.000, 0.000)],
        ['CB', (-0.546, -0.611, -1.293)]
    ],
    'SER': [
        ['N', (-0.529, 1.360, -0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.525, -0.000, -0.000)],
        ['CB', (-0.518, -0.777, -1.211)]
    ],
    'THR': [
        ['N', (-0.517, 1.364, 0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.526, 0.000, -0.000)],
        ['CB', (-0.516, -0.793, -1.215)]
    ],
    'TRP': [
        ['N', (-0.521, 1.363, 0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.525, -0.000, 0.000)],
        ['CB', (-0.523, -0.776, -1.212)]
    ],
    'TYR': [
        ['N', (-0.522, 1.362, 0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.524, -0.000, -0.000)],
        ['CB', (-0.522, -0.776, -1.213)]
    ],
    'VAL': [
        ['N', (-0.494, 1.373, -0.000)],
        ['CA', (0.000, 0.000, 0.000)],
        ['C', (1.527, -0.000, -0.000)],
        ['CB', (-0.533, -0.795, -1.213)]
    ],
}



atom_positions_r = {
    'U': [
        ['N1', (0.028, 0.464, 2.451)],
        ['C2', (-0.690, -0.671, 2.486)],
        ['O2', (-0.587, -1.474, 1.580)],
        ['N3', (-1.515, -0.936, 3.517)],
        ['C4', (-1.641, -0.055, 4.530)],
        ['O4', (-2.391, -0.292, 5.460)],
        ['C5', (-0.894, 1.146, 4.502)],
        ['C6', (-0.070, 1.384, 3.459)]
    ],
    'G': [
        ['N9', (-0.297, 0.162, -1.534)],
        ['C8', (-1.440, 0.880, -1.334)],
        ['N7', (-2.066, 1.037, -2.464)],
        ['C5', (-1.364, 0.431, -3.453)],
        ['C6', (-1.556, 0.279, -4.846)],
        ['O6', (-2.534, 0.755, -5.397)],
        ['N1', (-0.626, -0.401, -5.551)],
        ['C2', (0.459, -0.934, -4.923)],
        ['N2', (1.384, -1.626, -5.664)],
        ['N3', (0.649, -0.800, -3.630)],
        ['C4', (-0.226, -0.134, -2.868)]
    ],
    'A': [
        ['N9', (0.158, 0.029, 1.803)],
        ['C8', (1.265, 0.813, 1.672)],
        ['N7', (1.843, 0.963, 2.828)],
        ['C5', (1.143, 0.292, 3.773)],
        ['C6', (1.290, 0.091, 5.156)],
        ['N6', (2.344, 0.664, 5.846)],
        ['N1', (0.391, -0.656, 5.787)],
        ['C2', (-0.617, -1.206, 5.136)],
        ['N3', (-0.792, -1.051, 3.841)],
        ['C4', (0.056, -0.320, 3.126)]
    ],
    'C': [
        ['N1', (-0.036, -0.470, 2.453)],
        ['C2', (0.652, 0.683, 2.514)],
        ['O2', (0.529, 1.504, 1.620)],
        ['N3', (1.467, 0.945, 3.535)],
        ['C4', (1.620, 0.070, 4.520)],
        ['N4', (2.464, 0.350, 5.569)],
        ['C5', (0.916, -1.151, 4.483)],
        ['C6', (0.087, -1.399, 3.442)]
    ],
    'UR': [
        ["C5'", (198.210, 136.843, 126.521)],
        ["C4'", (198.152, 136.215, 127.896)],
        ["O4'", (198.325, 137.239, 128.914)],
        ["C3'", (196.843, 135.546, 128.282)],
        ["O3'", (196.708, 134.241, 127.723)],
        ["C2'", (196.943, 135.524, 129.802)],
        ["O2'", (197.798, 134.502, 130.278)],
        ["C1'", (197.586, 136.886, 130.076)] 
    ],
    'GR': [
        ["C5'", (124.553, 108.841, 167.200)],
        ["C4'", (125.185, 107.758, 168.048)],
        ["O4'", (126.115, 106.984, 167.242)],
        ["C3'", (126.031, 108.227, 169.223)],
        ["O3'", (125.220, 108.572, 170.346)],
        ["C2'", (126.909, 107.008, 169.487)],
        ["O2'", (126.238, 105.991, 170.206)],
        ["C1'", (127.174, 106.510, 168.062)]
    ],
    'AR': [
        ["C5'", (126.529, 107.531, 172.838)],
        ["C4'", (127.790, 106.775, 173.189)],
        ["O4'", (128.591, 106.571, 171.997)],
        ["C3'", (128.737, 107.454, 174.168)],
        ["O3'", (128.283, 107.203, 175.500)],
        ["C2'", (130.068, 106.765, 173.865)],
        ["O2'", (130.187, 105.503, 174.491)],
        ["C1'", (129.963, 106.536, 172.352)]
    ],
    'CR': [
        ["C5'", (127.838, 105.828, 177.814)],
        ["C4'", (127.993, 104.898, 178.997)],
        ["O4'", (127.203, 103.704, 178.737)],
        ["C3'", (129.400, 104.373, 179.250)],
        ["O3'", (130.157, 105.279, 180.057)],
        ["C2'", (129.133, 103.049, 179.962)],
        ["O2'", (128.817, 103.215, 181.331)],
        ["C1'", (127.882, 102.557, 179.226)]
    ]
}
'''