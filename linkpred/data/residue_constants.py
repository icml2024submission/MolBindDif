from openfold.utils import rigid_utils

import numpy as np
import torch 

#_chem_comp_atom.pdbx_model_Cartn_x_ideal 
rigid_group_atom_positions = {
    'U': [
        ['P', 3, (0.799, 1.377, 0.0)],
        ['OP1', 4, (0.465, -0.925, -1.059)],
        ['OP2', 4, (0.458, 1.408, 0.0)],
        ["O5'", 2, (0.555, 1.304, 0.0)],
        ["C5'", 1, (0.681, 1.348, -0.0)],
        ["C4'", 0, (-0.309, 1.476, -0.0)],
        ["O4'", 6, (0.334, 1.411, 0.0)],
        ["C3'", 0, (0.0, 0.0, -0.0)],
        ["O3'", 0, (-0.567, -0.684, -1.096)],
        ["C2'", 0, (1.519, 0.0, -0.0)],
        ["O2'", 5, (0.496, -0.55, -1.21)],
        ["C1'", 5, (0.296, 1.502, -0.0)]
    ],
    'G': [
        ['P', 3, (0.819, 1.374, 0.0)],
        ['OP1', 4, (0.546, -1.029, -0.929)],
        ['OP2', 4, (0.373, 1.435, 0.0)],
        ["O5'", 2, (0.528, 1.321, 0.0)],
        ["C5'", 1, (0.675, 1.349, -0.0)],
        ["C4'", 0, (-0.33, 1.484, 0.0)],
        ["O4'", 6, (0.341, 1.406, 0.0)],
        ["C3'", 0, (0.0, -0.0, 0.0)],
        ["O3'", 0, (-0.578, -0.746, -1.06)],
        ["C2'", 0, (1.522, -0.0, -0.0)],
        ["O2'", 5, (0.525, -0.592, -1.178)],
        ["C1'", 5, (0.294, 1.503, -0.0)]
    ],
    'A': [
        ['P', 3, (0.833, 1.352, -0.0)],
        ['OP1', 4, (0.538, -0.949, -1.002)],
        ['OP2', 4, (0.128, 1.476, 0.0)],
        ["O5'", 2, (0.477, 1.346, -0.0)],
        ["C5'", 1, (0.62, 1.363, 0.0)],
        ["C4'", 0, (-0.35, 1.466, -0.0)],
        ["O4'", 6, (0.342, 1.399, 0.0)],
        ["C3'", 0, (0.0, 0.0, 0.0)],
        ["O3'", 0, (-0.544, -0.649, -1.112)],
        ["C2'", 0, (1.507, 0.0, -0.0)],
        ["O2'", 5, (0.473, -0.558, -1.194)],
        ["C1'", 5, (0.318, 1.491, 0.0)]
    ],
    'C': [
        ['P', 3, (0.757, 1.385, 0.0)],
        ['OP1', 4, (0.493, -0.937, -1.054)],
        ['OP2', 4, (0.332, 1.443, 0.0)],
        ["O5'", 2, (0.526, 1.311, 0.0)],
        ["C5'", 1, (0.68, 1.336, -0.0)],
        ["C4'", 0, (-0.313, 1.485, -0.0)],
        ["O4'", 6, (0.347, 1.403, 0.0)],
        ["C3'", 0, (0.0, 0.0, 0.0)],
        ["O3'", 0, (-0.515, -0.654, -1.141)],
        ["C2'", 0, (1.518, 0.0, 0.0)],
        ["O2'", 5, (0.468, -0.559, -1.228)],
        ["C1'", 5, (0.314, 1.506, -0.0)]
    ]
}

restypes = [
    'U', 'G', 'A', 'C'
]
restypes_order = {restype: i for i, restype in enumerate(restypes)}
restypes_num = len(restypes)




atom_types = []
for key in restypes:
    for atom, group_idx, pos in rigid_group_atom_positions[key]:
        atom_types.append(atom)
atom_types = list(set(atom_types))
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


restype_atom_to_rigid_group = np.zeros([restypes_num, atom_types_num], dtype=int)
restype_atom_mask = np.zeros([restypes_num, atom_types_num], dtype=np.float32)
restype_atom_rigid_group_positions = np.zeros([restypes_num, atom_types_num, 3], dtype=np.float32)
restype_rigid_group_default_frame = np.zeros([restypes_num, 7, 4, 4], dtype=np.float32)




def _make_rigid_group_constants():
    for restype, resname in enumerate(restypes):
        
        for atomname, group_idx, atom_position in rigid_group_atom_positions[resname]:
            atomtype = atom_order[atomname]
            restype_atom_to_rigid_group[restype, atomtype] = group_idx
            restype_atom_mask[restype, atomtype] = 1
            restype_atom_rigid_group_positions[
                restype, atomtype, :
            ] = atom_position


    for restype, resname in enumerate(restypes):
        atom_positions = {
            name: np.array(pos)
            for name, _, pos in rigid_group_atom_positions[resname]
        }

        # backbone to backbone
        restype_rigid_group_default_frame[restype, 0, :, :] = np.eye(4)

        # delta-frame to backbone
        mat = _make_rigid_transformation_4x4(
            ex=atom_positions["C4'"],
            ey=np.array([1.0, 0.0, 0.0]),
            translation=atom_positions["C4'"],
        )
        restype_rigid_group_default_frame[restype, 6, :, :] = mat

        # gamma-frame to delta-frame
        mat = _make_rigid_transformation_4x4(
            ex=atom_positions["C5'"],
            ey=np.array([-1.0, 0.0, 0.0]),
            translation=atom_positions["C5'"],
        )
        restype_rigid_group_default_frame[restype, 5, :, :] = mat

        # beta-frame to gamma-frame
        mat = _make_rigid_transformation_4x4(
            ex=atom_positions["O5'"],
            ey=np.array([-1.0, 0.0, 0.0]),
            translation=atom_positions["O5'"],
        )
        restype_rigid_group_default_frame[restype, 4, :, :] = mat

        # alpha-frame to beta-frame
        mat = _make_rigid_transformation_4x4(
            ex=atom_positions["P"],
            ey=np.array([-1.0, 0.0, 0.0]),
            translation=atom_positions["P"],
        )
        restype_rigid_group_default_frame[restype, 3, :, :] = mat

        # nu2-frame to backbone
        mat = _make_rigid_transformation_4x4(
            ex=atom_positions["C2'"],
            ey=np.array([0.0, 1.0, 0.0]),
            translation=atom_positions["C2'"],
        )
        restype_rigid_group_default_frame[restype, 1, :, :] = mat

        # nu3-frame to backbone
        mat = _make_rigid_transformation_4x4(
            ex=atom_positions["C4'"],
            ey=np.array([1.0, 0.0, 0.0]),
            translation=atom_positions["C4'"],
        )
        restype_rigid_group_default_frame[restype, 2, :, :] = mat


_make_rigid_group_constants()






