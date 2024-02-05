# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from data import residue_constants as b_residue_constants
from linkpred.data import residue_constants as s_residue_constants
import numpy as np
import torch
import os
import re
from data import all_atom






class RNP:
  def __init__(self, rna_pos, restype, protpdb):

    rna_atom_pos, atom_mask = all_atom.compute_sparse_backbone_r(torch.tensor(rna_pos).unsqueeze(0), torch.tensor(restype).unsqueeze(0))

    self.atom_positions = rna_atom_pos.squeeze(0).numpy()
    self.restype = restype
    self.atom_mask = atom_mask.squeeze(0).numpy()
    self.protpdb = protpdb

    

  

  def to_pdb(self, model=1) -> str:

    atom_types = b_residue_constants.atom_types_r
    res_names = b_residue_constants.restypes_r
    
    pdb_lines = [f'MODEL     {model}'] + self.protpdb

    atom_index = 1
    num_nucs = int(len(self.restype)/2)
    
    for i in range(num_nucs):
      
      aa =  res_names[self.restype[i]]

      for atom_name, pos, mask in zip(
            atom_types, self.atom_positions[i], self.atom_mask[i]):
        if mask < 0.5:
            continue
        record_type = 'ATOM'
        name = atom_name if len(atom_name) == 4 else f' {atom_name}'
        alt_loc = ''
        insertion_code = ''
        occupancy = 1.00
        b_factor = 80.00
        element = atom_name[0]  # Protein supports only C, N, O, S, this works.
        charge = ''
        # PDB is a columnar format, every space matters here!
        atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                      f'{aa:>3} B'
                      f'{i:>4}{insertion_code:>1}   '
                      f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                      f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                      f'{element:>2}{charge:>2}')
        pdb_lines.append(atom_line)
        atom_index += 1
      
      for atom_name, pos, mask in zip(
            atom_types, self.atom_positions[i+num_nucs], self.atom_mask[i+num_nucs]):
        if mask < 0.5:
            continue
        record_type = 'ATOM'
        name = atom_name if len(atom_name) == 4 else f' {atom_name}'
        alt_loc = ''
        insertion_code = ''
        occupancy = 1.00
        b_factor = 80.00
        element = atom_name[0]  # Protein supports only C, N, O, S, this works.
        charge = ''
        # PDB is a columnar format, every space matters here!
        atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                      f'{aa:>3} B'
                      f'{i:>4}{insertion_code:>1}   '
                      f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                      f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                      f'{element:>2}{charge:>2}')
        pdb_lines.append(atom_line)
        atom_index += 1


    pdb_lines.append('ENDMDL')
    # Pad all lines to 80 characters.
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return '\n'.join(pdb_lines) + '\n'




def write_rnp_to_pdb(
        rna_pos: np.ndarray,
        file_path: str,
        restype,
        protein: str,
        overwrite=False,
        no_indexing=False
    ):
    if overwrite:
        max_existing_idx = 0
    else:
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path).strip('.pdb')
        existing_files = [x for x in os.listdir(file_dir) if file_name in x]
        max_existing_idx = max([
            int(re.findall(r'_(\d+).pdb', x)[0]) for x in existing_files if re.findall(r'_(\d+).pdb', x)
            if re.findall(r'_(\d+).pdb', x)] + [0])
    if not no_indexing:
        save_path = file_path.replace('.pdb', '') + f'_{max_existing_idx+1}.pdb'
    else:
        save_path = file_path
    with open(save_path, 'w') as f:
        if rna_pos.ndim == 3:
            for t, pos in enumerate(rna_pos):
                pdb_rnp = RNP(pos, restype, protein).to_pdb(model=t + 1)
                f.write(pdb_rnp)
        elif rna_pos.ndim == 2:
            pdb_rnp = RNP(rna_pos, restype, protein).to_pdb(model=1)
            f.write(pdb_rnp)

        else:
            raise ValueError(f'Invalid positions shape {rna_pos.shape}')
        f.write('END')
    return save_path



def write_to_pdb(s_atom_pos,
                b_atom_pos,
                b_atom_mask,
                restype,
                protein
    ):
   
    s_atom_types = s_residue_constants.atom_types
    s_res_names = s_residue_constants.restypes

    b_atom_types = b_residue_constants.atom_types_r
    b_res_names = b_residue_constants.restypes_r
    
    pdb_lines =  protein

    atom_index = 1
    num_nucs = int(len(restype)/2)
    
    for i in range(num_nucs):
      
      aa =  b_res_names[restype[i]]

      for atom_name, pos, mask in zip(
            b_atom_types, b_atom_pos[i], b_atom_mask[i]):
        if mask < 0.5:
            continue
        record_type = 'ATOM'
        name = atom_name if len(atom_name) == 4 else f' {atom_name}'
        alt_loc = ''
        insertion_code = ''
        occupancy = 1.00
        b_factor = 80.00
        element = atom_name[0]  # Protein supports only C, N, O, S, this works.
        charge = ''
        # PDB is a columnar format, every space matters here!
        atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                      f'{aa:>3} B'
                      f'{i+1:>4}{insertion_code:>1}   '
                      f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                      f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                      f'{element:>2}{charge:>2}')
        pdb_lines.append(atom_line)
        atom_index += 1
      
      for atom_name, pos in zip(
            s_atom_types, s_atom_pos[i]):
        
        record_type = 'ATOM'
        name = atom_name if len(atom_name) == 4 else f' {atom_name}'
        alt_loc = ''
        insertion_code = ''
        occupancy = 1.00
        b_factor = 80.00
        element = atom_name[0]  # Protein supports only C, N, O, S, this works.
        charge = ''
        # PDB is a columnar format, every space matters here!
        atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                      f'{aa:>3} B'
                      f'{i+1:>4}{insertion_code:>1}   '
                      f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                      f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                      f'{element:>2}{charge:>2}')
        pdb_lines.append(atom_line)
        atom_index += 1

    # Pad all lines to 80 characters.
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return '\n'.join(pdb_lines) + '\n'