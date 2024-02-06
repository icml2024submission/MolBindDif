import os
import sys
import time
import tree
import numpy as np
import hydra
import torch
import logging
import copy
import pandas as pd
from datetime import datetime
import GPUtil

from data import all_atom as bd_all_atom
from linkpred.data import all_atom as lp_all_atom
from data import utils as du
from openfold.utils import rigid_utils as ru
from data import ppi_data_loader, rnp
from typing import Dict
from experiments import train_bd
from omegaconf import DictConfig, OmegaConf
from experiments import utils as eu
from linkpred.experiments import train_lp






class Inference:

    def __init__(
            self,
            conf: DictConfig
        ):

        self._log = logging.getLogger(__name__)

        # Remove static type checking.
        OmegaConf.set_struct(conf, False)

        
        # Prepare configs.
        self._conf = conf
        self.bd_conf = conf.bd
        self.lp_conf = conf.lp
        self._diff_conf = self.bd_conf.diffusion


        # Set-up accelerator
        if torch.cuda.is_available():
            if self.bd_conf.gpu_id is None:
                available_gpus = ''.join(
                    [str(x) for x in GPUtil.getAvailable(
                        order='memory', limit = 8)])
                self.device = f'cuda:{available_gpus[0]}'
            else:
                self.device = f'cuda:{self.bd_conf.gpu_id}'
        else:
            self.device = 'cpu'
        self._log.info(f'Using device: {self.device}')

        # Set-up directories
        self.bd_model_path = self.bd_conf.model_path
        for root, _, files in os.walk(self.bd_model_path):
            for file in files:
                if file.endswith('.pth'):
                    self.bd_weights_path = os.path.join(root, file)

        self.lp_model_path = self.lp_conf.model_path
        for root, _, files in os.walk(self.lp_model_path):
            for file in files:
                if file.endswith('.pth'):
                    self.lp_weights_path = os.path.join(root, file)
        

        output_dir = self.bd_conf.output_dir
        if self.bd_conf.name is None:
            dt_string = datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss")
        else:
            dt_string = self.bd_conf.name
        self._output_dir = os.path.join(output_dir, dt_string)
        os.makedirs(os.path.join(self._output_dir, 'data'), exist_ok=True)
        self._log.info(f'Saving results to {self._output_dir}')

        config_path = os.path.join(self._output_dir, 'data/inference_conf.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(config=self._conf, f=f)
        self._log.info(f'Saving inference config to {config_path}')

        # Load models and experiment
        #self.valid_csv = pd.read_csv(self._valid_set_path)
        self.names = pd.read_csv(self.bd_conf.input_file).ppi3d_name.values
        self._load_ckpt()
        
       
        

    def _load_ckpt(self):
        """Loads in model checkpoint."""
        self._log.info(f'Loading weights from {self.bd_weights_path}')
        self._log.info(f'Loading weights from {self.lp_weights_path}')

        # Read checkpoint and create experiment.
        bd_weights_pkl = du.read_pkl(self.bd_weights_path, use_torch=True, map_location=self.device)
        lp_weights_pkl = du.read_pkl(self.lp_weights_path, use_torch=True, map_location=self.device)

        # Merge base experiment config with checkpoint config.
        self.bd_base_conf = bd_weights_pkl['conf']
        self.lp_base_conf = lp_weights_pkl['conf']

        # Prepare model
        self.bd_base_conf.experiment.ckpt_dir = None
        self.bd_base_conf.experiment.warm_start = None
        self.bd_exp = train_bd.Experiment(conf=self.bd_base_conf)
        self.bd_model = self.bd_exp.model

        self.lp_base_conf.experiment.ckpt_dir = None
        self.lp_base_conf.experiment.warm_start = None
        self.lp_exp = train_lp.Experiment(conf=self.lp_base_conf)
        self.lp_model = self.lp_exp.model

        # Remove module prefix if it exists.
        bd_model_weights = bd_weights_pkl['model']
        bd_model_weights = {k.replace('module.', ''):v for k,v in bd_model_weights.items()}
        self.bd_model.load_state_dict(bd_model_weights)
        self.bd_model = self.bd_model.to(self.device)
        self.bd_model.eval()
        self.diffuser = self.bd_exp.diffuser

        lp_model_weights = lp_weights_pkl['model']
        lp_model_weights = {k.replace('module.', ''):v for k,v in lp_model_weights.items()}
        self.lp_model.load_state_dict(lp_model_weights)
        self.lp_model = self.lp_model.to(self.device)
        self.lp_model.eval()



    def run_inference(self):

        

        self.valid_csv = du.process_data(self._output_dir, self.names)

        valid_loader = self.create_valid_loader()

        return self.inference_fn(
            self._output_dir, 
            valid_loader, 
            self.device, 
            min_t=self._diff_conf.min_t, 
            num_t=self._diff_conf.num_t, 
            noise_scale=self._diff_conf.noise_scale)




    def create_valid_loader(self):
        valid_csv = self.valid_csv
        
        valid_dataset = ppi_data_loader.PPIDataset(
            csv = valid_csv,
            data_conf=None,
            diffuser=self.diffuser,
            is_training=False,
            is_validating=True
        )


        valid_sampler = ppi_data_loader.ValidSampler(
                data_conf=None,
                dataset=valid_dataset,
                nof_samples=self.bd_conf.nof_samples,
            )

        # Loaders
        num_workers = 5

        valid_loader = du.create_data_loader(
            valid_dataset,
            sampler=valid_sampler,
            np_collate=False,
            length_batch=False,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

        
        return valid_loader

    

    def inference_fn(self, out_dir, valid_loader, device, min_t=None, num_t=None, noise_scale=1.0):
        
        for valid_feats, pdb_name in valid_loader:

            pdb_name = pdb_name[0]
            protein_pdb = open(os.path.join(out_dir, f'data/{pdb_name}.pdb'), 'r').readlines()
            protein_pdb = [line.strip() for line in protein_pdb if line[20:22].strip() == 'A']

            valid_feats = tree.map_structure(lambda x: x.to(device), valid_feats)

            # Run inference
            bd_output = self.bd_exp.inference_fn(valid_feats, num_t=num_t, min_t=min_t, noise_scale=noise_scale)

            rest_feats = self.make_rest_feats(bd_output['T_r_0'], valid_feats['rtype_r'], valid_feats['ridx_r'])

            lp_output = self.lp_exp.inference_fn(rest_feats)

            s_atom_pos, b_atom_pos, b_atom_mask = self.extract_atom_pos(bd_output, lp_output, valid_feats, rest_feats)

            sample_path = os.path.join(out_dir, f'pred_{pdb_name}.pdb')

            pdb_rnp = rnp.write_to_pdb(
                s_atom_pos=du.move_to_np(s_atom_pos.squeeze(0)),
                b_atom_pos=du.move_to_np(b_atom_pos.squeeze(0)),
                b_atom_mask=du.move_to_np(b_atom_mask.squeeze(0)),
                restype=du.move_to_np(valid_feats['rtype_r'].squeeze(0)),
                protein=protein_pdb
            )
            with open(sample_path, 'w') as f:
                f.write(pdb_rnp)
            
            self._log.info(f'Done sample {pdb_name}: {sample_path}')
            
        return 0

    def make_rest_feats(self, T_r, rtype_r, ridx_r):
        num_res = int(rtype_r.shape[-1] / 2)

        com = T_r[:,num_res:].get_trans().mean(dim=-2)
        new_T_r = self.to_com(T_r[:,num_res:], com)

        rest_feats = {
            'T_r': new_T_r.to_tensor_7(),
            'rtype_r': rtype_r[:,num_res:]-4,
            'ridx_r': ridx_r[:,num_res:],
            'com': com
        }
        rest_feats['sc_r'] = torch.clone(new_T_r.get_trans())
        
        return rest_feats
      
        
    def extract_atom_pos(self, bd_output, lp_output, v_feats, r_feats):

        T_r = self.out_of_com(bd_output['T_r_0'], v_feats['com'])
        num_res = r_feats['rtype_r'].shape[-1]

        pred_all_frames_to_global = lp_all_atom.torsion_angles_to_frames(T_r[:,num_res:], lp_output['angles'], r_feats['rtype_r'])
        s_atoms_pos = lp_all_atom.frames_and_literature_positions_to_atom_pos(pred_all_frames_to_global, r_feats['rtype_r'])

        bs_atom_pos, atom_mask = bd_all_atom.compute_sparse_backbone_r(T_r, v_feats['rtype_r'])
        b_atom_pos, b_atom_mask = bs_atom_pos[:num_res], atom_mask[:num_res]

        return s_atoms_pos, b_atom_pos, b_atom_mask


    def out_of_com(self, T_r, com):
        fn_out_of_com = lambda x: x + com
        T_r = copy.deepcopy(T_r).apply_trans_fn(fn_out_of_com).to_tensor_7()
        return T_r
    
    def to_com(self, T_r, com):
        fn_out_of_com = lambda x: x - com
        T_r = copy.deepcopy(T_r).apply_trans_fn(fn_out_of_com)
        return T_r

 

@hydra.main(version_base=None, config_path="./config", config_name="inference")
def run(conf: DictConfig) -> None:

    # Read model checkpoint.
    print('Starting inference')
    start_time = time.time()
    inference = Inference(conf)
    inference.run_inference()
    elapsed_time = time.time() - start_time
    print(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    run()
