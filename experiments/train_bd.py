"""Pytorch script for training SE(3) protein diffusion.

To run:

> python experiments/train_se3_diffusion.py


To modify config options with the command line,

> python experiments/train_se3_diffusion.py experiment.batch_size=32

"""
import os
import torch
import GPUtil
import time
import tree
import numpy as np
import copy
import hydra
import logging
import copy
import random
import urllib.request
import gc

from collections import defaultdict
from collections import deque
from datetime import datetime
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from openfold.utils import rigid_utils as ru
from hydra.core.hydra_config import HydraConfig

from data import rnp
from data import ppi_data_loader
from data import residue_constants
from data import se3_diffuser
from data import utils as du
from data import all_atom
from model import main_network
from experiments import utils as eu



class Experiment:

    def __init__(
            self,
            *,
            conf: DictConfig,
        ):
        """Initialize experiment.

        Args:
            exp_cfg: Experiment configuration.
        """
        self._log = logging.getLogger(__name__)
        self._available_gpus = ''.join(
            [str(x) for x in GPUtil.getAvailable(
                order='memory', limit = 8)])

        # Configs
        self._conf = conf
        self._exp_conf = conf.experiment
        if HydraConfig.initialized() and 'num' in HydraConfig.get().job:
            self._exp_conf.name = (
                f'{self._exp_conf.name}_{HydraConfig.get().job.num}')
        self._diff_conf = conf.diffuser
        self._model_conf = conf.model
        self._data_conf = conf.data
        self._use_ddp = self._exp_conf.use_ddp

        if self._use_ddp :
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            dist.init_process_group(backend='nccl')
            self.ddp_info = eu.get_ddp_info()
            if self.ddp_info['rank'] not in [0,-1]:
                self._log.addHandler(logging.NullHandler())
                self._log.setLevel("ERROR")
                self._exp_conf.ckpt_dir = None
        # Warm starting
        ckpt_model = None
        ckpt_opt = None
        self.trained_epochs = 0
        self.trained_steps = 0
        if conf.experiment.warm_start:
            ckpt_dir = conf.experiment.warm_start
            self._log.info(f'Warm starting from: {ckpt_dir}')
            ckpt_files = [
                x for x in os.listdir(ckpt_dir)
                if 'pkl' in x or '.pth' in x
            ]
            if len(ckpt_files) != 1:
                raise ValueError(f'Ambiguous ckpt in {ckpt_dir}')
            ckpt_name = ckpt_files[0]
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            self._log.info(f'Loading checkpoint from {ckpt_path}')
            ckpt_pkl = du.read_pkl(ckpt_path, use_torch=True)
            ckpt_model = ckpt_pkl['model']

            if conf.experiment.use_warm_start_conf:
                OmegaConf.set_struct(conf, False)
                conf = OmegaConf.merge(conf, ckpt_pkl['conf'])
                OmegaConf.set_struct(conf, True)
            conf.experiment.warm_start = ckpt_dir

            # For compatibility with older checkpoints.
            if 'optimizer' in ckpt_pkl:
                ckpt_opt = ckpt_pkl['optimizer']
            if 'epoch' in ckpt_pkl:
                self.trained_epochs = ckpt_pkl['epoch']
            if 'step' in ckpt_pkl:
                self.trained_steps = ckpt_pkl['step']

        # Initialize experiment objects
        self._diffuser = se3_diffuser.SE3Diffuser(self._diff_conf)
        self._model = main_network.ScoreNetwork(self._model_conf, self.diffuser)

        if ckpt_model is not None:
            ckpt_model = {k.replace('module.', ''):v for k,v in ckpt_model.items()}
            self._model.load_state_dict(ckpt_model, strict=True)

        # Set environment variables for which GPUs to use.
        if HydraConfig.initialized() and 'num' in HydraConfig.get().job:
            replica_id = int(HydraConfig.get().job.num)
        else:
            replica_id = 0
       
        assert(not self._exp_conf.use_ddp or self._exp_conf.use_gpu)

        # GPU mode
        if torch.cuda.is_available() and self._exp_conf.use_gpu:
            # single GPU mode
            if self._exp_conf.num_gpus==1 :
                gpu_id = self._available_gpus[replica_id]
                self.device = f"cuda:{gpu_id}"
        else:
            self.device = 'cpu'

        self._model = self.model.to(self.device)
        self._log.info(f"Using device: {self.device}")


        num_parameters = sum(p.numel() for p in self._model.parameters())
        self._exp_conf.num_parameters = num_parameters
        self._log.info(f'Number of model parameters {num_parameters}')
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._exp_conf.learning_rate, betas=(0.9, 0.999), eps=1e-06, amsgrad=True)
        if ckpt_opt is not None:
            self._optimizer.load_state_dict(ckpt_opt)
        self._scheduler = torch.optim.lr_scheduler.ExponentialLR(self._optimizer, gamma=0.9)

        dt_string = datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss")
        if self._exp_conf.ckpt_dir is not None:
            # Set-up checkpoint location
            ckpt_dir = os.path.join(
                self._exp_conf.ckpt_dir,
                self._exp_conf.name,
                dt_string)
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir, exist_ok=True)
            self._exp_conf.ckpt_dir = ckpt_dir
            self._log.info(f'Checkpoints saved to: {ckpt_dir}')
        else:  
            self._log.info('Checkpoint not being saved.')
        if self._exp_conf.eval_dir is not None :
            eval_dir = os.path.join(
                self._exp_conf.eval_dir,
                self._exp_conf.name,
                dt_string)
            self._exp_conf.eval_dir = eval_dir
            if not os.path.exists(eval_dir):
                os.makedirs(eval_dir, exist_ok=True)
            if not os.path.exists(eval_dir+'/pdb_files'):
                os.makedirs(eval_dir+'/pdb_files', exist_ok=True)
            self._log.info(f'Evaluation saved to: {eval_dir}')
        else:
            self._exp_conf.eval_dir = os.devnull
            self._log.info(f'Evaluation will not be saved.')
        self._aux_data_history = deque(maxlen=100)

        

    @property
    def diffuser(self):
        return self._diffuser

    @property
    def model(self):
        return self._model

    @property
    def conf(self):
        return self._conf

    def create_dataset(self):

        train_csv, test_csv = ppi_data_loader.create_split(self._conf)
        test_csv.to_csv(self._exp_conf.eval_dir + '/test_samples.csv')
        # Datasets
        train_dataset = ppi_data_loader.PPIDataset(
            csv = train_csv,
            data_conf=self._data_conf,
            diffuser=self._diffuser,
            is_training=True,
            is_validating=False
        )

        test_dataset = ppi_data_loader.PPIDataset(
            csv = test_csv,
            data_conf=self._data_conf,
            diffuser=self._diffuser,
            is_training=False,
            is_validating=False
        )

        valid_dataset = ppi_data_loader.PPIDataset(
            csv = test_csv,
            data_conf=self._data_conf,
            diffuser=self._diffuser,
            is_training=False,
            is_validating=True
        )

        if not self._use_ddp:
            train_sampler = ppi_data_loader.TrainSampler(
                data_conf=self._data_conf,
                dataset=train_dataset,
                batch_size=self._exp_conf.batch_size,
                sample_mode=self._exp_conf.sample_mode,
            )
            test_sampler = ppi_data_loader.TestSampler(
                data_conf=self._data_conf,
                dataset=test_dataset,
                batch_size=self._exp_conf.batch_size,
                sample_mode=self._exp_conf.sample_mode,
            )
            valid_sampler = ppi_data_loader.ValidSampler(
                data_conf=self._data_conf,
                dataset=valid_dataset,
                nof_samples=self._exp_conf.nof_samples
            )
        else:
            train_sampler = ppi_data_loader.DistributedTrainSampler(
                data_conf=self._data_conf,
                dataset=train_dataset,
                batch_size=self._exp_conf.batch_size,
            )
            test_sampler = ppi_data_loader.DistributedTestSampler(
                    data_conf=self._data_conf,
                    dataset=test_dataset,
                    batch_size=self._exp_conf.batch_size,
                )
            valid_sampler = ppi_data_loader.DistributedValidSampler(
                    data_conf=self._data_conf,
                    dataset=valid_dataset,
                    nof_samples=self._exp_conf.nof_samples,
                )

        # Loaders
        num_workers = self._exp_conf.num_loader_workers
        train_loader = du.create_data_loader(
            train_dataset,
            sampler=train_sampler,
            np_collate=False,
            length_batch=False,
            batch_size=self._exp_conf.batch_size if not self._exp_conf.use_ddp else self._exp_conf.batch_size // self.ddp_info['world_size'],
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            max_squared_res=self._exp_conf.max_squared_res,
        )
        test_loader = du.create_data_loader(
            test_dataset,
            sampler=test_sampler,
            np_collate=False,
            length_batch=False,
            batch_size=self._exp_conf.batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
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

        return train_loader, test_loader, valid_loader, train_sampler, test_sampler



    def start_training(self, return_logs=False):
        
        self._model.train()
        self._optimizer.zero_grad()

        (
            train_loader,
            test_loader,
            valid_loader,
            train_sampler,
            test_sampler
        ) = self.create_dataset()

        logs = []
        for epoch in range(self.trained_epochs, self._exp_conf.num_epoch):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if test_sampler is not None:
                test_sampler.set_epoch(epoch)
            self.trained_epochs = epoch
    
            epoch_log = self.train_epoch(
                train_loader,
                test_loader,
                return_logs=return_logs
            )

            if epoch%5 == 0:
                self._scheduler.step()
            if return_logs:
                logs.append(epoch_log)

        self._log.info('Start evaluating')

        rmsd = self.eval_fn(self._exp_conf.eval_dir, valid_loader)

        self._log.info(f'Evalution is done, mean RMSD: {rmsd}')

        self._log.info('Done')
        return logs


    def train_epoch(self, train_loader, valid_loader, return_logs=False):
        log_lossses = defaultdict(list)
        global_logs = []
        log_time = time.time()
        step_time = time.time()

        # Training
        for train_feats in train_loader:
            
            train_feats = tree.map_structure(
                lambda x: x.to(self.device), train_feats)
            
            loss, aux_data = self.update_fn(train_feats)
            

            if torch.isnan(loss):
                raise Exception(f'NaN encountered')
            
            if return_logs:
                global_logs.append(loss)
            for k,v in aux_data.items():
                log_lossses[k].append(du.move_to_np(v))
            self.trained_steps += 1

            # Logging to terminal train loss
            if self.trained_steps == 1 or self.trained_steps % self._exp_conf.log_freq == 0:
                elapsed_time = time.time() - log_time
                log_time = time.time()
                step_per_sec = self._exp_conf.log_freq / elapsed_time
                rolling_losses = tree.map_structure(np.mean, log_lossses)
                loss_log = ' '.join([
                    f'{k}={v[0]:.4f}'
                    for k,v in rolling_losses.items() if 'batch' not in k
                ])
                self._log.info(
                    f'[Train {self.trained_steps}]: {loss_log}, steps/sec={step_per_sec:.5f}')
                log_lossses = defaultdict(list)

        # Testing
        print(f'End of training for epoch {self.trained_epochs+1}!')
        log_lossses_test = defaultdict(list)
        for test_feats in valid_loader:

            test_feats = tree.map_structure(lambda x: x.to(self.device), test_feats)
            
            self._model.eval()

            with torch.no_grad():
                loss, aux_data = self.loss_fn(test_feats)

            for k,v in aux_data.items():
                log_lossses_test[k].append(du.move_to_np(v))

         # Logging to terminal test loss
        elapsed_time = time.time() - log_time
        log_time = time.time()
        step_per_sec = self._exp_conf.log_freq / elapsed_time
        rolling_losses = tree.map_structure(np.mean, log_lossses_test)
        loss_log = ' '.join([
            f'{k}={v[0]:.4f}'
            for k,v in rolling_losses.items() if 'batch' not in k
        ])
        self._log.info(
            f'Test [{self.trained_epochs+1}]: {loss_log}, steps/sec={step_per_sec:.5f}')
        log_lossses_test = defaultdict(list)

        # Take checkpoint
        if self._exp_conf.ckpt_dir is not None:
            ckpt_path = os.path.join(
                self._exp_conf.ckpt_dir, f'step_{self.trained_steps}.pth')
            self._ckpt_path = ckpt_path
            du.write_checkpoint(
                ckpt_path,
                copy.deepcopy(self.model.state_dict()),
                self._conf,
                copy.deepcopy(self._optimizer.state_dict()),
                self.trained_epochs,
                self.trained_steps,
                logger=self._log,
                use_torch=True
            )

        if return_logs:
            return global_logs
        

    def update_fn(self, data):
        self._model.train()
        self._optimizer.zero_grad()
        loss, aux_data = self.loss_fn(data)
        '''current_memory = torch.cuda.memory_allocated(self.device)
        print(f"GPU memory usage after loss: {current_memory / (1024 ** 3):.2f} GB")'''
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1)
        
        self._optimizer.step()
        return loss, aux_data


    def _self_conditioning(self, batch):
        model_sc = self.model(batch)
        batch['sc_r'] = model_sc['T_r_0'][..., 4:]
        return batch

    def loss_fn(self, batch):

        if self._model_conf.embed.embed_self_conditioning and random.random() > 0.5:
            with torch.no_grad():
                batch = self._self_conditioning(batch)
        model_out = self.model(batch)

        batch_size, num_res_r = batch['ridx_r'].shape

        # Pairwise distance loss
        pred_atoms_pos_r, num_atoms_r = all_atom.compute_backbone_r(model_out['T_r_0'], batch['rtype_r'])
        gt_atoms_pos_r, _ = all_atom.compute_backbone_r(batch['T_r_0'], batch['rtype_r'])
        gt_atoms_pos_p = all_atom.compute_backbone_p(batch['T_p_0'], batch['rtype_p'])
        
        gt_flat_atoms_r = gt_atoms_pos_r.reshape([batch_size, -1, 3])
        gt_flat_atoms_dist_rr = gt_flat_atoms_r.unsqueeze(-2) - gt_flat_atoms_r.unsqueeze(-3)
        gt_pair_dists_rr = torch.linalg.norm(gt_flat_atoms_dist_rr, dim=-1).float()
        n_atoms_r = gt_flat_atoms_r.shape[1]

        gt_flat_atoms_p = gt_atoms_pos_p.reshape([batch_size, -1, 3])
        gt_flat_atoms_dist_rp = gt_flat_atoms_r.unsqueeze(-2) - gt_flat_atoms_p.unsqueeze(-3)
        gt_pair_dists_rp = torch.linalg.norm(gt_flat_atoms_dist_rp, dim=-1).float()
        n_atoms_p = gt_flat_atoms_p.shape[1]

        pred_flat_atoms_r = pred_atoms_pos_r.reshape([batch_size, -1, 3])
        pred_flat_atoms_dist_rr = pred_flat_atoms_r.unsqueeze(-2) - pred_flat_atoms_r.unsqueeze(-3)
        pred_flat_atoms_dist_rp = pred_flat_atoms_r.unsqueeze(-2) - gt_flat_atoms_p.unsqueeze(-3)
        pred_pair_dists_rr = torch.linalg.norm(pred_flat_atoms_dist_rr, dim=-1).float()
        pred_pair_dists_rp = torch.linalg.norm(pred_flat_atoms_dist_rp, dim=-1).float()

        del gt_flat_atoms_dist_rr, gt_flat_atoms_dist_rp, pred_flat_atoms_dist_rr, pred_flat_atoms_dist_rp, gt_flat_atoms_r, gt_flat_atoms_p, pred_flat_atoms_r
        
        rr_dist_mat_mse = (gt_pair_dists_rr - pred_pair_dists_rr)**2
        rr_dist_mat_loss = torch.sum(rr_dist_mat_mse, dim=(1, 2))
        
        rp_dist_mat_mse = (gt_pair_dists_rp - pred_pair_dists_rp)**2
        rp_dist_mat_loss = torch.sum(rp_dist_mat_mse,dim=(1, 2))

        del rp_dist_mat_mse, rr_dist_mat_mse

        rr_dist_mat_loss /= (n_atoms_r**2 - num_atoms_r.square().sum(-1))
        rr_dist_mat_loss *= self._exp_conf.rr_dist_mat_loss_weight
        rr_dist_mat_loss *= batch['t'] <= self._exp_conf.rr_dist_mat_loss_t_filter
        
        rp_dist_mat_loss /= (n_atoms_r*n_atoms_p)
        rp_dist_mat_loss *= self._exp_conf.rp_dist_mat_loss_weight
        rp_dist_mat_loss *= batch['t'] <= self._exp_conf.rp_dist_mat_loss_t_filter

        clash_dist_rr = torch.maximum(2.0 - pred_pair_dists_rr, torch.tensor(0))
        eu.zero_diagonal_blocks(clash_dist_rr, num_atoms_r.long())
        clash_loss_rr = clash_dist_rr.sum((-1, -2))
        clash_loss_rr *= self._exp_conf.clash_loss_weight
        clash_loss_rr *= batch['t'] <= self._exp_conf.clash_loss_t_filter

        clash_dist_rp = torch.maximum(2.0 - pred_pair_dists_rp, torch.tensor(0))
        clash_loss_rp = clash_dist_rp.sum((-1, -2))
        clash_loss_rp *= self._exp_conf.clash_loss_weight
        clash_loss_rp *= batch['t'] <= self._exp_conf.clash_loss_t_filter

        if torch.isnan(rr_dist_mat_loss.sum()):
                print((n_atoms_r**2 - num_atoms_r.square().sum(-1)))
                raise Exception(f'NaN encountered rr')
        if torch.isinf(rr_dist_mat_loss.sum()):
                print((n_atoms_r**2 - num_atoms_r.square().sum(-1)))
                raise Exception(f'inf encountered rr')
        if torch.isnan(rp_dist_mat_loss.sum()):
                raise Exception(f'NaN encountered rp')
        if torch.isinf(rp_dist_mat_loss.sum()):
                raise Exception(f'inf encountered rp')
        
        final_loss = rr_dist_mat_loss + rp_dist_mat_loss + clash_loss_rr + clash_loss_rp
        
        def normalize_loss(x):
            return x.sum() /  (batch_size)
        

        '''aux_data = {
            'batch_train_loss': final_loss,
            'batch_rr_dist_mat_loss': rr_dist_mat_loss,
            'batch_rp_dist_mat_loss': rp_dist_mat_loss,
            'total_loss': normalize_loss(final_loss),
            'rr_dist_mat_loss': normalize_loss(rr_dist_mat_loss),
            'rp_dist_mat_loss': normalize_loss(rp_dist_mat_loss),
            'examples_per_step': torch.tensor(batch_size)
        }'''

        aux_data = {
            'batch_train_loss': final_loss,
            'batch_rr_dist_mat_loss': rr_dist_mat_loss,
            'batch_rp_dist_mat_loss': rp_dist_mat_loss,
            'total_loss': normalize_loss(final_loss),
            'rr_dist_mat_loss': normalize_loss(rr_dist_mat_loss),
            'rp_dist_mat_loss': normalize_loss(rp_dist_mat_loss),
            'rr_clash_loss': normalize_loss(clash_loss_rr),
            'rp_clash_loss': normalize_loss(clash_loss_rp),
            'examples_per_step': torch.tensor(batch_size)
        }
       
        assert final_loss.shape == (batch_size,)
        
        return normalize_loss(final_loss), aux_data
    


    def calculate_rmsd(self, pred_T_r, gt_T_r, rtype_r):

        pred_atoms_pos = all_atom.compute_backbone_r(pred_T_r, rtype_r)
        gt_atoms_pos = all_atom.compute_backbone_r(gt_T_r, rtype_r)
        gt_atoms_pos = gt_atoms_pos.to(pred_atoms_pos.device)
        
        rmsd = (pred_atoms_pos - gt_atoms_pos).square().sum(-1).sqrt().mean()

        return rmsd
    

    def eval_fn(self, eval_dir, valid_loader, min_t=None, num_t=None, noise_scale=0.0):
        
         rmsd = 0.0
         for valid_feats, pdb_name in valid_loader:

            pdb_name = pdb_name[0]
            urllib.request.urlretrieve(f'https://bioinformatics.lt/ppi3d/download/interface_coordinates/{pdb_name}.pdb', filename=eval_dir+f'/pdb_files/{pdb_name}.pdb')
            protein_pdb = open(eval_dir+f'/pdb_files/{pdb_name}.pdb', 'r').readlines()
            protein_pdb = [line.strip() for line in protein_pdb if line[20:22].strip() == 'A']
        
            valid_feats = tree.map_structure(
                lambda x: x.to(self.device), valid_feats)

            # Run inference
            sample_output = self.inference_fn(
                valid_feats,
                num_t=num_t,
                min_t=min_t,
                noise_scale=noise_scale,
            )

            rmsd+=self.calculate_RMSD(sample_output['T_r_traj'][0], valid_feats['T_r_0'][0].unsqueeze(0), valid_feats['rtype_r'].unsqueeze(0))


            sample_path = os.path.join(eval_dir, 'sample')
            T_r_traj = du.extract_true_positions(sample_output['T_r_traj'], valid_feats['com'])
            sample_path = rnp.write_rnp_to_pdb(
                rna_pos=T_r_traj,
                file_path=sample_path,
                restype=du.move_to_np(valid_feats['rtype_r'].squeeze(0)),
                protein=protein_pdb
            )
            self._log.info(f'Done sample {pdb_name}: {sample_path}')

            rmsd += self.calculate_rmsd(sample_output['T_r_traj'][0].to_tensor_7(), valid_feats['T_r_0'], valid_feats['rtype_r'])
              
         return rmsd.cpu().numpy()/len(valid_loader)
    

    def _set_t_feats(self, feats, t, t_placeholder):
        feats['t'] = t * t_placeholder
        rot_score_scaling, trans_score_scaling = self.diffuser.score_scaling(t)
        feats['rot_score_scaling'] = rot_score_scaling * t_placeholder
        feats['trans_score_scaling'] = trans_score_scaling * t_placeholder
        return feats


    def inference_fn(
            self,
            data_init,
            num_t=None,
            min_t=None,
            noise_scale=1.0,
        ):

        # Run reverse process.
        sample_feats = copy.deepcopy(data_init)
        device = sample_feats['T_r_t'].device
        if sample_feats['T_r_t'].ndim == 2:
            t_placeholder = torch.ones((1,)).to(device)
        else:
            t_placeholder = torch.ones((sample_feats['T_r_t'].shape[0],)).to(device)
        
        if num_t is None:
            num_t = self._data_conf.num_t
        if min_t is None:
            min_t = self._data_conf.min_t

        reverse_steps = torch.linspace(1.0, min_t, num_t)
        dt = torch.tensor(1/num_t)

        all_rigids = [copy.deepcopy(ru.Rigid.from_tensor_7(sample_feats['T_r_t']))]
        
        with torch.no_grad():
            for t in reverse_steps:
                sample_feats = self._set_t_feats(sample_feats, t, t_placeholder)     #Not needed wtf???
                model_out = self.model(sample_feats)
                rot_score = model_out['rot_score']
                trans_score = model_out['trans_score']
                
                T_r_prev_t = self.diffuser.reverse(
                    rigid_t=ru.Rigid.from_tensor_7(sample_feats['T_r_t']),
                    rot_score=rot_score,
                    trans_score=trans_score,
                    t=t,
                    dt=dt,
                    noise_scale=noise_scale
                )
                

                sample_feats['T_r_t'] = T_r_prev_t.to_tensor_7()    # NO POINT IN JUMPING BETWEEN TYPES
                all_rigids.append(copy.deepcopy(T_r_prev_t))
        
        ret = {'T_r_0': all_rigids[::-1][0]}
        #ret = {'T_r_traj': ru.Rigid.from_tensor_7(model_out['T_r_0'])}
        
        return ret


@hydra.main(version_base=None, config_path="../config", config_name="base")
def run(conf: DictConfig) -> None:

    exp = Experiment(conf=conf)
    exp.start_training()
    


if __name__ == '__main__':
    run()
