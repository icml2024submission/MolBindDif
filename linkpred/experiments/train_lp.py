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

from collections import defaultdict
from collections import deque
from datetime import datetime
from omegaconf import DictConfig
from omegaconf import OmegaConf
import torch.distributed as dist
from hydra.core.hydra_config import HydraConfig

from linkpred.data import ppi_data_loader
from linkpred.data import utils as du
from linkpred.data import all_atom
from linkpred.model import main_network
from linkpred.experiments import utils as eu



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
        
        self._model = main_network.RestoreNetwork(self._model_conf)
   
        

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
            is_training=True,
            is_validating=False
        )

        test_dataset = ppi_data_loader.PPIDataset(
            csv = test_csv,
            data_conf=self._data_conf,
            is_training=False,
            is_validating=False
        )

        valid_dataset = ppi_data_loader.PPIDataset(
            csv = test_csv,
            data_conf=self._data_conf,
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
        loss.backward()
        self._optimizer.step()
        return loss, aux_data


    def loss_fn(self, batch):

        model_out = self.model(batch)

        batch_size, num_res_r = batch['ridx_r'].shape

        pred_all_frames_to_global = all_atom.torsion_angles_to_frames(batch['T_r'], model_out['angles'], batch['rtype_r'])
        pred_atoms_pos_r = all_atom.frames_and_literature_positions_to_atom_pos(pred_all_frames_to_global, batch['rtype_r'])

        gt_all_frames_to_global = all_atom.torsion_angles_to_frames(batch['T_r'], batch['angles'], batch['rtype_r'])
        gt_atoms_pos_r = all_atom.frames_and_literature_positions_to_atom_pos(gt_all_frames_to_global, batch['rtype_r'])

        gt_flat_atoms_r = gt_atoms_pos_r.reshape([batch_size, -1, 3])
        gt_flat_atoms_dist_rr = gt_flat_atoms_r.unsqueeze(-2) - gt_flat_atoms_r.unsqueeze(-3)
        gt_pair_dists_rr = torch.linalg.norm(gt_flat_atoms_dist_rr, dim=-1).float()
        n_atoms_r = gt_flat_atoms_r.shape[1]

        pred_flat_atoms_r = pred_atoms_pos_r.reshape([batch_size, -1, 3])
        pred_flat_atoms_dist_rr = pred_flat_atoms_r.unsqueeze(-2) - pred_flat_atoms_r.unsqueeze(-3)
        pred_pair_dists_rr = torch.linalg.norm(pred_flat_atoms_dist_rr, dim=-1).float()

        del gt_flat_atoms_dist_rr, pred_flat_atoms_dist_rr, gt_flat_atoms_r, pred_flat_atoms_r
        
        rr_dist_mat_mse = (gt_pair_dists_rr - pred_pair_dists_rr)**2
        rr_dist_mat_loss = torch.sum(rr_dist_mat_mse, dim=(1, 2))

        del rr_dist_mat_mse

        rr_dist_mat_loss /= (n_atoms_r**2 - n_atoms_r) 
        rr_dist_mat_loss *= self._exp_conf.rr_dist_mat_loss_weight
      
        final_loss = rr_dist_mat_loss
        
        def normalize_loss(x):
            return x.sum() / batch_size

        aux_data = {
            'batch_train_loss': final_loss,
            'total_loss': normalize_loss(final_loss),
            'examples_per_step': torch.tensor(batch_size)
        }
       
        assert final_loss.shape == (batch_size,)
        
        return normalize_loss(final_loss), aux_data
    


    


    def inference_fn(
            self,
            data_init
        ):

        sample_feats = copy.deepcopy(data_init)
        self.model.eval()
        with torch.no_grad():
            model_out = self.model(sample_feats)

        ret = {'angles': model_out['angles']}
        
        return ret


@hydra.main(version_base=None, config_path="../config", config_name="base")
def run(conf: DictConfig) -> None:

    exp = Experiment(conf=conf)
    exp.start_training()
    


if __name__ == '__main__':
    run()
