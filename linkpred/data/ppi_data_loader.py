import torch

import os
import numpy as np
import torch
import pandas as pd
import logging
import random

from torch.utils import data
from linkpred.data import utils as du
from openfold.utils import rigid_utils


def create_split(conf):

    if conf.experiment.warm_start:
        ckpt_dir = conf.experiment.warm_start
        ckpt_files = [x for x in os.listdir(ckpt_dir) if '.csv' in x]

        if len(ckpt_files) != 1:
            raise ValueError(f'Ambiguous test set in {ckpt_dir}')
        
        ckpt_name = ckpt_files[0]
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        test_csv = pd.read_csv(ckpt_path).drop(columns = ['Unnamed: 0'])

        pdb_csv = pd.read_csv(conf.data.csv_path)
        train_csv = pd.merge(pdb_csv, test_csv, how='outer', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1).drop_duplicates().reset_index()
        
    else:
        pdb_csv = pd.read_csv(conf.data.csv_path)
        train_idx = []
        test_idx = []
        clusters = list(set(pdb_csv['cluster'].values))
        random.shuffle(clusters)

        test_size = (1.0 - conf.data.split)*len(pdb_csv)
        for cluster in clusters:
            test_idx += list(pdb_csv[pdb_csv['cluster'] == cluster].index)
            if len(test_idx) >= test_size: break

        train_idx = list(set(pdb_csv.index).difference(test_idx))

        train_csv = pdb_csv.loc[train_idx]
        test_csv = pdb_csv.loc[test_idx]

        train_csv = train_csv.reset_index(drop=True)
        test_csv = test_csv.reset_index(drop=True)


    _log = logging.getLogger(__name__)
    _log.info(
        f'Training: {len(train_csv)} examples')
    _log.info(
        f'Testing: {len(test_csv)} examples')
    
    return train_csv, test_csv


class PPIDataset(data.Dataset):
    def __init__(
            self,
            *,
            csv,
            data_conf,
            is_training,
            is_validating
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._is_validating = is_validating
        self._data_conf = data_conf
        self.csv = csv

    @property
    def is_training(self):
        return self._is_training
    
    @property
    def is_validating(self):
        return self._is_validating

    @property
    def data_conf(self):
        return self._data_conf



    # cache make the same sample in same batch 
    #@fn.lru_cache(maxsize=100)
    def _process_csv_row(self, B_file_path):
    
        featsB = pd.read_csv(B_file_path)
        
        B_res_idx = featsB['id'].values
        
        ridx_r = B_res_idx - np.min(B_res_idx)

        rtype_r = du.encode_res(featsB['residue'].values, ribose=False)
        if self._is_validating:
            Bb_linears = featsB.iloc[:,2:14].values
        else:
            Bb_linears = featsB.iloc[:,3:15].values

        com = np.sum(Bb_linears, axis=0)[:3]/len(Bb_linears)
        Bb_linears[:,:3] -= com
                         
        Bb_frames = du.linear_to_4x4(Bb_linears)

        T_r = torch.tensor(Bb_frames).float()
        T_r = rigid_utils.Rigid.from_tensor_4x4(T_r.unsqueeze(1))[:, 0]

        angles = torch.tensor(featsB.iloc[:,15:27].values).float().reshape(-1,6,2)
        if self._is_validating:
                final_feats = {
                'rtype_r': torch.tensor(rtype_r).to(torch.int),
                'ridx_r': torch.tensor(ridx_r).to(torch.int16),
                'T_r': T_r,
                'com': com
            }
        else:
            final_feats = {
                'rtype_r': torch.tensor(rtype_r).to(torch.int),
                'ridx_r': torch.tensor(ridx_r).to(torch.int16),
                'angles': angles,
                'T_r': T_r,
                'com': com
            }

        return final_feats


    def __len__(self):
        return len(self.csv)

    def __getitem__(self, example_idx):

        # Sample data example.
        csv_row = self.csv.iloc[example_idx]
        pdb_name = csv_row['pdb_name']
        
        B_file_path = csv_row['B_path']
        complex_feats = self._process_csv_row(B_file_path)

        complex_feats['sc_r'] = torch.zeros_like(complex_feats['T_r'].get_trans())
        
        # Convert all features to tensors.
        complex_feats['T_r'] = complex_feats['T_r'].to_tensor_7()
        
        if self.is_validating:
            return complex_feats, pdb_name
        else:
            return complex_feats


class TrainSampler(data.Sampler):

    def __init__(
            self,
            *,
            data_conf,
            dataset,
            batch_size,
            sample_mode,
        ):
        
        self._data_conf = data_conf
        self._dataset = dataset
        self._data_csv = self._dataset.csv
        self._dataset_indices = list(range(len(self._data_csv)))
        self._data_csv['index'] = self._dataset_indices
        self._batch_size = batch_size
        self.epoch = 0
        self._sample_mode = sample_mode
        self.sampler_len = len(self._dataset_indices)
        
    
    def __iter__(self):

        # Each batch contains multiple proteins of the same length.
        grouped = self._data_csv.groupby('lenB')
        
        batches_indices = []

        # Iterate over groups and create batches of row indices
        for name, group in grouped:
            # Repeat row indices if necessary to form complete batches
            repeated_indices = np.tile(group.index, (self._batch_size + len(group) - 1) // len(group))
            
            last_batch_size = len(repeated_indices) % self._batch_size

            if last_batch_size < self._batch_size:
                remaining_indices = np.random.choice(repeated_indices, self._batch_size - last_batch_size)
                repeated_indices = np.concatenate([repeated_indices, remaining_indices])

            # Shuffle the repeated indices
            np.random.shuffle(repeated_indices)
            
            # Split the shuffled indices into batches of size batch_size
            batches_indices.extend(np.array_split(repeated_indices, len(repeated_indices) // self._batch_size))

        np.random.shuffle(batches_indices)
        
        return iter(np.concatenate(batches_indices).tolist())
  

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.sampler_len
    

class TestSampler(data.Sampler):

    def __init__(
            self,
            *,
            data_conf,
            dataset,
            batch_size,
            sample_mode,
        ):
        
        self._data_conf = data_conf
        self._dataset = dataset
        self._data_csv = self._dataset.csv
        self._dataset_indices = list(range(len(self._data_csv)))
        self._data_csv['index'] = self._dataset_indices
        self._batch_size = batch_size
        self.epoch = 0
        self._sample_mode = sample_mode
        self.sampler_len = len(self._dataset_indices)

        
    
    def __iter__(self):
        
        # Each batch contains multiple proteins of the same length.
        grouped = self._data_csv.groupby('lenB')
        
        batches_indices = []

        # Iterate over groups and create batches of row indices
        for name, group in grouped:
            # Repeat row indices if necessary to form complete batches
            repeated_indices = np.tile(group.index, (self._batch_size + len(group) - 1) // len(group))
            
            last_batch_size = len(repeated_indices) % self._batch_size

            if last_batch_size < self._batch_size:
                remaining_indices = np.random.choice(repeated_indices, self._batch_size - last_batch_size)
                repeated_indices = np.concatenate([repeated_indices, remaining_indices])

            # Shuffle the repeated indices
            np.random.shuffle(repeated_indices)
            
            # Split the shuffled indices into batches of size batch_size
            batches_indices.extend(np.array_split(repeated_indices, len(repeated_indices) // self._batch_size))

        np.random.shuffle(batches_indices)

        return iter(np.concatenate(batches_indices).tolist())
        
        

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.sampler_len
    



class ValidSampler(data.Sampler):

    def __init__(
            self,
            *,
            data_conf,
            dataset,
            nof_samples
        ):
        
        self._data_conf = data_conf
        self._dataset = dataset
        self._data_csv = self._dataset.csv
        self._dataset_indices = list(range(len(self._data_csv)))
        self._data_csv['index'] = self._dataset_indices
        self.epoch = 0
        self._nof_samples = nof_samples
        self.sampler_len = len(self._dataset_indices)

        
    
    def __iter__(self):
        
        if self._nof_samples == 'all':
            indices = self._dataset_indices
        else:
            random.shuffle(self._dataset_indices)
            if self._nof_samples > len(self._dataset_indices):
                indices = random.choices(self._dataset_indices, self._nof_samples)
            else:
                indices = random.sample(self._dataset_indices, self._nof_samples)
        return iter(indices)
        
        

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.sampler_len