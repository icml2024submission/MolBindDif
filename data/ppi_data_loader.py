import math
from typing import Optional

import torch
import torch.distributed as dist

import os
import tree
import numpy as np
import torch
import pandas as pd
import logging
import random
import functools as fn

from torch.utils import data
from data import utils as du
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
            diffuser,
            is_training,
            is_validating
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._is_validating = is_validating
        self._data_conf = data_conf
        self.csv = csv
        self._diffuser = diffuser

    @property
    def is_training(self):
        return self._is_training
    
    @property
    def is_validating(self):
        return self._is_validating

    @property
    def diffuser(self):
        return self._diffuser

    @property
    def data_conf(self):
        return self._data_conf



    # cache make the same sample in same batch 
    #@fn.lru_cache(maxsize=100)
    def _process_csv_row(self, A_file_path, B_file_path):
      
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

        center_around_bind = False
        if center_around_bind:
            com = np.sum(A_linears*A_res_bind[:, np.newaxis], axis=0)[:3]/np.sum(A_res_bind)
        else:
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
        return final_feats


    def __len__(self):
        return len(self.csv)

    def __getitem__(self, example_idx):

        # Sample data example.
        csv_row = self.csv.iloc[example_idx]
        pdb_name = csv_row['pdb_name']
        
        A_file_path = csv_row['A_path']
        B_file_path = csv_row['B_path']
        complex_feats = self._process_csv_row(A_file_path, B_file_path)
       
        rng = np.random.default_rng(None)
        complex_feats['sc_r'] = torch.zeros_like(complex_feats['T_r_0'].get_trans())
        complex_feats['sc_p'] = torch.clone(complex_feats['T_p_0'].get_trans())

        # Sample t and diffuse.
        if not self.is_validating:
            t = torch.tensor(rng.uniform(self._data_conf.min_t, 1.0))
            diff_feats_t = self._diffuser.forward_marginal(rigids_0=complex_feats['T_r_0'], t=t)
        else:
            t = torch.tensor(1.0)
            diff_feats_t = self.diffuser.sample_ref(n_samples=complex_feats['T_r_0'].shape[0])
        
        
        complex_feats.update(diff_feats_t)
        complex_feats['t'] = t

        # Convert all features to tensors.
        '''final_feats = tree.map_structure(lambda x: x if torch.is_tensor(x) else torch.tensor(x), complex_feats)'''
        complex_feats['T_r_0'] = complex_feats['T_r_0'].to_tensor_7()
        complex_feats['T_r_t'] = complex_feats['T_r_t'].to_tensor_7()
        complex_feats['T_p_0'] = complex_feats['T_p_0'].to_tensor_7()
        
        

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
        self.sampler_len = len(self._dataset_indices) * self._batch_size

        
    
    def __iter__(self):
        if self._sample_mode == 'time_batch':
            # Each batch contains multiple time steps of the same protein.
            random.shuffle(self._dataset_indices)
            repeated_indices = np.repeat(self._dataset_indices, self._batch_size)
            return iter(repeated_indices)
        
        elif self._sample_mode == 'cluster_time_batch':
            # Each batch contains multiple time steps of a protein from a cluster.
            sampled_clusters = self._data_csv.groupby('cluster').sample(
                1, random_state=self.epoch)
            dataset_indices = sampled_clusters.index.tolist()
            repeated_indices = np.repeat(dataset_indices, self._batch_size)
            return iter(repeated_indices.tolist())
        else:
            raise ValueError(f'Invalid sample mode: {self._sample_mode}')

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
        self.sampler_len = len(self._dataset_indices) * self._batch_size

        
    
    def __iter__(self):
        # Each batch contains multiple time steps of the same protein.
        random.shuffle(self._dataset_indices)
        repeated_indices = np.repeat(self._dataset_indices, self._batch_size)
        return iter(repeated_indices)
        
        

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
            nof_samples,
        ):
        
        self._data_conf = data_conf
        self._dataset = dataset
        self._data_csv = self._dataset.csv
        self._dataset_indices = list(range(len(self._data_csv)))
        self._data_csv['index'] = self._dataset_indices
        self._nof_samples = nof_samples
        self.epoch = 0
        self.sampler_len = nof_samples

        
    
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
    


    # modified from torch.utils.data.distributed.DistributedSampler
# key points: shuffle of each __iter__ is determined by epoch num to ensure the same shuffle result for each proccessor
class DistributedTrainSampler(data.Sampler):
    r"""Sampler that restricts data loading to a subset of the dataset.

    modified from torch.utils.data.distributed import DistributedSampler

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, 
                *,
                data_conf,
                dataset,
                batch_size,
                num_replicas: Optional[int] = None,
                rank: Optional[int] = None, shuffle: bool = True,
                seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self._data_conf = data_conf
        self._dataset = dataset
        self._data_csv = self._dataset.csv
        self._dataset_indices = list(range(len(self._data_csv)))
        self._data_csv['index'] = self._dataset_indices
        # _repeated_size is the size of the dataset multiply by batch size
        self._repeated_size = batch_size * len(self._data_csv)
        self._batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and self._repeated_size % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (self._repeated_size - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(self._repeated_size / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed


    def __iter__(self) :
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            self._dataset_indices
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self._data_csv), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self._data_csv)))  # type: ignore[arg-type]

        # indices is expanded by self._batch_size times
        indices = np.repeat(indices, self._batch_size)
        
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices = np.concatenate((indices, indices[:padding_size]), axis=0)
            else:
                indices = np.concatenate((indices, np.repeat(indices, math.ceil(padding_size / len(indices)))[:padding_size]), axis=0)

        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        # 
        assert len(indices) == self.num_samples

        return iter(indices)
    

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch



class DistributedTestSampler(data.Sampler):

    def __init__(self, 
                *,
                data_conf,
                dataset,
                batch_size,
                num_replicas: Optional[int] = None,
                rank: Optional[int] = None, shuffle: bool = True,
                seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self._data_conf = data_conf
        self._dataset = dataset
        self._data_csv = self._dataset.csv
        self._dataset_indices = list(range(len(self._data_csv)))
        self._data_csv['index'] = self._dataset_indices
        # _repeated_size is the size of the dataset multiply by batch size
        self._repeated_size = batch_size * len(self._data_csv)
        self._batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and self._repeated_size % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (self._repeated_size - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(self._repeated_size / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed


    def __iter__(self) :
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            self._dataset_indices
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self._data_csv), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self._data_csv)))  # type: ignore[arg-type]

        # indices is expanded by self._batch_size times
        indices = np.repeat(indices, self._batch_size)
        
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices = np.concatenate((indices, indices[:padding_size]), axis=0)
            else:
                indices = np.concatenate((indices, np.repeat(indices, math.ceil(padding_size / len(indices)))[:padding_size]), axis=0)

        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        # 
        assert len(indices) == self.num_samples

        return iter(indices)
    

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch




class DistributedValidSampler(data.Sampler):

    def __init__(self, 
                *,
                data_conf,
                dataset,
                nof_samples,
                num_replicas: Optional[int] = None,
                rank: Optional[int] = None, shuffle: bool = True,
                seed: int = 0, drop_last: bool = False) -> None:
        
        
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self._data_conf = data_conf
        self._dataset = dataset
        self._data_csv = self._dataset.csv
        self._dataset_indices = list(range(len(self._data_csv)))
        self._data_csv['index'] = self._dataset_indices
        if nof_samples < len(self._data_csv):
            self._nof_samples = nof_samples
        else:
            self._nof_samples = len(self._data_csv)
        
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and self._nof_samples % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (self._nof_samples - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(self._nof_samples / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed



    def __iter__(self) :
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            self._dataset_indices
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self._data_csv), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self._data_csv)))  # type: ignore[arg-type]

        # indices is expanded by self._batch_size times
        indices = indices[:self._nof_samples]
        
        
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices = np.concatenate((indices, indices[:padding_size]), axis=0)
            else:
                indices = np.concatenate((indices, np.repeat(indices, math.ceil(padding_size / len(indices)))[:padding_size]), axis=0)

        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        # 
        assert len(indices) == self.num_samples

        return iter(indices)

    

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch



