from __future__ import annotations
from typing import *

import time
import logging

import torch
import torch.utils.data

from quasimetric_rl.modules import QRLConf, QRLAgent, QRLLosses, InfoT
from quasimetric_rl.data import Dataset, BatchData, ConcatDataset
from quasimetric_rl.data.d4rl.maze2d_custom import update_env_seed
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import numpy as np

def seed_worker(_):
    worker_seed = torch.utils.data.get_worker_info().seed % (2 ** 32)
    np.random.seed(worker_seed)


class Trainer(object):
    agent: QRLAgent
    losses: QRLLosses
    device: torch.device
    dataset: Dataset
    batch_size: int
    dataloader: torch.utils.data.DataLoader


    def __init__(self, *,
                 agent_conf: QRLConf,
                 device: torch.device,
                 dataset: Dataset,
                 batch_size: int,
                 total_optim_steps: int,
                 dataloader_kwargs: Dict[str, Any] = {},
                 num_envs: int,
                 cfg_env: QRLConf,):

        self.device = device
        self.batch_size = batch_size
        self.device = device

        first_dataset = dataset

        self.agent, self.losses = agent_conf.make(
            env_spec=first_dataset.env_spec,
            total_optim_steps=total_optim_steps)
        self.agent.to(device)
        self.losses.to(device)

        logging.info('Agent:\n\t' +
                     str(self.agent).replace('\n', '\n\t') + '\n\n')
        logging.info('Losses:\n\t' +
                     str(self.losses).replace('\n', '\n\t') + '\n\n')
        
        list_of_datasets = self.create_dataset_list(first_dataset, num_envs, cfg_env)

        self.dataset_concat = ConcatDataset(list_of_datasets)

        self.dataloader = self.create_dataloader_from_concat_dataset(
            batch_size=batch_size,
            **dataloader_kwargs,
        )


    def create_dataset_list(self, first_dataset: Dataset, num_envs: int, cfg_env: QRLConf) -> List[Dataset]:
        dataset_list = [first_dataset]

        for i in range(1,num_envs):
            update_env_seed(i)
            dataset_list.append(cfg_env.make())

        return dataset_list

    @property
    def num_batches(self):
        return len(self.dataloader)

    def create_dataloader_from_concat_dataset(self, *,
                       batch_size: int, shuffle: bool = False,
                       drop_last: bool = False,
                       pin_memory: bool = False,
                       num_workers: int = 0, persistent_workers: bool = False,
                       **kwargs) -> torch.utils.data.DataLoader:
        sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(self.dataset_concat),
            batch_size=batch_size,
            drop_last=drop_last,
        )
        return torch.utils.data.DataLoader(
            self.dataset_concat,
            batch_size=None,
            sampler=sampler,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            **kwargs,
        )
    

    def iter_training_data(self) -> Iterator[Tuple[BatchData, InfoT]]:
        r"""
        Yield data to train on for each optimization iteration.

        yield (
            data,
            info,
        )
        """
        data_t0 = time.time()
        data: BatchData
  
        for data in self.dataloader: #TODO:Going one by one for each dataloader until run on all of then
            data = data.to(self.device)
            yield data, dict(data_time=time.time() - data_t0)
            data_t0 = time.time()

    def train_step(self, data: BatchData, *, optimize: bool = True) -> InfoT:
        return self.losses(self.agent, data, optimize=optimize).info
