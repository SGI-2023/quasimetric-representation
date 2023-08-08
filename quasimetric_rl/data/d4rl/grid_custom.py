from __future__ import annotations
from typing import *

import logging
import functools

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from gym import Env, spaces
from torch.utils.data import Dataset

import d4rl.pointmaze

from ..base import register_offline_env, EpisodeData
from . import load_environment, convert_dict_to_EpisodeData_iter, sequence_dataset
import random
import os
from pathlib import Path
import collections


random.seed(0)
np.random.seed(0)

class Maze_simple(Env):

    def initialize_to_a_position(self):
        x = random.randrange(int(self.observation_boundary[0] * 0.05), int(self.observation_boundary[0] * 0.10))
        y = random.randrange(int(self.observation_boundary[1] * 0.15), int(self.observation_boundary[1] * 0.20))

        return np.array([x,y], dtype=np.int32)
    
    def get_position_coordinates(self,pos_normalized):
        x = int(self.observation_boundary[0] * pos_normalized[0])
        y = int(self.observation_boundary[1] * pos_normalized[1])

        return np.array([x,y],dtype=np.int32)
    
    
    def __init__(self, goal = (0.5, 0.5), size = 60):
        super(Maze_simple, self).__init__()
        
        self.size = size
        self.observation_boundary = (self.size, self.size)
        self.observation_space = spaces.Box(low = np.zeros(2), 
                                            high = np.ones(2)*self.size,
                                            dtype = np.int32)
    
        
        # Define an action space ranging from 0 to 4
        self.action_space = spaces.Discrete(4,)

        self.position = self.initialize_to_a_position()

        self.action_ditct = {'up':0, 'down':1, 'left':2, 'right':3}

        self.goal = self.get_position_coordinates(goal)
        

    def reset(self, seed=0):
        self.position = self.initialize_to_a_position()

        return self.position

    def position_alter_depending_on_coordinate(self, action):
        if action==0:
            return np.array([5,0],dtype=np.int32)
        elif action==1:
            return np.array([-5,0],dtype=np.int32)
        elif action==2:
            return np.array([0,-5],dtype=np.int32)
        else:
            return np.array([0,5],dtype=np.int32)
    
    def step(self,action):

        translation_to_do = self.position_alter_depending_on_coordinate(action)

        self.position = (self.position + translation_to_do)
        self.position = self.position.clip(min=0, max=self.size)

        distance_to_goal = np.linalg.norm(self.position-self.goal)
        print(distance_to_goal)
        print()
        reward = -distance_to_goal

        done = False
        if distance_to_goal <= 3:
            done = True

        return self.position, reward, done, {}
    

class TrajectoriesDataset(Dataset):
    def __init__(self, trajectories_folder ='/home/danperazzo/Desktop/SGI_projects/quasimetric-rl/offline/trajectories/' ):

        self.trajectories_folder = Path(trajectories_folder)
        self.size = 2

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        if idx > self.size:
            return StopIteration
        
        img_name = Path(f'test_{idx:04}.npz')

        path_to_pick_image = self.trajectories_folder / img_name


        dict = np.load(path_to_pick_image)
        
        return dict
    
class Trajectories_Iterator(collections.Iterator):
    def __init__(self, trajectories_folder ='/home/danperazzo/Desktop/SGI_projects/quasimetric-rl/offline/trajectories/' ):
        self.trajectories_folder = Path(trajectories_folder)
        self.size = 400
        self.counter = 0

    def __next__(self):
        if self.counter > self.size:
            raise StopIteration
        
        img_name = Path(f'test_{self.counter:04}.npz')
        path_to_pick_image = self.trajectories_folder / img_name

        dict = np.load(path_to_pick_image)
        
        self.counter = self.counter + 1
        
        return dict 

def create_maze_simple_env():
    return Maze_simple()


def convert_dict_to_EpisodeData_iter_discrete(sequence_dataset_episodes: Iterator[Mapping[str, np.ndarray]]):
    for episode in sequence_dataset_episodes:
        episode_dict = dict(
            episode_lengths=torch.as_tensor([len(episode['all_observations']) - 1], dtype=torch.int64),
            all_observations=torch.as_tensor(episode['all_observations'], dtype=torch.float32),
            actions=torch.as_tensor(episode['actions'], dtype=torch.int64),
            rewards=torch.as_tensor(episode['rewards'], dtype=torch.float32),
            terminals=torch.as_tensor(episode['terminals'], dtype=torch.bool),
            timeouts=(
                torch.as_tensor(episode['timeouts'], dtype=torch.bool) if 'timeouts' in episode else
                torch.zeros(episode['terminals'].shape, dtype=torch.bool)
            ),
            observation_infos={},
            transition_infos={},
        )
        for k, v in episode.items():
            if k.startswith('observation_infos/'):
                episode_dict['observation_infos'][k.split('/', 1)[1]] = v
            elif k.startswith('transition_infos/'):
                episode_dict['transition_infos'][k.split('/', 1)[1]] = v
        yield EpisodeData(**episode_dict)


def load_episodes_simple_dataset():
    iterator_dataset_maze = Trajectories_Iterator()
    iterator_dataset_maze = iter(iterator_dataset_maze)
    yield from convert_dict_to_EpisodeData_iter_discrete(
        iterator_dataset_maze
    )


for name in ['custom-grid-umaze-v1']:
    register_offline_env(
        'd4rl', name,
        create_env_fn=create_maze_simple_env,
        load_episodes_fn=load_episodes_simple_dataset,
    )


