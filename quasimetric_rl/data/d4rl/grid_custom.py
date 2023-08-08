from __future__ import annotations
from typing import *

import numpy as np
import os
import torch
import torch.utils.data
from gym import Env, spaces

from ..base import register_offline_env, EpisodeData
import random
from pathlib import Path

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
        reward = -distance_to_goal

        done = False
        if distance_to_goal <= 3:
            done = True

        return self.position, reward, done, {}

def create_maze_simple_env():
    return Maze_simple()

def generator_load_episodes_custom_dataset(folder_name='trajectories'):
    _, _, files = next(os.walk(folder_name))
    size = len(files)
    
    folder_trajectories_name = Path(folder_name)

    for idx in range(size):
        test_name = Path(f'test_{idx:04}.npz')
        path_to_pick_episode = folder_trajectories_name / test_name

        dict_episode = np.load(path_to_pick_episode)

        episode_dict = dict(
            episode_lengths=torch.as_tensor([len(dict_episode['all_observations']) - 1], dtype=torch.int64),
            all_observations=torch.as_tensor(dict_episode['all_observations'], dtype=torch.float32),
            actions=torch.as_tensor(dict_episode['actions'], dtype=torch.int64),
            rewards=torch.as_tensor(dict_episode['rewards'], dtype=torch.float32),
            terminals=torch.as_tensor(dict_episode['terminals'], dtype=torch.bool),
            timeouts=(
                torch.as_tensor(dict_episode['timeouts'], dtype=torch.bool) if 'timeouts' in dict_episode else
                torch.zeros(dict_episode['terminals'].shape, dtype=torch.bool)
            )
        )

        episode_data = EpisodeData(**episode_dict)
        yield episode_data

for name in ['custom-grid-umaze-v1']:
    register_offline_env(
        'd4rl', name,
        create_env_fn=create_maze_simple_env,
        load_episodes_fn=generator_load_episodes_custom_dataset,
    )