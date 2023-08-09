from __future__ import annotations
from typing import *

import numpy as np
import os
import torch
import torch.utils.data
from gym import Env, spaces

from ..base import register_offline_env, EpisodeData
from pathlib import Path

class Tank_reach_goal(Env):

    def get_position_coordinates(self, position):
        x = int(self.observation_boundary[0] * position[0])
        y = int(self.observation_boundary[1] * position[1])

        return np.array([x,y], dtype=np.int32)
    
    def __init__(self, goal = (0.5, 0.5), init_position= (0.3,0.3), size = 60, steering_direction_subdivision = np.pi/12, velocity = 0.05):
        super(Tank_reach_goal, self).__init__()
        
        self.size = size
        self.steering_direction_subdivision = steering_direction_subdivision
        self.velocity_step = velocity*size

        self.observation_boundary = (self.size, self.size)
        self.observation_space = spaces.Box(low = np.zeros(2), 
                                            high = np.ones(2)*self.size,
                                            dtype = np.int32)
            
        self.action_space = spaces.Discrete(3,)

        self.position = self.get_position_coordinates(init_position)
        self.goal = self.get_position_coordinates(goal)

        self.action_ditct = {'front':0, 'left':1, 'right':2}

        self.steering_direction = np.zeros(1)

    def reset(self,init_position= (0.1,0.2)):
        self.position = self.get_position_coordinates(init_position)
        self.steering_direction = np.zeros(1)

        return self.position
    
    def go_front(self):
        y_to_go = np.sin(self.steering_direction)
        x_to_go = np.cos(self.steering_direction)

        translation_direction = np.concatenate([x_to_go, y_to_go])
        translation_to_go = translation_direction*self.velocity_step
        translation_to_go = translation_to_go.astype(int)

        self.position = translation_to_go + self.position

        self.position = np.clip(self.position, a_min=0, a_max=self.size)

    def action_to_take(self, action):

        if action==self.action_ditct['front']:
            self.go_front()

        elif action == self.action_ditct['left']:
            self.steering_direction -= self.steering_direction_subdivision

        elif action == self.action_ditct['right']:
            self.steering_direction += self.steering_direction_subdivision
    
    def step(self,action):

        self.action_to_take(action)

        distance_to_goal = np.linalg.norm(self.position-self.goal)
        reward = -1

        done = False
        if distance_to_goal <= 3:
            done = True

        return self.position, reward, done, {}

def create__tank_reach_goal_env():
    return Tank_reach_goal()

def generator_load_episodes_custom_dataset(folder_name='trajectories_custom'):
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

for name in ['custom-grid-tank-goal-v1']:
    register_offline_env(
        'd4rl', name,
        create_env_fn=create__tank_reach_goal_env,
        load_episodes_fn=generator_load_episodes_custom_dataset,
    )