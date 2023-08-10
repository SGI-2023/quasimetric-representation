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

    def get_observation(self):
        observation = np.concatenate([self.position, self.steering_direction])   #, self.goal])
        return observation

    def distance_function(self, pos, goal):
        return np.linalg.norm(pos-goal)

    def initialize_init_pos_and_goal(self):
        '''
        self.position = np.random.uniform(0.01,2*np.pi*0.09,2)
        self.steering_direction = np.random.uniform(0,2*np.pi,1)

        candidate_goal = np.random.uniform(0.01,2*np.pi*0.09,2)
        while self.distance_function(self.position, candidate_goal)< self.epsolon_distance_goal:
            candidate_goal = np.random.uniform(0.01,2*np.pi*0.09,2)
        self.goal = candidate_goal
        '''
        self.steering_direction = np.zeros(1)
        self.position = np.array([2*np.pi*0.2, 2*np.pi*0.2 ], dtype=np.float32)
        self.goal = np.array([2*np.pi*0.8, 2*np.pi*0.8 ], dtype=np.float32)

    def set_random_state(self):
        self.position = np.random.uniform(0.,2*np.pi*0.2, 2).astype(np.float32)
        self.steering_direction = np.random.uniform(0,2*np.pi,1).astype(np.float32)

    def __init__(self, angle_velocity = np.pi/8, velocity = 0.05):
        super(Tank_reach_goal, self).__init__()

        self.size = 2*np.pi
        self.angle_velocity = angle_velocity
        self.velocity_radius = velocity
        self.velocity_step = velocity*self.size
        self.epsolon_distance_goal = self.size*self.velocity_radius*0.25

        self.observation_boundary = (self.size, self.size)
        self.observation_space = spaces.Box(low = np.zeros(3),
                                            high = np.ones(3)*self.size,
                                            dtype = np.float64)

        self.initialize_init_pos_and_goal()

        self.action_space = spaces.Discrete(3,)
        self.action_ditct = {'front':0, 'left':1, 'right':2}

    def reset(self):
        self.initialize_init_pos_and_goal()
        observation = self.get_observation()

        return observation

    def go_front(self):
        y_to_go = np.sin(self.steering_direction)
        x_to_go = np.cos(self.steering_direction)

        translation_direction = np.concatenate([x_to_go, y_to_go])
        translation_to_go = translation_direction*self.velocity_step

        self.position = translation_to_go + self.position

        self.position = np.clip(self.position, a_min=0, a_max=self.size)

    def action_to_take(self, action):

        if action==self.action_ditct['front']:
            self.go_front()

        elif action == self.action_ditct['left']:

            self.steering_direction -= self.angle_velocity
            if self.steering_direction < 0:
                self.steering_direction = np.zeros(1)

        elif action == self.action_ditct['right']:

            self.steering_direction += self.angle_velocity
            if self.steering_direction > self.size:
                self.steering_direction = np.ones(1)*self.size

    def step(self,action):

        self.action_to_take(action)

        distance_to_goal = self.distance_function(self.position, self.goal)
        reward = -1

        done = False
        if distance_to_goal <= self.epsolon_distance_goal:
            done = True

        observation = self.get_observation()

        return observation, reward, done, {}

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
