from __future__ import annotations
from typing import *

import functools

import numpy as np
import os
import torch
import torch.utils.data
from gym import Env, spaces

from ..base import register_offline_env, EpisodeData
from pathlib import Path


class Tank_reach_goal(Env):

    def get_observation(self):
        angle = self.steering_direction * 2 * np.pi / self.nangles
        observation = np.concatenate([
            self.position / (self.size - 1) * 2 - 1,
            np.sin(angle), np.cos(angle)])   #, self.goal])
        if self.goal_obs_indic:
            observation = np.concatenate([observation, -np.ones([1])])
        return observation

    def randomize_goal(self):
        while True:
            self.goal = np.random.randint(self.size, size=[2]).astype(np.float32)
            distance_to_goal = self.distance_function(self.position, self.goal)
            if not (distance_to_goal <= self.epsolon_distance_goal) or True:
                # print(self.goal)
                return

    def get_goal_observation(self):
        if not self.normalized_goal_obs:
            # old behavior w/o normalizing to [-1,1]
            observation = np.concatenate([self.goal, np.zeros_like(self.goal)])
        else:
            observation = np.concatenate([self.goal / (self.size - 1) * 2 - 1, np.zeros_like(self.goal)])
        if self.goal_obs_indic:
            observation = np.concatenate([observation, np.ones([1])])
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
        self.position = np.array([int(self.size / 5), int(self.size / 5)], dtype=np.float32)
        if self.rand_goal:
            self.randomize_goal()
        else:
            self.goal = np.array([ int(self.size * 4 / 5), int(self.size * 4 / 5) ], dtype=np.float32)

    def set_random_state(self):
        self.position = np.random.uniform(0.,2*np.pi*0.2, 2).astype(np.float32)
        self.steering_direction = np.random.uniform(0,2*np.pi,1).astype(np.float32)

    def __init__(self, size=20, nangles=16, normalized_goal_obs=False, goal_obs_indic=False,
                 rand_s0=False, rand_goal=False):
        super(Tank_reach_goal, self).__init__()

        self.size = size
        self.nangles = nangles
        self.epsolon_distance_goal = 1
        self.normalized_goal_obs = normalized_goal_obs
        self.goal_obs_indic = goal_obs_indic
        self.rand_goal = rand_goal

        self.observation_space = spaces.Box(low = -1,
                                            high = 1,
                                            shape=(4 + int(goal_obs_indic),),
                                            dtype = np.float32)

        self.initialize_init_pos_and_goal()

        self.action_space = spaces.Discrete(3,)
        self.action_ditct = {'front':0, 'left':1, 'right':2}

    def reset(self):
        self.initialize_init_pos_and_goal()
        observation = self.get_observation()
        return observation

    def go_front(self):
        angle = self.steering_direction * 2 * np.pi / self.nangles
        y_to_go = np.sin(angle)
        x_to_go = np.cos(angle)
        translation = np.concatenate([x_to_go, y_to_go])
        self.position = translation + self.position
        self.position = np.clip(self.position, a_min=0, a_max=self.size - 1)

    def action_to_take(self, action):

        if action==self.action_ditct['front']:
            self.go_front()

        elif action == self.action_ditct['left']:
            self.steering_direction -= 1

        elif action == self.action_ditct['right']:
            self.steering_direction += 1

    def step(self,action):
        reward = -1

        distance_to_goal = self.distance_function(self.position, self.goal)
        if distance_to_goal <= self.epsolon_distance_goal:
            done = True
            # observation = np.concatenate([self.goal, np.zeros_like(self.goal)])
            observation = self.get_goal_observation()
            return observation, reward, done, {}

        self.action_to_take(action)
        done = False
        observation = self.get_observation()

        return observation, reward, done, {}


def create__tank_reach_goal_env():
    return Tank_reach_goal()

def generator_load_episodes_custom_dataset(folder_name='trajectories_ez_custom'):
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


register_offline_env(
    'd4rl', 'custom-grid-tank-goal-tz-v1',
    create_env_fn=Tank_reach_goal,
    load_episodes_fn=generator_load_episodes_custom_dataset,
)


register_offline_env(
    'd4rl', 'custom-grid-tank-goal-tz-normG-v1',
    create_env_fn=functools.partial(Tank_reach_goal, normalized_goal_obs=True),
    load_episodes_fn=functools.partial(generator_load_episodes_custom_dataset, 'trajectories_tz_normG'),
)

# THIS IS WHAT I USED IN THE END
register_offline_env(
    'd4rl', 'custom-grid-tank-goal-tz-normG-randG-v1',
    create_env_fn=functools.partial(Tank_reach_goal, normalized_goal_obs=True, rand_goal=True),
    load_episodes_fn=functools.partial(generator_load_episodes_custom_dataset, 'trajectories_tz_normG_randG'),
)

register_offline_env(
    'd4rl', 'custom-grid-tank-goal-tz-normG-indic-v1',
    create_env_fn=functools.partial(Tank_reach_goal, goal_obs_indic=True, normalized_goal_obs=True),
    load_episodes_fn=functools.partial(generator_load_episodes_custom_dataset, 'trajectories_tz_normG_indic'),
)


register_offline_env(
    'd4rl', 'custom-grid-tank-goal-tz-normG-indic-randG-v1',
    create_env_fn=functools.partial(Tank_reach_goal, goal_obs_indic=True, normalized_goal_obs=True, rand_goal=True),
    load_episodes_fn=functools.partial(generator_load_episodes_custom_dataset, 'trajectories_tz_normG_indic_randG'),
)

