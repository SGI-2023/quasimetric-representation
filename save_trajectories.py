import numpy as np
import random

import os

from stable_baselines.common.env_checker import check_env
from quasimetric_rl.data.d4rl.grid_custom import Maze_simple

random.seed(0)
np.random.seed(0)

maze_env = Maze_simple()

print(check_env(maze_env))

name = 'trajectories'

if not os.path.exists(name):
    os.makedirs(name)


for i in range(500):
    observation_list = []
    next_obervation_list = []
    reward_list = []
    terminal_list = []
    actions_list = []
    for j in range(1000):
        dict_data = {}
        random_action = random.randrange(4)
        observation = maze_env.position
        next_observation, reward, terminal, _ = maze_env.step(random_action)

        observation_list.append(observation)
        next_obervation_list.append(next_observation)
        reward_list.append(reward)
        terminal_list.append(terminal)
        actions_list.append(random_action)

    dict_data['observations']=np.array(observation_list)
    dict_data['next_observations'] = np.array(next_obervation_list)
    dict_data['rewards'] = np.array(reward_list)
    dict_data['terminals'] = np.array(terminal_list)
    dict_data['all_observations'] = np.concatenate(
                [dict_data['observations'], dict_data['next_observations'][-1:]], axis=0)
    dict_data['actions'] = np.array(actions_list,dtype=np.int64)


    np.savez(name+f'/test_{i:04}', **dict_data)
    print(i)

print(check_env(maze_env))