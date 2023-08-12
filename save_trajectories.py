r'''
ENV='custom-grid-tank-goal-tz-normG-randG-v1' python save_trajectories.py
'''

import numpy as np
import random

import os

from quasimetric_rl.data import Dataset


random.seed(0)
np.random.seed(0)

env_name = os.environ.get('ENV', 'custom-grid-tank-goal-v1')
assert env_name.startswith('custom-grid-tank-goal-')
name = '_'.join(
    ['trajectories'] + list(env_name.split('-')[4:-1])
)

print(env_name, name)
# 'trajectories_ez_custom'

env = Dataset.Conf(kind='d4rl', name=env_name).make(dummy=True).create_env()

if not os.path.exists(name):
    os.makedirs(name)


for i in range(1000):
    env.reset()
    observation_list = []
    next_obervation_list = []
    reward_list = []
    terminal_list = []
    actions_list = []
    for j in range(1000):
        dict_data = {}
        random_action = random.randrange(len(env.action_ditct))
        observation = env.get_observation()
        next_observation, reward, terminal, _ = env.step(random_action)


        observation_list.append(observation)
        next_obervation_list.append(next_observation)
        reward_list.append(reward)
        terminal_list.append(terminal)
        actions_list.append(random_action)

        if terminal:
            print("Found the end!")
            break

    dict_data['observations']=np.array(observation_list)
    dict_data['next_observations'] = np.array(next_obervation_list)
    dict_data['rewards'] = np.array(reward_list)
    dict_data['terminals'] = np.array(terminal_list)

    dict_data['all_observations'] = np.concatenate(
                [dict_data['observations'], dict_data['next_observations'][-1:]], axis=0)
    dict_data['actions'] = np.array(actions_list,dtype=np.int64)


    np.savez(name+f'/test_{i:04}', **dict_data)
    print(i)

