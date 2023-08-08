from gym import Env, spaces

import numpy as np
import random
from stable_baselines import A2C

from stable_baselines.common.env_checker import check_env


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

    
maze_env = Maze_simple()

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

 
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(5,5)

print(check_env(maze_env))

'''
for i in range(1000):
    dict = np.load(f'trajectories/test_{i:04}.npz')
    print(list(dict.keys()))

exit(0)
'''

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


    np.savez(f'trajectories/test_{i:04}', **dict_data)
    print(i)


print(check_env(maze_env))