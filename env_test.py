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
        print(distance_to_goal)
        print()
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
exit(0)
 
model = A2C("MlpPolicy", maze_env, verbose=1)
model.learn(total_timesteps=1000000)

vec_env = model.get_env()
point = vec_env.reset()

def animate(i):
    ax.clear()
    # Get the point from the points list at index i
    action, _states = model.predict(point)
    point, rewards, dones, info = maze_env.step(action)

    # Plot that point using the x and y coordinates
    ax.plot(point[0], point[1], color='green', 
            label='original', marker='o')
    
    goal_point = maze_env.goal
    ax.plot(goal_point[0], goal_point[1], color='red', 
            label='original', marker='o')



ani = FuncAnimation(fig, animate, frames=500,
                    interval=500, repeat=False)

ani.save("simple_animation.gif", dpi=300,
         writer=PillowWriter(fps=1))
plt.close()

