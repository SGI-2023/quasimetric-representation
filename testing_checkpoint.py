import os
import torch
from omegaconf import OmegaConf, SCMode
import yaml

from quasimetric_rl.data import Dataset
from quasimetric_rl.modules import QRLAgent, QRLConf

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
import numpy as np
 
from visualize_trajectories import visualize_trajectory

expr_checkpoint = 'offline/results/test_direction/test.pth'  # FIXME

expr_dir = os.path.dirname(expr_checkpoint)
with open(expr_dir + '/config.yaml', 'r') as f:
    # load saved conf
    conf = OmegaConf.create(yaml.safe_load(f))


# 1. How to create env
dataset: Dataset = Dataset.Conf(kind=conf.env.kind, name=conf.env.name).make(dummy=True)  # dummy: don't load data
env = dataset.create_env()  # <-- you can use this now!
# episodes = list(dataset.load_episodes())  # if you want to load episodes for offline data


# 2. How to re-create QRL agent
agent_conf: QRLConf = OmegaConf.to_container(
  OmegaConf.merge(OmegaConf.structured(QRLConf()), conf.agent),  # overwrite with loaded conf
  structured_config_mode=SCMode.INSTANTIATE,  # create the object
)
agent: QRLAgent = agent_conf.make(env_spec=dataset.env_spec, total_optim_steps=1)[0]  # you can move to your fav device


# 3. Load checkpoint
agent.load_state_dict(torch.load(expr_checkpoint, map_location='cpu')['agent'])
actions = torch.tensor([0, 1, 2])  
critic= agent.critics[0]

# greedy 1-step planning
observation = env.reset()
pos_goal = np.concatenate( [env.goal, np.zeros(2)])
goal_obs = torch.tensor(pos_goal, dtype=torch.float32)
trajectory_points = []

done = False
for i in range(1000):
  
  observation_tensor = torch.tensor(observation,  dtype=torch.float32)
  trajectory_points.append(observation)
  if done:
    break 
  
  distances = critic(
      observation_tensor[None, :],  
      goal_obs[None, :],
      action=actions,   
  )  
  
  best_action = distances.argmin(dim=0)  
  observation, _, done, _ = env.step(best_action)
  
  print(i)

trajectory_points = np.array(trajectory_points)

visualize_trajectory(trajectory_points,env, name="agent_trajectory2.gif")