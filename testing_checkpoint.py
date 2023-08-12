r'''
for s in {1..100}; do
env SEED=$s TEMP=1 CKPT_NAME="checkpoint_00077_00026.pth" RAND_GOAL=1 DYN=1 python testing_checkpoint.py
done
'''

import os
import sys
import torch
from omegaconf import OmegaConf, SCMode
import yaml
from distutils.util import strtobool

from quasimetric_rl.data import Dataset
from quasimetric_rl.modules import QRLAgent, QRLConf
from quasimetric_rl import utils

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
import numpy as np
from typing import cast

from visualize_trajectories import visualize_trajectory

expr_checkpoint = 'offline/results/test_direction/test.pth'  # FIXME

expr_checkpoint = 'offline/results/d4rl_custom-grid-tank-goal-randinit-v1/iqe(dim=2048,components=64)_dyn=1_seed=60912/checkpoint_00025_03549.pth'

expr_checkpoint = 'offline/results/d4rl_custom-grid-tank-goal-v1/iqe(dim=2048,components=64)_dyn=1_seed=60912_4dir/checkpoint_00062_00000_final.pth'

dyn = float(os.environ['DYN'])
expr_checkpoint = f"offline/results/d4rl_custom-grid-tank-goal-ez-v1/iqe(dim=2048,components=64)_dyn={dyn:g}_seed=60912/checkpoint_00273_00000_final.pth"
ckpt_name = os.environ.get('CKPT_NAME', 'checkpoint_00273_00000_final.pth')
if not ckpt_name.endswith('.pth'):
    ckpt_name += '.pth'
expr_checkpoint = f"offline/results/d4rl_custom-grid-tank-goal-tz-normG-randG-v1/iqe(dim=2048,components=64)_dyn={dyn:g}_seed=60912/{ckpt_name}"

print(expr_checkpoint)

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
seed = int(os.environ.get('SEED', '4567'))

torch_seed, np_seed, env_seed = utils.split_seed(cast(int, seed), 3)
np.random.seed(np.random.Generator(np.random.PCG64(np_seed)).integers(1 << 31))
torch.manual_seed(np.random.Generator(np.random.PCG64(torch_seed)).integers(1 << 31))
env.seed(np.random.Generator(np.random.PCG64(env_seed)).integers(1 << 31))

observation = env.reset()
rand_goal = strtobool(os.environ.get('RAND_GOAL', '0'))
if rand_goal:
    env.randomize_goal()
temp = float(os.environ.get('TEMP', '0'))
observation = env.get_observation()
if hasattr(env, 'get_goal_observation'):
    # tz's version
    goal_obs = torch.tensor(env.get_goal_observation(), dtype=torch.float32)
else:
    # original env
    pos_goal = np.concatenate( [env.goal, np.zeros(2)])
    goal_obs = torch.tensor(pos_goal, dtype=torch.float32)
trajectory_points = []

done = False
for i in range(1000):

  observation_tensor = torch.tensor(observation,  dtype=torch.float32)
  trajectory_points.append(observation)
  if done:
    break

  with torch.no_grad():
    distances = critic(
      observation_tensor[None, :],
      goal_obs[None, :],
      action=actions,
    )

  if temp == 0:
    best_action = distances.argmin(dim=0).item()
  else:
    best_action = torch.distributions.Categorical(logits=(-distances) / temp).sample()
  # if best_action == 1:
  #     import pdb; pdb.set_trace()
  observation, _, done, _ = env.step(best_action)

  print(i, observation, best_action, distances.data.numpy())

trajectory_points = np.array(trajectory_points)

savedir = f"agent_dyn={dyn:g}/{ckpt_name.rsplit('.', 1)[0]}/plan_randG={str(rand_goal)[0]}_tau={temp:g}/"
utils.mkdir(savedir)
visualize_trajectory((trajectory_points + 1) / 2 * (env.size - 1),env, name=f"{savedir}/s={seed:06d}_trajectory.gif")

print(f"{savedir}/s={seed:06d}_trajectory.gif")
with open(f"{savedir}/s={seed:06d}.json", "w") as f:
    import json
    print(dict(succ=done, ts=trajectory_points.shape[0]))
    print(json.dumps(dict(succ=done, ts=trajectory_points.shape[0])), file=f)

