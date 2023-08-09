import os
import torch
from omegaconf import OmegaConf, SCMode
import yaml

from quasimetric_rl.data import Dataset
from quasimetric_rl.modules import QRLAgent, QRLConf

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
 
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(5,5)

expr_checkpoint = 'offline/results_old/d4rl_custom-grid-umaze-v1/iqe(dim=2048,components=64)_dyn=1_seed=60912/checkpoint_00082_00000_final.pth'  # FIXME

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
