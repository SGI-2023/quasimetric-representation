import numpy as np

from quasimetric_rl.data.d4rl.type_of_mazes import convert_float_maze_to_string
from d4rl.pointmaze import maze_model
import matplotlib.pyplot as plt

import os
import h5py
import torch

from omegaconf import OmegaConf, SCMode
import yaml
from distutils.util import strtobool

from quasimetric_rl.data import Dataset
from quasimetric_rl.modules import QRLAgent, QRLConf
from quasimetric_rl import utils

import matplotlib.pyplot as plt
from typing import cast
import sys

env_seed = 0

def create_circle_in_image_given_array(image_array:np.array, circle_centers_array_float:np.array, distances_array:np.array):
	circle_centers_array_float = circle_centers_array_float + 0.5
	circle_centers_array_float = circle_centers_array_float.astype(int)
	
	distances_array = distances_array/np.max(distances_array)

	counter = 0

	for point,distance in zip(circle_centers_array_float, distances_array):

		image_array[point[0], point[1], :] = (0,0,255*distance)

		counter = counter + 1

	image_array[circle_centers_array_float[0,0], circle_centers_array_float[0,1], :] = (0,255,255*distance)

if __name__ == "__main__":

	dataset_string = 'dataset_resources/paths_mazes/' + f'maze2d-custom-v0_{str(env_seed).zfill(3)}.hdf5'

	with h5py.File(dataset_string, 'r') as dataset_file:
		dataset_obs = dataset_file["observations"]
		choosen_maze_Layout = dataset_file['environment_attributes'][0]

	chosen_maze_string = convert_float_maze_to_string(choosen_maze_Layout)
	offline_env = maze_model.MazeEnv(maze_spec=chosen_maze_string)
	dataset_maze = offline_env.get_dataset(h5path=dataset_string)

	number_evals = 50000


	expr_checkpoint = f"objectives/iqe(dim=2048,components=64)_dyn=1_actor(goal=Rand,BC=0.05)_seed=20000000/checkpoint_04590_00049.pth"
	

	print('expr', expr_checkpoint)
	seed = int(os.environ.get('SEED', '4567'))
	temp = float(os.environ.get('TEMP', '0'))

	savedir = f"test/"


	expr_dir = os.path.dirname(expr_checkpoint)
	with open(expr_dir + '/config.yaml', 'r') as f:
		# load saved conf
		conf = OmegaConf.create(yaml.safe_load(f))


	# 1. How to create env
	dataset: Dataset = Dataset.Conf(kind=conf.env.kind, name=conf.env.name).make()  # dummy: don't load data
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
	actions = torch.tensor([0, 1, 2, 3])
	critic= agent.critics[0]

	observations = torch.tensor(dataset_maze['observations'][:1,:]).repeat(number_evals,1)
	observations_next = torch.tensor(dataset_maze['observations'][:number_evals,:])

	environment_attributes = torch.tensor(dataset_maze['environment_attributes'][:number_evals])

	with torch.no_grad():
			zx, zy = critic.get_encoded_attributes(observations, observations_next, environment_attributes)
			distances = critic.quasimetric_model(zx, zy)

	particle_position_trajectory = observations_next[:, :2].numpy()

	# Creating an empty RGB image
	rgb_image_array = np.zeros((choosen_maze_Layout.shape[0], choosen_maze_Layout.shape[1], 3), dtype=np.uint8)

	create_circle_in_image_given_array(rgb_image_array, particle_position_trajectory, distances.numpy())

	# Assigning the binary values to the red channel
	rgb_image_array[:, :, 0] = choosen_maze_Layout * 255  # Multiply by 255 to get the full intensity of red

	# Using matplotlib to save the RGB image
	plt.imshow(rgb_image_array)
	plt.axis('off')  # Turn off axis numbers and labels
	plt.axis('off')  # Turn off axis numbers and labels

	# Save the image
	image_filename = 'binary_image.png'
	plt.savefig(image_filename, bbox_inches='tight', pad_inches=0)



	print("finished")
