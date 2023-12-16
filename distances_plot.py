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
from collections import deque

from scipy.stats import linregress


env_seed = 0

def create_circle_in_image_given_array(image_array:np.array, circle_centers_array_float:np.array, distances_array:np.array):
	circle_centers_array_float = circle_centers_array_float + 0.5
	circle_centers_array_float = circle_centers_array_float.astype(int)
	
	distances_array = distances_array/np.max(distances_array)

	counter = 0

	for point,distance in zip(circle_centers_array_float, distances_array):

		image_array[point[0], point[1], 2] = (255*distance)

		counter = counter + 1

	image_array[circle_centers_array_float[0,0], circle_centers_array_float[0,1], :] = (0,255,255*distances_array[0])


def create_observations_from_maze_spec(maze_spec:np.array):
	observations = []
	for row in range(maze_spec.shape[0]):
		for column in range(maze_spec.shape[1]):
			if maze_spec[row, column] == 0:
				observations.append([row-0.5, column-0.5, 0, 0])

	observations = np.array(observations)
	return observations

def get_distances_from_quasimetrics(observations_start, observations_next, environment_attributes, critic):
	with torch.no_grad():
		zx, zy = critic.get_encoded_attributes(observations_start, observations_next, environment_attributes)
		distances = critic.quasimetric_model(zx, zy)
	return distances


def min_distance(maze, start):
    """
    Calculate the minimum distance to travel from the start state to any other state in the maze.
    
    :param maze: 2D list representing the maze with 0s as passable and 1s as impassable.
    :param start: Tuple (x, y) representing the starting coordinates.
    :return: 2D list with the minimum distance from the start to each cell. -1 for impassable cells.
    """
    rows, cols = len(maze), len(maze[0])
    distance = [[-1 for _ in range(cols)] for _ in range(rows)]  # Initialize distance array with -1

    # Directions: up, down, left, right
    directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]

    # BFS queue
    queue = deque([start])
    distance[start[0]][start[1]] = 0  # Distance to the start cell is 0

    while queue:
        x, y = queue.popleft()

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Check if the new position is within the maze and not a wall
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 0 and distance[nx][ny] == -1:
                distance[nx][ny] = distance[x][y] + 1
                queue.append((nx, ny))

    return distance

def get_distances_ground_truth(observations_start, observations_next, environment_attributes):
	start = observations_start[0, :2].to(int) + 1
	next_obs = observations_next[:, :2].to(int) + 1

	distance_matrix = min_distance(environment_attributes[0], start)
	distance_matrix = torch.tensor(distance_matrix)

	distances = []
	for obs in next_obs:
		distances.append(distance_matrix[obs[0], obs[1]])

		
	return torch.tensor(distances)




def calculate_distances(dataset_path: str, critic: torch.nn.Module) -> tuple:
	all_distances_pred = []
	all_distances_gt = []

	for env_seed in range(50):
		dataset_string = dataset_path + f'maze2d-custom-v0_{str(env_seed).zfill(3)}.hdf5'

		with h5py.File(dataset_string, 'r') as dataset_file:
			choosen_maze_Layout = dataset_file['environment_attributes'][0]

		chosen_maze_string = convert_float_maze_to_string(choosen_maze_Layout)
		offline_env = maze_model.MazeEnv(maze_spec=chosen_maze_string)
		dataset_maze = offline_env.get_dataset(h5path=dataset_string)

		observations = create_observations_from_maze_spec(choosen_maze_Layout)

		num_spaces = len(observations)

		first_observation = observations[0, :]
		observations_start = np.tile(first_observation[np.newaxis, :], (num_spaces, 1))
		observations_next = observations

		observations_start = torch.tensor(observations_start, dtype=torch.float32)
		observations_next = torch.tensor(observations_next, dtype=torch.float32)

		environment_attributes = torch.tensor(dataset_maze['environment_attributes'][:1, :, :]).repeat(num_spaces, 1, 1)

		distances_ground_truth = get_distances_ground_truth(observations_start, observations_next, environment_attributes)
		distances_predicted = get_distances_from_quasimetrics(observations_start, observations_next, environment_attributes, critic)

		all_distances_pred.extend(distances_predicted)
		all_distances_gt.extend(distances_ground_truth)

	return all_distances_pred, all_distances_gt



if __name__ == "__main__":



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

	all_distances_pred, all_distances_gt = calculate_distances('dataset_resources/evaluation_mazes/paths_mazes/', critic)

	plt.scatter(all_distances_pred, all_distances_gt,s=0.7,color='blue', label='Untrained Mazes')



	all_distances_pred, all_distances_gt = calculate_distances('dataset_resources/trained_mazes/paths_mazes/', critic)

	plt.scatter(all_distances_pred, all_distances_gt,s=0.7,color='red', label='Trained Mazes')

	plt.xlabel('Distances predicted')
	plt.ylabel('Distances ground truth')

	# Adding legend
	plt.legend()

	plt.savefig( f'scatter_plot_distances_predicted_distances_ground_truth.png')
	plt.close()
