from __future__ import annotations
from typing import *

""" A pointmass maze env."""
import numpy as np
from sklearn import preprocessing

from d4rl.pointmaze import maze_model


from ..base import register_offline_env
from . import load_environment, convert_dict_to_EpisodeData_iter, sequence_dataset
from .maze2d import preprocess_maze2d_fix
from .type_of_mazes import generate_maze, convert_float_maze_to_string
import h5py

env_seed = 0
list_strings = []

#TODO:
# Convert to one hot: torch.nn.functional.one_hot
def load_maze2d_custom_string():

    if env_seed == len(list_strings):
        dataset_string = 'dataset_resources/paths_mazes/' + f'maze2d-custom-v0_{str(env_seed).zfill(3)}.hdf5'


        with h5py.File(dataset_string, 'r') as dataset_file:
                choosen_maze_Layout = dataset_file['environment_attributes'][0]

        list_strings.append(choosen_maze_Layout)
        
    else:
         choosen_maze_Layout = list_strings[env_seed]
    
    chosen_maze = convert_float_maze_to_string(choosen_maze_Layout)

    return chosen_maze


def update_env_seed(new_seed):
    global env_seed
    env_seed = new_seed


def load_episodes_maze2d_custom():

    chosen_maze = load_maze2d_custom_string()


    offline_maze = maze_model.MazeEnv(chosen_maze)
    offline_maze.name = 'test_custom'

    yield from convert_dict_to_EpisodeData_iter(
        sequence_dataset(
            offline_maze,
            preprocess_maze2d_fix(
                env=offline_maze,
                dataset=offline_maze.get_dataset(
                    h5path='dataset_resources/paths_mazes/' + f'maze2d-custom-v0_{str(env_seed).zfill(3)}.hdf5')
            ),
        ),
    )


def load_environment_custom():

    chosen_maze = load_maze2d_custom_string()

    env = maze_model.MazeEnv(chosen_maze)
    env_proccess = load_environment(env)

    return env_proccess


register_offline_env(
    'd4rl', 'maze2d-custom',
    create_env_fn=load_environment_custom,
    load_episodes_fn=load_episodes_maze2d_custom,
)
