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

def load_maze2d_custom_string():
    dataset_string = 'dataset_resources/paths_mazes/' + f'maze2d-custom-v0_{str(env_seed).zfill(3)}.hdf5'


    with h5py.File(dataset_string, 'r') as dataset_file:
            choosen_maze_Layout = dataset_file['environment_attributes'][0]

    
    chosen_maze = convert_float_maze_to_string(choosen_maze_Layout)

    return chosen_maze


def update_env_seed(new_seed):
    global env_seed
    env_seed = new_seed


def pre_process_maze2d_fix_custom(env: 'd4rl.pointmaze.MazeEnv', dataset: Mapping[str, np.ndarray], maze_string: str):
    dataset_fix = preprocess_maze2d_fix(env, dataset)
    size_of_dataset = dataset_fix['actions'].shape[0]

    le = preprocessing.LabelEncoder()

    maze_splitted_char = list(maze_string)

    encoded_info = le.fit_transform(maze_splitted_char)

    type_of_maze_data = np.array(encoded_info)[None, ...]

    type_of_maze_data_expanded = np.repeat(
        type_of_maze_data, size_of_dataset, axis=0)
    dataset_fix['environment_attributes'] = type_of_maze_data_expanded

    return dataset_fix


def load_episodes_maze2d_custom():

    chosen_maze = load_maze2d_custom_string()


    offline_maze = maze_model.MazeEnv(chosen_maze)
    offline_maze.name = 'test_custom'

    yield from convert_dict_to_EpisodeData_iter(
        sequence_dataset(
            offline_maze,
            pre_process_maze2d_fix_custom(
                env=offline_maze,
                dataset=offline_maze.get_dataset(
                    h5path='dataset_resources/paths_mazes/' + f'maze2d-custom-v0_{str(env_seed).zfill(3)}.hdf5'),
                maze_string=chosen_maze
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
