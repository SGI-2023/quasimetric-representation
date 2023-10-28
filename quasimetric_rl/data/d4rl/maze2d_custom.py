from __future__ import annotations
from typing import *

""" A pointmass maze env."""
import numpy as np
from sklearn import preprocessing

from d4rl.pointmaze import maze_model


from ..base import register_offline_env
from . import load_environment, convert_dict_to_EpisodeData_iter, sequence_dataset
from .maze2d import preprocess_maze2d_fix
from .type_of_mazes import generate_maze

env_seed = 0


def pre_process_maze2d_fix_custom(env: 'd4rl.pointmaze.MazeEnv', dataset: Mapping[str, np.ndarray],seed: int):
    dataset_fix = preprocess_maze2d_fix(env, dataset)
    size_of_dataset = dataset_fix['actions'].shape[0]

    le = preprocessing.LabelEncoder()

    chosen_maze = generate_maze(15, 15, seed)

    maze_splitted_char = list(chosen_maze)

    encoded_info = le.fit_transform(maze_splitted_char)

    type_of_maze_data = np.array(encoded_info)[None, ...]

    type_of_maze_data_expanded = np.repeat(
        type_of_maze_data, size_of_dataset, axis=0)
    dataset_fix['environment_attributes'] = type_of_maze_data_expanded

    return dataset_fix


def load_episodes_maze2d_custom():
    chosen_maze = generate_maze(15, 15, env_seed)

    offline_maze = maze_model.MazeEnv(chosen_maze)
    offline_maze.name = 'test_custom'

    yield from convert_dict_to_EpisodeData_iter(
        sequence_dataset(
            offline_maze,
            pre_process_maze2d_fix_custom(
                env=offline_maze,
                dataset=offline_maze.get_dataset(
                    h5path='dataset_resources/paths_mazes/' + f'maze2d-custom-v0_{str(env_seed).zfill(3)}.hdf5'),
                seed=env_seed
            ),
        ),
    )


def load_environment_custom():
    chosen_maze = generate_maze(15, 15, env_seed)

    env = maze_model.MazeEnv(chosen_maze)
    env_proccess = load_environment(env)

    return env_proccess


register_offline_env(
    'd4rl', 'maze2d-custom',
    create_env_fn=load_environment_custom,
    load_episodes_fn=load_episodes_maze2d_custom,
)
