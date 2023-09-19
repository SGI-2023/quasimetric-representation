from __future__ import annotations
from typing import *

""" A pointmass maze env."""
from gym.envs.mujoco import mujoco_env
from gym import utils
from d4rl import offline_env
from d4rl.pointmaze.dynamic_mjc import MJCModel
import numpy as np
import random
from sklearn import preprocessing
import torch

import functools

from d4rl.pointmaze import maze_model


from ..base import register_offline_env
from . import load_environment, convert_dict_to_EpisodeData_iter, sequence_dataset
from .maze2d import preprocess_maze2d_fix


LARGE_MAZE = \
        "############\\"+\
        "#OOOO#OOOOO#\\"+\
        "#O##O#O#O#O#\\"+\
        "#OOOOOO#OOO#\\"+\
        "#O####O###O#\\"+\
        "#OO#O#OOOOO#\\"+\
        "##O#O#O#O###\\"+\
        "#OO#OOO#OGO#\\"+\
        "############"

LARGE_MAZE_EVAL = \
        "############\\"+\
        "#OO#OOO#OGO#\\"+\
        "##O###O#O#O#\\"+\
        "#OO#O#OOOOO#\\"+\
        "#O##O#OO##O#\\"+\
        "#OOOOOO#OOO#\\"+\
        "#O##O#O#O###\\"+\
        "#OOOO#OOOOO#\\"+\
        "############"

MEDIUM_MAZE = \
        '########\\'+\
        '#OO##OO#\\'+\
        '#OO#OOO#\\'+\
        '##OOO###\\'+\
        '#OO#OOO#\\'+\
        '#O#OO#O#\\'+\
        '#OOO#OG#\\'+\
        "########"

MEDIUM_MAZE_EVAL = \
        '########\\'+\
        '#OOOOOG#\\'+\
        '#O#O##O#\\'+\
        '#OOOO#O#\\'+\
        '###OO###\\'+\
        '#OOOOOO#\\'+\
        '#OO##OO#\\'+\
        "########"

SMALL_MAZE = \
        "######\\"+\
        "#OOOO#\\"+\
        "#O##O#\\"+\
        "#OOOO#\\"+\
        "######"

U_MAZE = \
        "#####\\"+\
        "#GOO#\\"+\
        "###O#\\"+\
        "#OOO#\\"+\
        "#####"

U_MAZE_EVAL = \
        "#####\\"+\
        "#OOG#\\"+\
        "#O###\\"+\
        "#OOO#\\"+\
        "#####"

OPEN = \
        "#######\\"+\
        "#OOOOO#\\"+\
        "#OOGOO#\\"+\
        "#OOOOO#\\"+\
        "#######"


type_of_maze = U_MAZE

def pre_process_maze2d_fix_custom(env: 'd4rl.pointmaze.MazeEnv', dataset: Mapping[str, np.ndarray]):
    dataset_fix = preprocess_maze2d_fix(env, dataset)
    size_of_dataset = dataset_fix['actions'].shape[0]

    le = preprocessing.LabelEncoder()

    maze_splitted_char = list(type_of_maze)
    
    encoded_info = le.fit_transform(maze_splitted_char)

    type_of_maze_data = np.array(encoded_info)[None,...]

    type_of_maze_data_expanded = np.repeat(type_of_maze_data, size_of_dataset, axis=0)
    dataset_fix['environment_attributes'] = type_of_maze_data_expanded

    return dataset_fix

def load_episodes_maze2d_custom():
    offline_maze = maze_model.MazeEnv(type_of_maze)
    offline_maze.name = 'test_custom'

    yield from convert_dict_to_EpisodeData_iter(
        sequence_dataset(
            offline_maze,
            pre_process_maze2d_fix_custom(
                offline_maze,
                offline_maze.get_dataset(h5path='quasimetric_rl/data/d4rl/maze2d-umaze-v1.hdf5'),
            ),
        ),
    )

def load_environment_custom():
    env = maze_model.MazeEnv(type_of_maze)
    env_proccess = load_environment(env)

    return env_proccess


register_offline_env(
        'd4rl', 'maze2d-custom',
        create_env_fn=load_environment_custom,
        load_episodes_fn=load_episodes_maze2d_custom,
    )