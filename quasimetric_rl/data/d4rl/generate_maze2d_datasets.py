from d4rl.pointmaze import waypoint_controller
from d4rl.pointmaze import maze_model
import numpy as np
import h5py
import argparse

from type_of_mazes import generate_maze, display_maze, convert_maze_array_to_float, convert_float_maze_to_string


def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'rewards': [],
            'infos/goal': [],
            'infos/qpos': [],
            'infos/qvel': [],
            }


def append_data(data, s, a, tgt, done, env_data):
    data['observations'].append(s)
    data['actions'].append(a)
    data['rewards'].append(0.0)
    data['terminals'].append(done)
    data['infos/goal'].append(tgt)
    data['infos/qpos'].append(env_data.qpos.ravel().copy())
    data['infos/qvel'].append(env_data.qvel.ravel().copy())


def npify(data):
    for k in data:
        if k == 'terminals':
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)


def maze_generator():

    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true',
                        help='Render trajectories')
    parser.add_argument('--noisy', action='store_true', help='Noisy actions')
    parser.add_argument('--env_name', type=str,
                        default='maze2d-custom-v0', help='Maze type')
    parser.add_argument('--num_samples', type=int,
                        default=int(1e6), help='Num samples to collect')
    parser.add_argument('--dim', type=int, default=15,
                        help='dimensions of the maze')

    parser.add_argument('--max_episode_steps', type=int, default=5000,
                        help='dimensions of the maze')

    parser.add_argument('--dataset_folder', type=str,
                        default='dataset_resources/paths_mazes/')
    
    parser.add_argument('--num_envs', type=int, default=50)

    args = parser.parse_args()

    for maze_seed in range(args.num_envs):
        print(maze_seed)
        create_dataset_path(maze_seed, args)

def create_dataset_path(maze_seed:int, args):
    maze_spec = generate_maze(args.dim, args.dim, maze_seed)

    maze_string = display_maze(maze_spec)

    controller = waypoint_controller.WaypointController(maze_string)


    env = maze_model.MazeEnv(maze_spec=maze_string)

    env.set_target()
    s = env.reset()
    act = env.action_space.sample()
    done = False

    data = reset_data()
    ts = 0
    for time in range(args.num_samples):

        position = s[0:2]
        velocity = s[2:4]
        act, done = controller.get_action(position, velocity, env._target)
        if args.noisy:
            act = act + np.random.randn(*act.shape)*0.5

        act = np.clip(act, -1.0, 1.0)
        if ts >= args.max_episode_steps:
            done = True
        append_data(data, s, act, env._target, done, env.sim.data)

        ns, _, _, _ = env.step(act)

        if len(data['observations']) % 10000 == 0:
            print(len(data['observations']))

        ts += 1
        if done:
            env.set_target()
            done = False
            ts = 0
        else:
            s = ns

        if args.render:
            env.render()


    type_of_maze_data = convert_maze_array_to_float(maze_spec)[None, ...]

    type_of_maze_data_expanded = np.repeat(
        type_of_maze_data, len(data['observations']), axis=0)
    data['environment_attributes'] = type_of_maze_data_expanded

    fname = args.dataset_folder + f'{args.env_name}_{str(maze_seed).zfill(3)}.hdf5'

    dataset = h5py.File(fname, 'w')
    npify(data)
    for k in data:
        dataset.create_dataset(k, data=data[k], compression='gzip')


if __name__ == '__main__':

    maze_generator()
