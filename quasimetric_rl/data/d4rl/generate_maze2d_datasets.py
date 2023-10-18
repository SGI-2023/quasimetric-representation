import gym
from d4rl.pointmaze import waypoint_controller
from d4rl.pointmaze import maze_model
import numpy as np
import h5py
import argparse
from sklearn import preprocessing

from type_of_mazes import generate_maze


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


def maze_generator(maze_seed):

    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true',
                        help='Render trajectories')
    parser.add_argument('--noisy', action='store_true', help='Noisy actions')
    parser.add_argument('--env_name', type=str,
                        default='maze2d-umaze-v1', help='Maze type')
    parser.add_argument('--num_samples', type=int,
                        default=int(1e4), help='Num samples to collect')
    parser.add_argument('--dim', type=int, default=19,
                        help='dimensions of the maze')

    args = parser.parse_args()

    maze_spec = generate_maze(args.dim, args.dim, maze_seed)
    env = gym.make(args.env_name)
    max_episode_steps = env._max_episode_steps

    controller = waypoint_controller.WaypointController(maze_spec)
    env = maze_model.MazeEnv(maze_spec)

    env.set_target()
    s = env.reset()
    act = env.action_space.sample()
    done = False

    data = reset_data()
    ts = 0
    for time in range(args.num_samples):

        print(time)

        position = s[0:2]
        velocity = s[2:4]
        act, done = controller.get_action(position, velocity, env._target)
        if args.noisy:
            act = act + np.random.randn(*act.shape)*0.5

        act = np.clip(act, -1.0, 1.0)
        if ts >= max_episode_steps:
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

    le = preprocessing.LabelEncoder()
    maze_splitted_char = list(maze_spec)

    encoded_info = le.fit_transform(maze_splitted_char)

    type_of_maze_data = np.array(encoded_info)[None, ...]

    type_of_maze_data_expanded = np.repeat(
        type_of_maze_data, len(data['observations']), axis=0)
    data['environment_attributes'] = type_of_maze_data_expanded

    if args.noisy:
        fname = '%s-noisy.hdf5' % args.env_name + str(maze_seed).zfill(6)
    else:
        fname = '%s.hdf5' % args.env_name + str(maze_seed).zfill(6)
    dataset = h5py.File(fname, 'w')
    npify(data)
    for k in data:
        dataset.create_dataset(k, data=data[k], compression='gzip')


def main():

    for i in range(50):
        print(i)
        maze_generator(i)



if __name__ == "__main__":
    main()
