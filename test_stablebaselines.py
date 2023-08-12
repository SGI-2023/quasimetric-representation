import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import A2C
from stable_baselines.common.env_checker import check_env

from quasimetric_rl.data.d4rl.grid_tank_goal import Tank_reach_goal
from visualize_trajectories import visualize_trajectory



env = Tank_reach_goal()

done = False

while not done:
    _ = env.reset()

    model = A2C(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=200000)
    trajectory_points = []
    observation = env.reset()

    for i in range(1000):
        
        trajectory_points.append(observation)

        action, _states = model.predict(observation)
        observation, _, done, _ = env.step(action)
        
        if done:
            print("Found it!")
            break 

    print("Finished a training and test setting")

trajectory_points = np.array(trajectory_points)

visualize_trajectory(trajectory_points,env, name="stable_baselines_grount_truth.gif")

