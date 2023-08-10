import numpy as np

from quasimetric_rl.data.d4rl.grid_tank_goal import Tank_reach_goal
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

def visualize_trajectory(trajectory_points, env,  name = "visualize_trajectory.gif"):
    num_frames = len(trajectory_points)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(5,5)

    def animate(i):
        ax.clear()
        # Get the point from the points list at index i
        observation = trajectory_points[i,:]
        point = observation[0:2]

        plt.xlim([0, env.size+1])
        plt.ylim([0, env.size+1])

        # Plot that point using the x and y coordinates
        ax.plot(point[0], point[1], color='green', 
                label='original', marker='o')
        
        goal_point = env.goal
        ax.plot(goal_point[0], goal_point[1], color='red', 
                label='original', marker='o')



    ani = FuncAnimation(fig, animate, frames=num_frames,
                        interval=1, repeat=False)

    ani.save(name, dpi=300,
            writer=PillowWriter(fps=30))
    plt.close()


if __name__ == "__main__":

    env = Tank_reach_goal()

    name = 'trajectories_custom/test_0970.npz'

    dict_episode = np.load(name)

    trajectory_points = dict_episode['observations']
    visualize_trajectory(trajectory_points, env)