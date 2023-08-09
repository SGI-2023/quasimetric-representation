import numpy as np

from quasimetric_rl.data.d4rl.grid_tank_goal import Tank_reach_goal
import matplotlib.pyplot as plt

env = Tank_reach_goal()

name = 'trajectories_custom/test_0200.npz'

dict_episode = np.load(name)

trajectory_points = dict_episode['observations']

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(5,5)


from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

def animate(i):
    ax.clear()
    # Get the point from the points list at index i
    point = trajectory_points[i,:]

    plt.xlim([0, env.size])
    plt.ylim([0, env.size])

    # Plot that point using the x and y coordinates
    ax.plot(point[0], point[1], color='green', 
            label='original', marker='o')
    
    goal_point = env.goal
    ax.plot(goal_point[0], goal_point[1], color='red', 
            label='original', marker='o')



ani = FuncAnimation(fig, animate, frames=300,
                    interval=1, repeat=False)

ani.save("simple_animation.gif", dpi=300,
         writer=PillowWriter(fps=1))
plt.close()