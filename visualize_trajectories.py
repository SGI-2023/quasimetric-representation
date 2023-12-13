import numpy as np

from quasimetric_rl.data.d4rl.type_of_mazes import convert_float_maze_to_string
from d4rl.pointmaze import maze_model
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
import cv2
import h5py

env_seed = 0

def create_circle_in_image_given_array(image_array:np.array, circle_centers_array_float:np.array):
	circle_centers_array_float = circle_centers_array_float + 0.5
	circle_centers_array_float = circle_centers_array_float.astype(int)
	for point in circle_centers_array_float:
		image_array[point[0], point[1], :] = (0,0,255)

if __name__ == "__main__":

	dataset_string = 'dataset_resources/paths_mazes_s/' + f'maze2d-custom-v0_{str(env_seed).zfill(3)}.hdf5'


	with h5py.File(dataset_string, 'r') as dataset_file:
		dataset_obs = dataset_file["observations"]
		choosen_maze_Layout = dataset_file['environment_attributes'][0]

	chosen_maze_string = convert_float_maze_to_string(choosen_maze_Layout)
	offline_env = maze_model.MazeEnv(maze_spec=chosen_maze_string)
	dataset_maze = offline_env.get_dataset(h5path=dataset_string)

	particle_position_trajectory = dataset_maze['observations'][:, :2]

	# Creating an empty RGB image
	rgb_image_array = np.zeros((choosen_maze_Layout.shape[0], choosen_maze_Layout.shape[1], 3), dtype=np.uint8)

	create_circle_in_image_given_array(rgb_image_array, particle_position_trajectory)


	# Assigning the binary values to the red channel
	rgb_image_array[:, :, 0] = choosen_maze_Layout * 255  # Multiply by 255 to get the full intensity of red

	# Using matplotlib to save the RGB image
	plt.imshow(rgb_image_array)
	plt.axis('off')  # Turn off axis numbers and labels
	plt.axis('off')  # Turn off axis numbers and labels

	# Save the image
	image_filename = 'binary_image.png'
	plt.savefig(image_filename, bbox_inches='tight', pad_inches=0)


	print(choosen_maze_Layout)