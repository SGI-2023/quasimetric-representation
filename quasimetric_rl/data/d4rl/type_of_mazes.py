from __future__ import annotations
from typing import *
import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


LAROE_MAZE = \
    "############\\" +\
    "#OOOO#OOOOO#\\" +\
    "#O##O#O#O#O#\\" +\
    "#OOOOOO#OOO#\\" +\
    "#O####O###O#\\" +\
    "#OO#O#OOOOO#\\" +\
    "##O#O#O#O###\\" +\
    "#OO#OOO#OOO#\\" +\
    "############"

LAROE_MAZE_EVAL = \
    "############\\" +\
    "#OO#OOO#OOO#\\" +\
    "##O###O#O#O#\\" +\
    "#OO#O#OOOOO#\\" +\
    "#O##O#OO##O#\\" +\
    "#OOOOOO#OOO#\\" +\
    "#O##O#O#O###\\" +\
    "#OOOO#OOOOO#\\" +\
    "############"

MEDIUM_MAZE = \
    '########\\' +\
    '#OO##OO#\\' +\
    '#OO#OOO#\\' +\
    '##OOO###\\' +\
    '#OO#OOO#\\' +\
    '#O#OO#O#\\' +\
    '#OOO#OO#\\' +\
    "########"

MEDIUM_MAZE_EVAL = \
    '########\\' +\
    '#OOOOOO#\\' +\
    '#O#O##O#\\' +\
    '#OOOO#O#\\' +\
    '###OO###\\' +\
    '#OOOOOO#\\' +\
    '#OO##OO#\\' +\
    "########"

SMALL_MAZE = \
    "######\\" +\
    "#OOOO#\\" +\
    "#O##O#\\" +\
    "#OOOO#\\" +\
    "######"

U_MAZE = \
    "#####\\" +\
    "#OOO#\\" +\
    "###O#\\" +\
    "#OOO#\\" +\
    "#####"

U_MAZE_EVAL = \
    "#####\\" +\
    "#OOO#\\" +\
    "#O###\\" +\
    "#OOO#\\" +\
    "#####"

OPEN = \
    "#######\\" +\
    "#OOOOO#\\" +\
    "#OOOOO#\\" +\
    "#OOOOO#\\" +\
    "#######"

chosen_maze = U_MAZE


def display_maze(maze):
    maze_str = []
    for row in maze:
        str_row = ''.join(row)
        maze_str.append(str(str_row))
    return '\\'.join(maze_str)


def break_some_walls(maze, width, height, factor=0.8):
    coords_mod = []

    for (i, j) in product(range(1, width-1), range(1, height-1)):

        if maze[i][j] == '#':
            vertical_corridor = maze[i][j+1] == 'O' and maze[i][j -
                                                                1] == 'O' and maze[i+1][j] == '#' and maze[i-1][j] == '#'
            horizontal_corridor = maze[i][j+1] == '#' and maze[i][j -
                                                                  1] == '#' and maze[i+1][j] == 'O' and maze[i-1][j] == 'O'
            random_dice = random.random() > factor

            if (vertical_corridor or horizontal_corridor) and random_dice:
                coords_mod.append((i, j))

    for (i, j) in coords_mod:
        maze[i][j] = 'O'


def append_wall(matrix):
    # Append a row of zeros

    matrix.reverse()

    for row in matrix:
        row.reverse()

    # Create a row of zeros with the same number of columns as the original matrix
    row_of_zeros = ['#'] * len(matrix[0])
    matrix.append(row_of_zeros)

    # Append a column of zeros
    for row in matrix:
        row.append('#')  # Add a zero to the end of each row

    return matrix


def generate_maze(width, height, seed=4):

    width = width - 1
    height = height - 1

    random.seed(seed)
    maze = [['#' for _ in range(width)] for _ in range(height)]

    stack = []

    directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]

    start_x = random.randint(0, (width-1)//2) * 2
    start_y = random.randint(0, (height-1)//2) * 2
    maze[start_y][start_x] = 'O'
    stack.append((start_x, start_y))

    while stack:
        x, y = stack[-1]
        random.shuffle(directions)

        has_neighbors = False

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx <= width-1 and 0 <= ny <= height-1 and maze[ny][nx] == '#':

                maze[ny][nx] = 'O'

                if dx == 2:
                    maze[y][x+1] = 'O'
                elif dx == -2:
                    maze[y][x-1] = 'O'
                elif dy == 2:
                    maze[y+1][x] = 'O'
                elif dy == -2:
                    maze[y-1][x] = 'O'

                stack.append((nx, ny))
                has_neighbors = True

        if not has_neighbors:
            stack.pop()

    break_some_walls(maze, width, height)

    append_wall(maze)

    return display_maze(maze)


def draw_and_save_maze(maze_string, filename):
    rows = maze_string.split('\\')
    maze_array = np.array(
        [[0 if cell == 'O' else 1 for cell in row] for row in rows])

    plt.imshow(maze_array, cmap='gray_r', interpolation='nearest')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


for i in range(50):
    maze_string = generate_maze(19, 19, i)
    draw_and_save_maze(maze_string, 'maze'+str(i)+'.png')
