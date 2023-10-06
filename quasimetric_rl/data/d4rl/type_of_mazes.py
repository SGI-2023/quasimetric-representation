from __future__ import annotations
from typing import *
import random
import numpy as np
import matplotlib.pyplot as plt


LAROE_MAZE = \
        "############\\"+\
        "#OOOO#OOOOO#\\"+\
        "#O##O#O#O#O#\\"+\
        "#OOOOOO#OOO#\\"+\
        "#O####O###O#\\"+\
        "#OO#O#OOOOO#\\"+\
        "##O#O#O#O###\\"+\
        "#OO#OOO#OOO#\\"+\
        "############"

LAROE_MAZE_EVAL = \
        "############\\"+\
        "#OO#OOO#OOO#\\"+\
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
        '#OOO#OO#\\'+\
        "########"

MEDIUM_MAZE_EVAL = \
        '########\\'+\
        '#OOOOOO#\\'+\
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
        "#OOO#\\"+\
        "###O#\\"+\
        "#OOO#\\"+\
        "#####"

U_MAZE_EVAL = \
        "#####\\"+\
        "#OOO#\\"+\
        "#O###\\"+\
        "#OOO#\\"+\
        "#####"

OPEN = \
        "#######\\"+\
        "#OOOOO#\\"+\
        "#OOOOO#\\"+\
        "#OOOOO#\\"+\
        "#######"

chosen_maze = U_MAZE



def generate_maze(width, height, seed = 4):

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
        neighbors = []
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx <= width-1 and 0 <= ny <= height-1 and maze[ny][nx] == '#':
                neighbors.append((nx, ny, dx, dy))

        if neighbors:
            nx, ny, dx, dy = random.choice(neighbors)
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
        else:
            stack.pop()

    def display_maze(maze):
        maze_str = []
        for row in maze:
            str_row = ''.join(row)
            print(str_row)
            maze_str.append(str(str_row))
        return '\\'.join(maze_str)

    return display_maze(maze)

def draw_and_save_maze(maze_string, filename):
    rows = maze_string.split('\\')
    maze_array = np.array([[0 if cell == 'O' else 1 for cell in row] for row in rows])
    
    plt.imshow(maze_array, cmap='gray_r', interpolation='nearest')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == "__main__":

	width = 20
	height = 20

	for i in range(50):
		maze_str = generate_maze(width, height, seed=i)
		draw_and_save_maze(maze_str, f"maze_{i+1}.png")
