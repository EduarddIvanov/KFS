import numpy as np
import matplotlib.pyplot as plt
import os
import random
from matplotlib import colors
from matplotlib.colors import ListedColormap

TREE = 0
BURNING = 1
EMPTY = 2

P_BURN = 0.3
T_BURN = 4
GRID_SIZE = 60
INITIAL_FIRE_PROB = 0.02
STEPS = 50
OUTPUT_DIR = "forest_fire_frames"

def initialize_forest(size, initial_fire_prob):
    forest = np.zeros((size, size), dtype=int)
    forest[:, :] = TREE

    for i in range(size):
        for j in range(size):
            if random.random() < initial_fire_prob:
                forest[i, j] = BURNING
    return forest

def update_forest(forest, burning_times):
    new_forest = forest.copy()
    new_burning_times = burning_times.copy()
    size = forest.shape[0]

    for i in range(size):
        for j in range(size):
            if forest[i, j] == BURNING:
                new_burning_times[i, j] -= 1
                if new_burning_times[i, j] <= 0:
                    new_forest[i, j] = EMPTY
            elif forest[i, j] == TREE:
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < size and 0 <= nj < size:
                        if forest[ni, nj] == BURNING and random.random() < P_BURN:
                            new_forest[i, j] = BURNING
                            new_burning_times[i, j] = T_BURN
                            break
    return new_forest, new_burning_times

def save_frame(forest, step, output_dir):
    plt.figure(figsize=(10, 10))

    cmap = ListedColormap(['green', 'red', 'black'])
    bounds = [0, 1, 2, 3]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(forest, cmap=cmap, norm=norm)
    plt.title(f"Лісова пожежа - Крок {step}")
    plt.colorbar(ticks=[0.5, 1.5, 2.5], label='Стан: 0=Дерево, 1=Горить, 2=Попіл')

    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, f"frame_{step:03d}.png")
    plt.savefig(filename, bbox_inches='tight', dpi=100)
    plt.close()

def simulate_and_save_frames():
    forest = initialize_forest(GRID_SIZE, INITIAL_FIRE_PROB)
    burning_times = np.zeros_like(forest)
    burning_times[forest == BURNING] = T_BURN

    save_frame(forest, 0, OUTPUT_DIR)

    for step in range(1, STEPS + 1):
        forest, burning_times = update_forest(forest, burning_times)
        save_frame(forest, step, OUTPUT_DIR)
        print(f"Ітерація {step}/{STEPS}")

if __name__ == "__main__":
    print("Початок симуляції лісової пожежі")
    simulate_and_save_frames()
    print(f"Симуляція завершена")
