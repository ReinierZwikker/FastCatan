import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import os.path


colors = ["#00A6D6", "#A50034", "#009B77", "#FFB81C", "#000000"]
max_gen = 0


def load_dataframes(log_folder: str, folder_id: int) -> list:
    dfs = []
    reading_logs = True
    current_log_id = 0
    while reading_logs:
        file_name = f"../../logs/{log_folder}/{folder_id}/game_{current_log_id}.csv"
        if os.path.isfile(file_name):
            df = pd.read_csv(file_name, skiprows=2)
            dfs.append(df[df[" Score"] > -500])
        else:
            reading_logs = False

        current_log_id += 1

    return dfs


def load_multiple_dataframes(log_folder: str) -> list[list]:
    dfs_list = []
    reading_folders = True
    current_folder_id = 0
    while reading_folders:
        path_name = f"../../logs/{log_folder}/{current_folder_id}"
        if os.path.isdir(path_name):
            dfs_list.append(load_dataframes(log_folder, current_folder_id))
        else:
            reading_folders = False

        current_folder_id += 1

    return dfs_list


def draw_average_plot(dataframe_folders: list[list], variable: str):
    global max_gen
    for i, dataframes in enumerate(dataframe_folders):
        max_values = []
        averages = []
        generations = []
        for j, dataframe in enumerate(dataframes):
            generations.append(j)
            array = dataframe[variable].array
            averages.append(np.mean(array[np.nonzero(array)]))
            max_values.append(np.max(array[np.nonzero(array)]))

        plt.plot(generations, averages, color=colors[i], label=f'Mean Run {i}')
        plt.plot(generations, max_values, color=colors[i], label=f'Max Run {i}', linestyle="--")

        if generations[-1] > max_gen:
            max_gen = generations[-1]


var = "Score"
file = "Baseline"

loaded_data = load_multiple_dataframes(file)
draw_average_plot(loaded_data, f" {var}")

plt.grid()
plt.xlim((0, None))
plt.ylim((None, None))
plt.xticks(range(0, max_gen + 1))

plt.xlabel("Generation")
plt.ylabel(var)
plt.legend(fancybox=True)

plt.tight_layout()
plt.show()
# plt.savefig(f"figures/{var.replace(' ', '')}{file}.svg")

