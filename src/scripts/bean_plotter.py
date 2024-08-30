import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import os.path


colors = ["#00A6D6", "#A50034", "#00B8C8", "#0076C2", "#000000"]


def load_dataframes(log_folder: str, folder_id: int) -> list:
    dfs = []
    reading_logs = True
    current_log_id = 0
    while reading_logs:
        file_name = f"../../logs/{log_folder}/{folder_id}/game_{current_log_id}.csv"
        if os.path.isfile(file_name):
            dfs.append(pd.read_csv(file_name, skiprows=2))
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
    for dataframes in dataframe_folders:
        for dataframe in dataframes:



dataframes = load_dataframes("Baseline", 0)

generations = np.arange(0, 20)
average_rounds = np.zeros((20, 100))
for i in range(0, 20):
    average_rounds[i] = dataframes[i][" Average Rounds"].array

average_rounds[average_rounds == 0] = np.nan

for i in range(0, 4):
    if i == 3:
        min_line = np.nanmin(average_rounds, axis=1)
        max_line = np.nanmax(average_rounds, axis=1)
    else:
        min_line = 0
        max_line = 0

    plt.fill_between(generations, min_line, max_line,
                     color=colors[0], alpha=0.4, label='Range (min-max)')

plt.plot(generations, np.nanmean(average_rounds, axis=1), color=colors[0], label='Mean')

# plt.ylim((0, 200))
plt.show()
