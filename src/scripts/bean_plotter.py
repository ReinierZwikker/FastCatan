import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

colors = ["#00A6D6", "#A50034", "#00B8C8", "#0076C2", "#000000"]

dataframes = []
for i in range(0, 20):
    dataframe = pd.read_csv(f'../../logs/game_{i}.csv', skiprows=2)
    # dataframe = dataframe[dataframe[" Games Played"] != 0]
    # dataframe = dataframe[dataframe[" Score"] > -500]
    dataframes.append(dataframe.sort_values(by=" Average Rounds"))


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
