import matplotlib
import matplotlib.pyplot as plt
import csv

import numpy as np

plt.figure(figsize=(10, 6))

# === TRAINING 1 ===

filename = "training_until_100000_1.csv"

with open(f"../../cmake-build-debug/trainings/{filename}") as file:
    train_data = list(csv.reader(file))[0]

train_data = [float(x) for x in train_data]

font = {'size': 15}

matplotlib.rc('font', **font)

plt.plot(np.arange(0, len(train_data)*180, 180), train_data, color="#00A6D6", label="Training 1")


# === TRAINING 2 ===

filename = "training_big_until_12500.csv"

with open(f"../../cmake-build-debug/trainings/{filename}") as file:
    train_data = list(csv.reader(file))[0]

train_data = [float(x) for x in train_data]

font = {'size': 15}

matplotlib.rc('font', **font)

plt.plot(np.arange(0, len(train_data)*300, 300), train_data, color="000000", label="Training 2")


# === TRAINING 3 ===

filename = "training_mid_until_100000.csv"

with open(f"../../cmake-build-debug/trainings/{filename}") as file:
    train_data = list(csv.reader(file))[0]

train_data = [float(x) for x in train_data]

font = {'size': 15}

matplotlib.rc('font', **font)

plt.plot(np.arange(0, len(train_data)*300, 300), train_data, color="#A50034", label="Training 3")


# === PLOTTING ===

plt.title("Training Result with Epoch Length of 15")
plt.xlabel("Games")
plt.ylabel("Average Score")
plt.grid()
plt.legend()
plt.show()

# === TRAINING 4 ===

plt.figure(figsize=(10, 6))

filename = "training_new_200000.csv"

with open(f"../../cmake-build-debug/trainings/{filename}") as file:
    train_data = list(csv.reader(file))[0]

train_data = [float(x) for x in train_data]

font = {'size': 15}

matplotlib.rc('font', **font)

plt.plot(np.arange(0, len(train_data)*300, 300), train_data, color="#00A6D6", label="Training 4")


# === PLOTTING ===

plt.title("Training Result with new Score Calculation")
plt.xlabel("Games")
plt.ylabel("Average Score")
plt.grid()
plt.legend()
plt.show()
