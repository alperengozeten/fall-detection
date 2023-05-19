import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def plot_2d(data : np.ndarray):
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], s=6)
    plt.show()

def min_max_scale(data : np.ndarray) -> np.ndarray:
    data_scaler = MinMaxScaler()
    return data_scaler.fit_transform(data)