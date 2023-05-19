import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def plot_2d(data : np.ndarray, title='Title', xLabel='x', yLabel='y', labels=['Data']):
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], s=6)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.legend(labels=labels)
    plt.show()

def min_max_scale(data : np.ndarray) -> np.ndarray:
    data_scaler = MinMaxScaler()
    return data_scaler.fit_transform(data)