import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def plot_2d(data: np.ndarray, title='Title', xLabel='x', yLabel='y', labels=['Data']):
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], s=6)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.legend(labels=labels)
    plt.show()


def plot_clusters(data: np.ndarray, predicted_labels: np.ndarray, nClusters, title='Title', xLabel='x', yLabel='y',
                  labels=['Data']):
    plt.figure()
    for i in range(nClusters):
        currentData = data[predicted_labels == i]
        plt.scatter(currentData[:, 0], currentData[:, 1], s=6)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.legend(labels=labels)
    plt.show()


def min_max_scale(data: np.ndarray) -> np.ndarray:
    data_scaler = MinMaxScaler()
    return data_scaler.fit_transform(data)

def accuracy_histogram(accList, model_name):
    # Plot the distribution of the accuracies
    plt.figure()
    plt.title(f'The number of models with a given validation accuracy for {model_name} model')
    plt.xlabel('Accuracy of the model')
    plt.ylabel('Number of models')
    plt.hist(accList)
    plt.show()
