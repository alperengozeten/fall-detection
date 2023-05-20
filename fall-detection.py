import os
import pandas as pd
import numpy as np
import warnings

from os import path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from utils import plot_2d, min_max_scale, plot_clusters

warnings.filterwarnings("ignore")

# get the current working directory
ROOT_DIR = path.abspath(os.curdir)
DATA_DIR = path.join(ROOT_DIR, 'data')

# read the full dataset
df = pd.read_csv(path.join(DATA_DIR, 'falldetection_dataset.csv'), header=None)

print(df.head())

# get the full data into np array
labels = df.iloc[:, 1].apply(lambda x: 1 if x == 'F' else 0).to_numpy()
full_data = df.iloc[:, 2 :].to_numpy()
full_data = min_max_scale(full_data)

print(full_data.shape)

# fit using PCA from sklearn
pca = PCA()
pca.fit(full_data)
explained_variance_ratios = pca.explained_variance_ / np.sum(pca.explained_variance_)

print('Variance explained by first PC: ' + str(pca.explained_variance_[0]))
print('Variance explained by second PC: ' + str(pca.explained_variance_[1]))

print('Proportion of variance explained by first PC: ' + str(explained_variance_ratios[0]))
print('Proportion of variance explained by second PC: ' + str(explained_variance_ratios[1]))

# transform the data using only the first two components
pca2 = PCA(n_components=2)
transformed_full_data = pca2.fit_transform(full_data)

plot_2d(transformed_full_data, 'Transformed Full Dataset', 'Projection Onto First PC', 'Projection Onto Second PC', labels=['Transformed Dataset'])

kmeans = KMeans(n_clusters=2, random_state=2023)
kmeans.fit(transformed_full_data)

predicted_labels = kmeans.predict(transformed_full_data)

print('\n-------------------------------------')
for k in range(2):
    clusterLabels = labels[predicted_labels == k]

    numFall = np.sum(clusterLabels)
    numNonFall = len(clusterLabels) - numFall

    isFallCluster = 1 if numFall >= numNonFall else 0
    acc = (numFall / len(clusterLabels)) if isFallCluster == 1 else (numNonFall / len(clusterLabels))
    
    clusterType = 'Fall' if isFallCluster == 1 else 'Non-Fall'
    print(f'The Cluster {k + 1} Consists Of Mostly {clusterType} Data')
    print(f'Accuracy When Cluster {k + 1} Predicted As {clusterType}: {acc}')

acc1 = accuracy_score(labels, predicted_labels)
acc2 = accuracy_score(labels, 1- predicted_labels)
acc = max(acc1, acc2)
print('The Overall Accuracy Obtained With 2-Means: ' + str(acc) + '\n-------------------------------------')


plot_clusters(transformed_full_data, predicted_labels, nClusters=2, title='2-Means On Transformed Data', xLabel='Projection Onto First PC', yLabel='Projection Onto Second PC', labels=['Cluster1', 'Cluster2'])

for k in range(3, 11):
    kmeans = KMeans(n_clusters=k, random_state=2023)
    kmeans.fit(transformed_full_data)
    predicted_labels = kmeans.predict(transformed_full_data)

    legendLabels = [f'Cluster{i}' for i in range(1, k + 1)]

    plot_clusters(transformed_full_data, predicted_labels, nClusters=k, title=f'{k}-Means On Transformed Data', xLabel='Projection Onto First PC', yLabel='Projection Onto Second PC', labels=legendLabels)

    print('\n-------------------------------------')
    numCorrectPreds = 0
    for k_ in range(k):
        clusterLabels = labels[predicted_labels == k_]
        
        numFall = np.sum(clusterLabels)
        numNonFall = len(clusterLabels) - numFall

        isFallCluster = 1 if numFall >= numNonFall else 0
        acc = (numFall / len(clusterLabels)) if isFallCluster == 1 else (numNonFall / len(clusterLabels))

        numCorrectPreds += numFall if isFallCluster == 1 else numNonFall
        
        clusterType = 'Fall' if isFallCluster == 1 else 'Non-Fall'
        print(f'The Cluster {k_ + 1} Consists Of Mostly {clusterType} Data')
        print(f'Accuracy When Cluster {k_ + 1} Predicted As {clusterType}: {acc}')

    print(f'Overall Accuracy When Clusters Are Predicted Based On Majority With {k}-Means: {numCorrectPreds / len(labels)}\n-------------------------------------')
        