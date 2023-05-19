import os
import pandas as pd
import numpy as np
from os import path
from sklearn.decomposition import PCA
from utils import plot_2d, min_max_scale

# get the current working directory
ROOT_DIR = path.abspath(os.curdir)
DATA_DIR = path.join(ROOT_DIR, 'data')

# read the full dataset
df = pd.read_csv(path.join(DATA_DIR, 'falldetection_dataset.csv'), header=None)

print(df.head())

# get the full data into np array
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
print(transformed_full_data.shape)
print(transformed_full_data)
