import os
import pandas as pd
import numpy as np
import warnings

from os import path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from utils import plot_2d, min_max_scale, plot_clusters
from itertools import product

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

'''
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
'''

# transform the data using the first 40 PCs
pcaNew = PCA(n_components=40)
transformed_full_data = pcaNew.fit_transform(full_data)
print('The total explained variance when first 40 PCs used: ' + str(np.sum(pcaNew.explained_variance_ratio_))) # check the total explained variance

# split the data into train, validation, and test datasets using 70-15-15 distribution
x_train, x_test, y_train, y_test = train_test_split(transformed_full_data, labels, test_size=0.30, random_state=2023)
x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.50, random_state=2023)

'''
------------ SVM Model ------------ 
'''
# set of possible values to try out
c_values        = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
kernel_types    = ["poly", "rbf", "sigmoid"]
gamma_values    = ["scale", "auto"]
degrees         = [1, 2, 3, 4, 5, 6]

svm_hyperparams = list(product(c_values, kernel_types, gamma_values, degrees))

# try out all possible configurations and append the results
results = []
for c, kernel_type, gamma, degree in svm_hyperparams:
    if kernel_type == "poly":
        print(f"\nRunning c={c}, kernel_type={kernel_type}, gamma={gamma}, degree={degree}")
        model = SVC(C=c, kernel=kernel_type, degree=degree, gamma=gamma, max_iter=100000, random_state=2023)
    else:
        degree = 'NULL'
        print(f"\nRunning c={c}, kernel_type={kernel_type}, gamma={gamma}, degree={degree}")
        model = SVC(C=c, kernel=kernel_type, gamma=gamma, max_iter=100000, random_state=2023)
    
    model.fit(x_train, y_train)
    predictions = model.predict(x_valid)
    val_accuracy = accuracy_score(y_valid, predictions) * 100
    result = {'Regularization Parameter': c, 'Kernel Type': kernel_type, 'Degree': degree, 'Kernel Coefficient': gamma, 'Validation Accuracy (%)': val_accuracy}
    print(result)

    results.append(val_accuracy)

# get the index with highest accuracy score
results = np.asarray(results)
svm_best_index = np.argmax(results)

# get the best set of parameters obtained
svm_best_params = svm_hyperparams[svm_best_index]
svm_best_c = svm_best_params[0]
svm_best_kernel = svm_best_params[1]
svm_best_gamma = svm_best_params[2]
svm_best_degree = svm_best_params[3]
svm_best_val_acc = results[svm_best_index]

# create new datasets to train the best model on
x_train_valid = np.concatenate((x_train, x_valid), axis=0)
y_train_valid = np.concatenate((y_train, y_valid), axis=0)

# if the best kernel type is not polynomial, change the degree to NULL (unimportant)
svm_best_degree = svm_best_degree if svm_best_kernel == 'poly' else 'NULL'

# output the best settings obtained
print(f'\n------------------------\nThe settings of the best SVM model: c={svm_best_c}, kernel_type={svm_best_kernel}, gamma={svm_best_gamma}, degree={svm_best_degree}')

# create instance differently depending on the kernel type
if svm_best_kernel == "poly":
    svm_best_model = SVC(C=svm_best_c, kernel=svm_best_kernel, degree=svm_best_degree,
                     gamma=svm_best_gamma, max_iter=10000, random_state=2023)
else:
    svm_best_model = SVC(C=svm_best_c, kernel=svm_best_kernel, gamma=svm_best_gamma, max_iter=10000, random_state=2023)

# train the best model on train + validation dataset
svm_best_model.fit(x_train_valid, y_train_valid)

# output the obtained accuries with the best model
svm_predictions = svm_best_model.predict(x_test)
svm_acc = 100 * accuracy_score(y_test, svm_predictions)
print(f"Accuracy of the best SVM model trained on (train + val) and tested on test data = {svm_acc} %\n")

'''
------------ MLP Model ------------ 
'''
learning_rates = [1e-2, 5e-2, 1e-3, 5e-4, 1e-4]
alphas = [1, 1e-1, 1e-2, 1e-3]
hidden_layer_sizes = [(8, 8), (16, 16), (32, 32), (64, 64)]
solvers = ["adam", "sgd"]
activation_functions = ["relu"]

nn_hyperparams = list(product(hidden_layer_sizes, learning_rates, alphas, solvers, activation_functions))

'''
results = []
for size, lr, alpha, solver, activation_function in nn_hyperparams:
    print(f"\nRunning size={size}, lr={lr}, alpha={alpha}, solver={solver}, act_func={activation_function}")
    model = MLPClassifier(hidden_layer_sizes=size, activation=activation_function, solver=solver, alpha=alpha, learning_rate_init=lr, max_iter=100000,
                          random_state=2023)
    model.fit(x_train, y_train)
    predictions = model.predict(x_valid)
    val_accuracy = accuracy_score(y_valid, predictions) * 100
    row = {"Hidden Layer Size": size,
            "Activation Function": activation_function,
            "Solver": solver,
            "Alpha": alpha,
            "Learning Rate": lr,
            "Validation Accuracy (%)": val_accuracy}
    results.append(row)
    print(row)

results_sorted = sorted(results, key=lambda x: x['Validation Accuracy (%)'], reverse=True)
'''