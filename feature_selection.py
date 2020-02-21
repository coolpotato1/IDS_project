# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 18:41:33 2020

@author: Henning
"""
import preprocessing as pre
import numpy as np
from sklearn.decomposition import PCA

DATA_PATH = "Datasets/KDDTrain+_20Percent.arff"
data, predictions = pre.load_and_process_data(DATA_PATH)
pca = PCA(n_components = 8)
reduced_data = pca.fit_transform(data)
print(pca.explained_variance_ratio_)
print(np.sum(pca.explained_variance_ratio_))