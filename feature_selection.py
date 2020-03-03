# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 18:41:33 2020

@author: Henning
"""
import preprocessing as pre
import numpy as np
from sklearn.decomposition import PCA

def pca(data, n_components = 8, is_test_data = False):
    pca = PCA(n_components)
    pca.fit(data)
    print(np.sum(pca.explained_variance_ratio_))
    return pca.transform(data)