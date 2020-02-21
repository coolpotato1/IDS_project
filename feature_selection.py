# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 18:41:33 2020

@author: Henning
"""
import preprocessing as pre
import numpy as np
from sklearn.decomposition import PCA

def pca(data, n_components = 8):
    pca = PCA(n_components)
    return pca.fit_transform(data)