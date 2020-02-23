# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:20:52 2020

@author: Henning
"""
import feature_selection as feat
import numpy as np
import preprocessing as pre
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate

TRAIN_DATA_PATH = "Datasets/KDDTrain+.arff"
TEST_DATA_PATH = "Datasets/KDDTest+.arff"
x_train, y_train = pre.load_and_process_data(TRAIN_DATA_PATH, 5)
x_test, y_test = pre.load_and_process_data(TEST_DATA_PATH)

x_train = feat.pca(x_train)
x_test = feat.pca(x_test)

clf = RandomForestClassifier()
#results = cross_validate(clf, x_train, y_train, cv=10)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2)
clf.fit(x_train, y_train)
results = clf.score(x_test, y_test)
print(results)