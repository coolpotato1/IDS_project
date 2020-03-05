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
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE 
import time
N_COMPONENTS = 15
TRAIN_DATA_PATH = "Datasets/KDDTrain+_filtered.arff"
TEST_DATA_PATH = "Datasets/KDDTest+_filtered.arff"
x_train, y_train, attributes = pre.load_and_process_data(TRAIN_DATA_PATH, do_normalize = True)

start_load = time.time()
x_test, y_test, attributes = pre.load_and_process_data(TEST_DATA_PATH, is_test_data = True, attributes = attributes, do_normalize= True)
x_test = np.asarray(x_test).astype(np.float32)
end_load = time.time()

x_train = np.asarray(x_train).astype(np.float32)


print("Getting to PCA")
pca = PCA(N_COMPONENTS)
pca.fit(x_train)
print("The variance explained is: ", str(np.sum(pca.explained_variance_ratio_)))
#
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)


clf = RandomForestClassifier(n_estimators = 150)
#clf = SVC()
#results = cross_validate(clf, x_train, y_train, cv=10)
#x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2)

#print("Doing RFE")
#rfe = RFE(clf, 16, 2)
#x_train = rfe.fit_transform(x_train, y_train)
#x_test = rfe.transform(x_test)
#print("Finished RFE")
print("Fitting classifier")
clf.fit(x_train, y_train)
results = clf.score(x_test, y_test)

print(results)