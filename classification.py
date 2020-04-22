# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:20:52 2020

@author: Henning
"""
import feature_selection as feat
import numpy as np
import preprocessing as pre
import dataset_manipulation as man
from keras.models import Sequential
from keras.layers import Dense
from NSL_KDD_attack_types import attack_types as attacks
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE 
import time
N_COMPONENTS = 15
TRAIN_DATA_PATH = "Datasets/svelteSinkhole12.arff"
TEST_DATA_PATH = "Datasets/svelteSinkhole3.arff"
CSV_DATA_PATH = "Datasets/KDDTest+.txt"

def NN_train(data, predictions):
    model = Sequential()
    model.add(Dense(20, input_dim = len(data[0]), activation="relu"))
    model.add(Dense(5, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(data, predictions, batch_size = 20, epochs = 500)
    return model

x_train, y_train, attributes = pre.load_and_process_data(TRAIN_DATA_PATH, do_normalize = True)

start_load = time.time()
x_test, y_test, attributes = pre.load_and_process_data(TEST_DATA_PATH, is_test_data = True, attributes = attributes, do_normalize= True)
x_test = np.asarray(x_test).astype(np.float32)
end_load = time.time()

x_train = np.asarray(x_train).astype(np.float32)


print("Getting to PCA")
#pca = PCA(N_COMPONENTS)
#pca.fit(x_train)
#print("The variance explained is: ", str(np.sum(pca.explained_variance_ratio_)))
#
#x_train = pca.transform(x_train)
#x_test = pca.transform(x_test)


#clf = RandomForestClassifier(n_estimators = 150)
#clf = SVC()
#results = cross_validate(clf, x_train, y_train, cv=10)

#print("Doing RFE")
#rfe = RFE(clf, 16, 2)
#x_train = rfe.fit_transform(x_train, y_train)
#x_test = rfe.transform(x_test)
#print("Finished RFE")
print("Fitting classifier")
#clf.fit(x_train, y_train)

pre_train = time.time()
clf = NN_train(x_train, y_train)
post_train = time.time()

#For the NN model, first value is loss
normal_score, anomaly_score = man.get_normal_and_anomaly_scores(clf, x_test, y_test)

#results = clf.score(x_test, y_test)
print("Normal accuracy is: ", normal_score)
print("Anomaly accuracy is: ", anomaly_score)
print("Time for training in seconds is:", post_train - pre_train)
#print("normal results are:", results)
#print("specific results are:", man.get_specific_scores(CSV_DATA_PATH, clf, x_test, y_test, [attacks.NORMAL.value, attacks.NEPTUNE.value], True))