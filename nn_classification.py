# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:20:52 2020

@author: Henning
"""
import feature_selection as feat
import numpy as np
import preprocessing as pre
import dataset_manipulation as man
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from NSL_KDD_attack_types import attack_types as attacks
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
import classification_utils as cl
import time
N_COMPONENTS = 25
TRAIN_DATA_PATH = "Datasets/combinedCoojas.arff"
TEST_DATASET = "coojaData3"
TEST_DATA_PATH = "Datasets/" + TEST_DATASET + ".arff"
CSV_DATA_PATH = "Datasets/KDDTest+.txt"

def NN_train(data, predictions):
    model = Sequential()
    model.add(Dense(25, input_dim = len(data[0]), activation="relu"))
    model.add(Dense(12, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    adam = optimizers.adam(lr=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.fit(data, predictions, batch_size = 10, epochs = 30)
    return model

x_train, y_train, attributes = pre.load_and_process_data(TRAIN_DATA_PATH, do_normalize=True)
actual_classes =np.squeeze(man.csv_read("Datasets/" + TEST_DATASET + "_attacks"))
x_test, y_test, test_attributes = pre.load_and_process_data(TEST_DATA_PATH, do_normalize=True)

# This check is done to ensure that the columns of the test and train datasets are in the same order, cause if not
# it will ruin the entire classification
#if attributes != test_attributes:
    #print("your datasets are fucked")
    #sys.exit()
#x_test = np.asarray(x_test).astype(np.float32)

x_train = np.asarray(x_train).astype(np.float32)

#x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

print("undersampling overrepresented class")
#x_train, y_train = cl.sample(x_train, y_train)

print("length of the training set is: ", len(x_train))
print("length of test set is: ", len(x_test))
print("Getting to PCA")
#pca = PCA(N_COMPONENTS)
#pca.fit(x_train)
#print("The variance explained is: ", str(np.sum(pca.explained_variance_ratio_)))
#x_train = pca.transform(x_train)
#x_test = pca.transform(x_test)


#clf = RandomForestClassifier(n_estimators = 150)


#print("Doing RFE")
#rfe = RFE(clf, 20, 4)
#x_train = rfe.fit_transform(x_train, y_train)
#x_test = rfe.transform(x_test)
#print("Finished RFE")
#print("Fitting classifier")
#clf.fit(x_train, y_train)

#pre_train = time.time()
clf = NN_train(x_train, y_train)
#post_train = time.time()

#For the NN model, first value is loss
precision, recall, fscore, accuracy = man.get_normal_and_anomaly_scores(clf, x_test, y_test)

#results = clf.score(x_test, y_test)
print("Precision is: ", precision)
print("Recall is: ", recall)
print("Fscore is: ", fscore)
print("Overall accuracy is: ", accuracy)
#print("normal results are:", results)
print("specific recalls are:", man.get_specific_recall(clf, x_test, actual_classes,
                                                       [attacks.UDP_NORMAL.value, attacks.UDP_DOS.value], keep_separated=True))