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
import tensorflow as tf
from tensorflow import keras
from NSL_KDD_attack_types import attack_types as attacks
import tensorflow_model_optimization as tfmot
from tensorflow.keras import activations
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
import classification_utils as cl
import time

N_COMPONENTS = 40
TRAIN_DATA_PATH = "Datasets/svelteSinkhole12combinedCoojasMitMTrainKDDTrain+_filtered.arff"
TEST_DATASET = "svelteSinkhole3coojaData4MitMTestKDDTest+_filtered"
TEST_DATA_PATH = "Datasets/" + TEST_DATASET + ".arff"
CSV_DATA_PATH = "Datasets/KDDTest+.txt"


def NN_train(data, predictions):
    model = keras.Sequential(
    [keras.layers.Dense(400, activation="relu", input_dim=len(data[0])),
     #tf.keras.layers.Activation(activations.relu),
     keras.layers.Dropout(0.5),
     keras.layers.Dense(80, activation="relu"),
     #tf.keras.layers.Activation(activations.relu),
     keras.layers.Dropout(0.5),
     keras.layers.Dense(1, activation='sigmoid')])
     #tf.keras.layers.Activation(activations.relu)])
    adam = tf.keras.optimizers.Adam(lr=0.02)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.fit(data, predictions, batch_size=200, epochs=35)
    return model

def get_quantized_model(NN):
    q_aware_model = tfmot.quantization.keras.quantize_model(NN)
    q_aware_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return q_aware_model


x_train, y_train, attributes = pre.load_and_process_data(TRAIN_DATA_PATH, do_normalize=True, export_configuration=True)
actual_classes = np.squeeze(man.csv_read("Datasets/" + TEST_DATASET + "_attacks"))
x_test, y_test, test_attributes = pre.load_and_process_data(TEST_DATA_PATH, do_normalize=True)
man.csv_write(y_test, "test_dataset_labels")

# This check is done to ensure that the columns of the test and train datasets are in the same order, cause if not
# it will ruin the entire classification
if attributes != test_attributes:
    print("your datasets are fucked")
    sys.exit()
# x_test = np.asarray(x_test).astype(np.float32)

x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

print("undersampling overrepresented class")
# x_train, y_train = cl.sample(x_train, y_train, sampling_type="over")

print("length of the training set is: ", len(x_train))
print("length of test set is: ", len(x_test))
#print("Getting to PCA")
#pca = PCA(N_COMPONENTS)
#pca.fit(x_train)
#print("The variance explained is: ", str(np.sum(pca.explained_variance_ratio_)))
#x_train = pca.transform(x_train)
#x_test = pca.transform(x_test)

# clf = RandomForestClassifier(n_estimators = 150)


# print("Doing RFE")
# rfe = RFE(clf, 20, 4)
# x_train = rfe.fit_transform(x_train, y_train)
# x_test = rfe.transform(x_test)
# print("Finished RFE")
# print("Fitting classifier")
# clf.fit(x_train, y_train)

# pre_train = time.time()
clf = NN_train(x_train, y_train)
q_clf = get_quantized_model(clf)
#clf.save("models/full_dataset_nn")
precision, recall, fscore, accuracy = man.get_normal_and_anomaly_scores(clf, x_test, y_test)
q_precision, q_recall, q_fscore, q_accuracy = man.get_normal_and_anomaly_scores(q_clf, x_test, y_test)
# results = clf.score(x_test, y_test)
print("Precision is: ", precision)
print("Recall is: ", recall)
print("Fscore is: ", fscore)
print("Overall accuracy is: ", accuracy)
print("q_precision is: ", q_precision)
print("q_recall is: ", q_recall)
print("q_fscore is: ", q_fscore)
print("q_accuracy is: ", q_accuracy)
# print("normal results are:", results)
# print(man.get_specific_recall(clf, x_test, actual_classes, [attacks.UDP_NORMAL.value, attacks.SINKHOLE_NORMAL.value,
# attacks.UDP_DOS.value, attacks.SINKHOLE.value], keep_separated=True))

print(man.get_specific_recall(clf, x_test, actual_classes, [attacks.NORMAL.value, attacks.MITM_NORMAL.value,
                                                            attacks.UDP_NORMAL.value, attacks.SINKHOLE_NORMAL.value,
                                                            attacks.DOS.value, attacks.PROBE.value, attacks.MITM.value,
                                                            attacks.UDP_DOS.value, attacks.SINKHOLE.value],
                              keep_separated=True))
