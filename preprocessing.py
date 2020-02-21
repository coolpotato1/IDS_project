# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 11:28:39 2020

@author: Henning    
"""

import arff
import gc
import sys
import numpy as np

def one_hot_encode(column):
    uniqueValues = set(column)
    one_hot_row = []
    for value in column:
        one_hot_row.append([int(value == unique) for unique in uniqueValues])
        
    return one_hot_row


def one_hot_insert(data, column_no):
    one_hot_values = one_hot_encode([row[column_no] for row in data])
    #If you run into memory issues, rewrite to use list methods instead of numpy
    left, delete, right = np.split(data, [column_no, column_no+1], axis = 1)
    data_return = np.concatenate((left, one_hot_values, right), axis = 1)
    
    return data_return.tolist()

def normalize(data):
    return_data = np.asarray(data).astype(float)
    max_values = np.amax(return_data, axis = 0)
    min_values = np.amin(return_data, axis = 0)
    difference = max_values - min_values
    
    #Remove columns where values are constant, done on min and diff arrays too, so dimensions fit for normalization.
    useless = [x for x in range(0, len(difference)) if difference[x] == 0]
    min_values = np.delete(min_values, useless)
    difference = np.delete(difference, useless)
    return_data = np.delete(return_data, useless, axis = 1)
    
    return_data = (return_data - min_values) / difference
    return return_data

def process_data(data_values, data_attributes):
    for i in range(len(data_attributes), 1, -1):
        if(type(data_attributes[-i][1]) == list):
            data_values = one_hot_insert(data_values, -i)
    
    #Assign the class values
    predictions = []
    for row in data_values:
        predictions.append(int(row.pop(-1) == "anomaly"))
       
    data_values = normalize(data_values)
    return data_values, predictions

def load_and_process_data(datapath):
    file = arff.load(open(datapath))
    data_values = file['data']
    attributes = file['attributes']
    return process_data(data_values, attributes)


    
            




    