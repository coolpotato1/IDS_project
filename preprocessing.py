# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 11:28:39 2020

@author: Henning    
"""

import arff
import gc
import sys
import numpy as np
from itertools import islice

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

def normalize(values, predictions):
    temp_values = np.asarray(values).astype(float)
    temp_predictions = np.asarray(values).astype(float)
    return normalize(np.concatenate(temp_values, temp_predictions, axis = 1))

def process_data(data_values, data_attributes):
    for i in range(len(data_attributes), 1, -1):
        if(type(data_attributes[-i][1]) == list):
            data_values = one_hot_insert(data_values, -i)
    
    #Assign the class values
    predictions = []
    for row in data_values:
        predictions.append(int(row.pop(-1) == "anomaly"))
       
    
    return data_values, predictions

def load_and_process_data(datapath, n_components):
    values = []
    predictions = []
    for i in range(0,n_components):
        file = arff.load(open(datapath.split(".")[0] + str(i+1) + "." + datapath.split(".")[1]))
        temp_data_values = file['data']
        attributes = file['attributes']
        temp_values, temp_predictions = process_data(temp_data_values, attributes)
        values.extend(temp_values)
        predictions.extend(temp_predictions)
        
    return normalize(values, predictions)

def split_dataset(datapath, n_sections):
    file = arff.load(open(datapath))
    data_values = file['data']
    
    iterator = iter(data_values)
    split_size = int(len(data_values)/n_sections)
    splits = [split_size for i in range(1, n_sections)]
    splits.append(len(data_values) - split_size * (n_sections-1))
    split_values = [list(islice(iterator, elem)) for elem in splits]    
    
    for i in range(0, n_sections):
        new_file_data = {
            'relation': 'KDDTrain',
            'description': '',
            'data': split_values[i],
            'attributes': file['attributes']}
        new_file = open(datapath.split(".")[0] + str(i+1) + "." + datapath.split(".")[1], "w+")
        arff.dump(new_file_data, new_file)


#split_dataset("Datasets/KDDTrain+.arff", 5)
    