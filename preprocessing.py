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

def process_data(data_values, data_attributes):
    for i in range(len(data_attributes), 1, -1):
        if(type(data_attributes[-i][1]) == list):
            data_values = one_hot_insert(data_values, -i)
    
    #Assign the class values
    for row in data_values:
        row[-1] = int(row[-1] == "abnormal")
        
    return data_values


def load_and_process_data(datapath):
    file = arff.load(open(datapath))
    data_values = file['data']
    attributes = file['attributes']
    return process_data(data_values, attributes)

            
            
            
            




    