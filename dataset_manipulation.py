# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:19:56 2020

@author: Henning
"""
import csv
import arff
import numpy as np
from NSL_KDD_attack_types import attack_types as attacks

def get_filter_indices(filtered_attacks, data_path):
    with open(data_path) as file:
        data = csv.reader(file)
        return_indexes = []
        i = 0
        for row in data:
            if(row[-2] in filtered_attacks):
                return_indexes.append(i)
            i += 1
                
    return return_indexes


def remove_attacks(data, filter_indices):
    return [data[i] for i in range(len(data)) if i not in filter_indices]


def create_filtered_dataset(file_name, filtered_attacks):
    file = arff.load(open(file_name + ".arff"))
    original_data = file['data']
    attributes = file['attributes']
    
    new_data = remove_attacks(original_data, get_filter_indices(filtered_attacks, file_name + ".txt"))
    return_arff = {
            'relation': 'KDDFiltered',
            'description': '',
            'data': new_data,
            'attributes': attributes
            }
    arff.dump(return_arff, open(file_name + "_filtered.arff", "w+"))
    
def csv_read(datapath):
    with open(datapath) as file:
        data = csv.reader(file)
        return [row for row in data]


def get_attack_types(train_path, test_path):
    test_data = csv_read(test_path)
    train_data = csv_read(train_path)
    
    test_set = set([row[-2] for row in test_data])
    train_set = set([row[-2] for row in train_data])
    
    return train_set, test_set
    
create_filtered_dataset("Datasets/KDDTest+", attacks.R2L.value + attacks.U2R.value)
create_filtered_dataset("Datasets/KDDTrain+", attacks.R2L.value + attacks.U2R.value)
#train_attacks, test_attacks = get_attack_types("Datasets/KDDTrain+.txt", "Datasets/KDDTest+.txt")
#print(train_attacks)
#print(test_attacks)