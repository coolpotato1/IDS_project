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
    
def get_attack_column(dataset_type):
    data = csv_read("Datasets/KDD" + dataset_type + "+.txt")
    return [row[-2] for row in data]


def get_attack_types(train_path, test_path):
    test_data = csv_read(test_path)
    train_data = csv_read(train_path)
    
    test_set = set([row[-2] for row in test_data])
    train_set = set([row[-2] for row in train_data])
    
    return train_set, test_set
   
    
def get_specific_scores(datapath, clf, data, actual_class, attack_types, keep_separated = False):
    csv_data = csv_read(datapath)
    attack_indices = []
    return_scores = []
    for attack in attack_types:
        if(keep_separated):
            attack_indices.append([i for i in range(0, len(csv_data)) if csv_data[i][-2] in attack])
        else:
            attack_indices.extend([i for i in range(0, len(csv_data)) if csv_data[i][-2] in attack])
            
    #Check if we have a 1-dimensional or 2 dimensional list
    if(type(attack_indices[0]) == list):
        for index_list in attack_indices:
            return_scores.append(clf.score([data[i] for i in range(0, len(data)) if i in index_list], 
                                            [actual_class[i] for i in range(0, len(actual_class)) if i in index_list]))
            
    else:
        return_scores.append(clf.score([data[i] for i in range(0, len(data)) if i in attack_indices], 
                                            [actual_class[i] for i in range(0, len(actual_class)) if i in attack_indices]))
        
    return return_scores
    
          
    
    
create_filtered_dataset("Datasets/KDDTest+", attacks.R2L.value + attacks.U2R.value)
create_filtered_dataset("Datasets/KDDTrain+", attacks.R2L.value + attacks.U2R.value)
#train_attacks, test_attacks = get_attack_types("Datasets/KDDTrain+.txt", "Datasets/KDDTest+.txt")
#print(train_attacks)
#print(test_attacks)