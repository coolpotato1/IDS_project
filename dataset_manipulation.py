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
   
#This method definitely needs to be refactored, holy fuck its ugly. This has probably added +5 hours to technical debt.
def combine_datasets(dataset1, dataset2):
    file1 = arff.load(open("Datasets/" + dataset1 + ".arff"))
    file2 = arff.load(open("Datasets/" + dataset2 + ".arff"))
    unique_columns = [i for i in range(len(file1["attributes"])) if file1["attributes"][i][0] not in [attribute[0]
                            for attribute in file2["attributes"]]]

    #needed in case any of the non-unique columns have values that do not exist in the other dataset
    non_unique_columns = [i for i in range(len(file1["attributes"])) if i not in unique_columns]
    for column in non_unique_columns:
        index = [i for i in range(len(file2["attributes"])) if file2["attributes"][i][0] == file1["attributes"][column][0]][0]
        #This should only happen if the compared attributes contain lists
        if file1["attributes"][column][1] != file2["attributes"][index][1]:
            file2["attributes"][index][1].extend(list(set(file1["attributes"][column][1]) - set(file2["attributes"][index][1])))

    #Also add the unique attributes
    file2["attributes"].extend([file1["attributes"][column_no] for column_no in unique_columns])

    #Match dimensions of second dataset, by padding with zeroes
    for row in file2["data"]:
        row.extend([0] * len(unique_columns))

    #Match dimensions for the first dataset and ensure that common attributes are placed in correct columns
    new_data1 = []
    new_column_indexes = []
    for i in range(len(non_unique_columns)):
        new_column_indexes.extend([j for j in range(len(file2["attributes"])) if file2["attributes"][j][0] == file1["attributes"][non_unique_columns[i]][0]])

    for i in range(len(file1["data"])):
        #Create zeroes for all of file2's attributes, and then add file 1's. Its so dimensions match
        new_data1.append([0] * (len(file2["attributes"]) - len(unique_columns)) + [file1["data"][i][j] for j in unique_columns])
        for j in range(len(new_column_indexes)):
            new_data1[i][new_column_indexes[j]] = file1["data"][i][non_unique_columns[j]]

    file2["data"].extend(new_data1)

    return_arff = {
        'relation': 'CombinedData',
        'description': '',
        'data': file2["data"],
        'attributes': file2["attributes"]
    }
    arff.dump(return_arff, open("Datasets/combinedDataset.arff", "w+"))


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
    

combine_datasets("coojaData1", "KDDTrain+")
