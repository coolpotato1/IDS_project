# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 11:28:39 2020

@author: Henning    
"""

import arff
import gc
import sys
import numpy as np
import math
import gensim
from itertools import islice
from collections import defaultdict
from dataset_manipulation import get_attack_column

# Define constants here
N_OBSERVATIONS_REQUIRED = 100


def one_hot_encode(column, column_no, attributes):
    one_hot_row = []
    for value in column:
        one_hot_row.append([int(value == attribute) for attribute in attributes[column_no][1]])

    return one_hot_row


def one_hot_insert(data, data_attributes, column_no):
    one_hot_values = one_hot_encode([row[column_no] for row in data], column_no, data_attributes)
    # If you run into memory issues, rewrite to use list methods instead of numpy
    left, delete, right = np.split(data, [column_no, column_no + 1], axis=1)
    data_return = np.concatenate((left, one_hot_values, right), axis=1)

    return data_return.tolist()


def binary_insert(data, data_attributes, column_no):
    values = {}
    for i in range(0, len(data_attributes[column_no][1])):
        values[data_attributes[column_no][1][i]] = i

    n_new_columns = math.ceil(math.log2(len(data_attributes[column_no][1])))
    for key in values:
        values[key] = format(values[key], '0{}b'.format(n_new_columns))

    binary_columns = []
    for row in data:
        binary_columns.append([char for char in values[row[column_no]]])

    left, delete, right = np.split(data, [column_no, column_no + 1], axis=1)
    data_return = np.concatenate((left, binary_columns, right), axis=1)

    return data_return.tolist()


def categorical_insert(data, data_attributes, column_no):
    for row in data:
        row[column_no] = [i for i in range(0, len(data_attributes[column_no][1]))
                          if row[column_no] == data_attributes[column_no][1][i]][0]

    return data


# Currently only works correctly on full datasets
def target_insert(data, data_attributes, column_no, is_test_data=False):
    # Get list of predictions for each unique value. This needs to be run before splitting predictions from data.
    if (not is_test_data):
        predictions_by_value = defaultdict(list)
        means = {}
        # This is used as a replacement value if there are not enough observations of a specific value to generalize
        overall_train_mean = np.mean([int(row[-1] == 'anomaly') for row in data])
        for row in data:
            predictions_by_value[row[column_no]].append(int(row[-1] == 'anomaly'))

        for key in predictions_by_value:
            if (len(predictions_by_value[key]) < N_OBSERVATIONS_REQUIRED):
                means[key] = overall_train_mean
            else:
                means[key] = np.mean(predictions_by_value[key])

            index = data_attributes[column_no][1].index(key)
            data_attributes[column_no][1][index] = (key, means[key])

    for row in data:
        mean_value = [i[1] for i in data_attributes[column_no][1] if i[0] == row[column_no]]
        row[column_no] = mean_value[0]

    return data


# use gensim.models.Word2Vec()
# Maybe look at this more later
def embedded_insert(data, data_attributes, column_no, is_test_data=False):
    embedded_values = []
    if (not is_test_data):
        attacks = get_attack_column("KDDTrain+")
        embed_data = [data[i][column_no - 2: column_no + 2] + [attacks[i]] for i in range(0, len(data))]
        model = gensim.models.Word2Vec(embed_data, min_count=1, size=5, window=5, iter=10)
        embedded_values = [model[row[column_no]] for row in data]

        for i in range(0, len(data_attributes[column_no][1])):
            data_attributes[column_no][1][i] = (
            data_attributes[column_no][1][i], model[data_attributes[column_no][1][i]])

    else:
        for row in data:
            embedded_values.append([i[1] for i in data_attributes[column_no][1] if i[0] == row[column_no]][0])

    left, delete, right = np.split(data, [column_no, column_no + 1], axis=1)
    data_return = np.concatenate((left, embedded_values, right), axis=1)

    return data_return


def choose_and_use_encoding(data, data_attributes, column_no):
    if type(data_attributes[column_no][1]) != list or len(data_attributes[column_no][1]) <= 2:
        return data
    elif len(data_attributes[column_no][1]) <= 6:
        return one_hot_insert(data, data_attributes, column_no)
    else:
        return binary_insert(data, data_attributes, column_no)


def remove_useless_columns(data):
    return_data = np.asarray(data).astype(np.float32)
    max_values = np.amax(return_data, axis=0)
    min_values = np.amin(return_data, axis=0)
    difference = max_values - min_values

    useless = [x for x in range(0, len(difference)) if difference[x] == 0]

    return_data = np.delete(return_data, useless, axis=1)
    return return_data


def normalize(data):
    # Remove columns where values are constant, done on min and diff arrays too, so dimensions fit for normalization.
    return_data = np.asarray(data).astype(np.float32)
    max_values = np.amax(return_data, axis=0)
    min_values = np.amin(return_data, axis=0)
    difference = max_values - min_values

    return_data = (return_data - min_values) / difference
    return_data = np.nan_to_num(return_data)
    return return_data


def process_data(data_values, data_attributes):
    for i in range(len(data_attributes), 1, -1):
        data_values = choose_and_use_encoding(data_values, data_attributes, -i)

    # Assign the class values
    predictions = []
    for row in data_values:
        predictions.append(int(row.pop(-1) == "anomaly"))

    # data_values = remove_useless_columns(data_values)
    return data_values, predictions


def load_and_process_data(datapath, n_components=1, do_normalize=False):
    values = []
    predictions = []
    if (n_components == 1):
        file = arff.load(open(datapath))
        values = file['data']
        attributes = file['attributes']

        values, predictions = process_data(values, attributes)
    else:
        for i in range(0, n_components):
            file = arff.load(open(datapath.split(".")[0] + str(i + 1) + "." + datapath.split(".")[1]))
            temp_data_values = file['data']
            attributes = file['attributes']
            temp_values, temp_predictions = process_data(temp_data_values, attributes, is_test_data)
            values.extend(temp_values)
            predictions.extend(temp_predictions)

    if (do_normalize):
        return normalize(values), predictions, attributes
    else:
        return values, predictions, attributes


def split_dataset(datapath, n_sections):
    file = arff.load(open(datapath))
    data_values = file['data']

    iterator = iter(data_values)
    split_size = int(len(data_values) / n_sections)
    splits = [split_size for i in range(1, n_sections)]
    splits.append(len(data_values) - split_size * (n_sections - 1))
    split_values = [list(islice(iterator, elem)) for elem in splits]

    for i in range(0, n_sections):
        new_file_data = {
            'relation': 'KDDTrain',
            'description': '',
            'data': split_values[i],
            'attributes': file['attributes']}
        new_file = open(datapath.split(".")[0] + str(i + 1) + "." + datapath.split(".")[1], "w+")
        arff.dump(new_file_data, new_file)

# split_dataset("Datasets/KDDTrain+.arff", 5)
