# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:19:56 2020

@author: Henning
"""
import csv
import arff
import numpy as np
import classification_utils as u
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from NSL_KDD_attack_types import attack_types as attacks


def get_filter_indices(filtered_attacks, data_path):
    with open(data_path) as file:
        data = csv.reader(file)
        return_indexes = []
        i = 0
        for row in data:
            if (row[-2] in filtered_attacks):
                return_indexes.append(i)
            i += 1

    return return_indexes


def remove_attacks(data, filter_indices):
    return [data[i] for i in range(len(data)) if i not in filter_indices]


def create_filtered_dataset(file_name, filtered_attacks):
    file = arff.load(open("Datasets/" + file_name + ".arff"))
    original_data = file['data']
    attributes = file['attributes']
    attack_types = get_attack_column(file_name)
    new_data = remove_attacks(original_data, get_filter_indices(filtered_attacks, "Datasets/" + file_name + ".txt"))
    new_attack_types = remove_attacks(attack_types, get_filter_indices(filtered_attacks, "Datasets/" + file_name + ".txt"))
    return_arff = {
        'relation': 'KDDFiltered',
        'description': '',
        'data': new_data,
        'attributes': attributes
    }
    arff.dump(return_arff, open("Datasets/" + file_name + "_filtered.arff", "w+"))

    file = open("Datasets/" + file_name + "_filtered_attacks", "w+")
    for attack in new_attack_types:
        file.write(attack + "\n")

    file.close()


def csv_read(datapath):
    with open(datapath) as file:
        data = csv.reader(file)
        return [row for row in data]


# Deprecated (somewhat)
def get_attack_column(dataset):
    data = csv_read("Datasets/" + dataset + ".txt")
    return [row[-2] for row in data]


def read_attacks_from_file(dataset):
    file = open("Datasets/" + dataset + "_attacks", "r")
    attacks = file.readlines()
    return attacks


def write_attack_column(dataset):
    attacks = get_attack_column(dataset)
    file = open("Datasets/" + dataset + "_attacks", "w+")
    for attack in attacks:
        file.write(attack + "\n")

    file.close()


def get_attack_types(train_path, test_path):
    test_data = csv_read(test_path)
    train_data = csv_read(train_path)

    test_set = set([row[-2] for row in test_data])
    train_set = set([row[-2] for row in train_data])

    return train_set, test_set


def match_attribute_datatypes(data, attributes):
    indexes = []
    for i in range(len(attributes)):

        # Do not allow None values in class labels, as it should always be labeled as either anomaly or normal
        # To support lists with different types, you could save the indexes with a tuple that includes list type
        if attributes[i][0] == "class":
            continue
        if type(attributes[i][1]).__name__ == "list":
            indexes.append(i)

            if "None" not in attributes[i][1]:
                attributes[i][1].append("None")



    # Iterate through data, and if the value in a column that takes string values, is not a string value, we replace it with the string "None"
    # This will bug out with an integer list
    for row in data:
        for i in indexes:
            if type(row[i]).__name__ != "str":
                row[i] = "None"


def combine_and_return_attacks(dataset1, dataset2):
    attacks1 = read_attacks_from_file(dataset1)
    attacks2 = read_attacks_from_file(dataset2)
    return attacks1 + attacks2


# This method definitely needs to be refactored, holy fuck its ugly. This has probably added +5 hours to technical debt.
def combine_datasets(dataset1, dataset2, combinedDataset, sampling=None):
    file1 = arff.load(open("Datasets/" + dataset1 + ".arff"))
    file2 = arff.load(open("Datasets/" + dataset2 + ".arff"))

    # create the combined attackfile, it is on purpose that the datasets are inverted
    combined_attacks = combine_and_return_attacks(dataset2, dataset1)

    unique_columns = [i for i in range(len(file1["attributes"])) if file1["attributes"][i][0]
                      not in [attribute[0] for attribute in file2["attributes"]]]

    # needed in case any of the non-unique columns have values that do not exist in the other dataset
    non_unique_columns = [i for i in range(len(file1["attributes"])) if i not in unique_columns]
    for column in non_unique_columns:
        index = \
            [i for i in range(len(file2["attributes"])) if file2["attributes"][i][0] == file1["attributes"][column][0]][
                0]
        # This should only happen if the compared attributes contain lists
        if file1["attributes"][column][1] != file2["attributes"][index][1]:
            file2["attributes"][index][1].extend(
                list(set(file1["attributes"][column][1]) - set(file2["attributes"][index][1])))

    # Also add the unique attributes
    file2["attributes"].extend([file1["attributes"][column_no] for column_no in unique_columns])

    # Match dimensions of second dataset, by padding with zeroes
    for row in file2["data"]:
        row.extend([0] * len(unique_columns))

    # Match dimensions for the first dataset and ensure that common attributes are placed in correct columns
    new_data1 = []
    new_column_indexes = []
    for i in range(len(non_unique_columns)):
        new_column_indexes.extend([j for j in range(len(file2["attributes"])) if
                                   file2["attributes"][j][0] == file1["attributes"][non_unique_columns[i]][0]])

    for i in range(len(file1["data"])):
        # Create zeroes for all of file2's attributes, and then add file 1's. Its so dimensions match
        new_data1.append(
            [0] * (len(file2["attributes"]) - len(unique_columns)) + [file1["data"][i][j] for j in unique_columns])
        for j in range(len(new_column_indexes)):
            new_data1[i][new_column_indexes[j]] = file1["data"][i][non_unique_columns[j]]

    file2["data"].extend(new_data1)

    # Find class label and return it to the last column (its rightful place)

    index = [i for i in range(len(file2["attributes"])) if file2["attributes"][i][0] == "class"][0]

    temp_attribute = file2["attributes"][index]
    file2["attributes"][index] = file2["attributes"][-1]
    file2["attributes"][-1] = temp_attribute
    for row in file2["data"]:
        temp_data = row[index]
        row[index] = row[-1]
        row[-1] = temp_data

    match_attribute_datatypes(file2["data"], file2["attributes"])

    if sampling is not None:
        file2["data"], combined_attacks = u.sample_with_attacks(file2["data"], combined_attacks, sampling)


    return_arff = {
        'relation': 'CombinedData',
        'description': '',
        'data': file2["data"],
        'attributes': file2["attributes"]
    }

    file = open("Datasets/" + combinedDataset + "_attacks", "w+")
    for attack in combined_attacks:
        file.write(attack)

    file.close()
    arff.dump(return_arff, open("Datasets/" + combinedDataset + ".arff", "w+"))


def get_normal_and_anomaly_scores(model, test_data, actual_classes=None, fscore_beta = 1, is_nn = True):
    # If no actual classes are given, its because they are part of the data, and we should extract them
    if actual_classes is None:
        actual_classes = [row.pop(-1) for row in test_data]

    if is_nn:
        predicted_classes = model.predict_classes(test_data)
    else:
        predicted_classes = model.predict(test_data)
    precision, recall, fscore, count = precision_recall_fscore_support(actual_classes, predicted_classes, beta=fscore_beta, average="binary")
    accuracy = len([i for i in range(len(predicted_classes)) if predicted_classes[i] == actual_classes[i]])/len(actual_classes)

    return precision, recall, fscore, accuracy


# Currently this one only works on NSL-KDD datasets
def get_specific_scores(datapath, clf, data, actual_class, attack_types, keep_separated=False):
    csv_data = csv_read(datapath)
    attack_indices = []
    return_scores = []
    for attack in attack_types:
        if (keep_separated):
            attack_indices.append([i for i in range(0, len(csv_data)) if csv_data[i][-2] in attack])
        else:
            attack_indices.extend([i for i in range(0, len(csv_data)) if csv_data[i][-2] in attack])

    # Check if we have a 1-dimensional or 2 dimensional list
    if (type(attack_indices[0]) == list):
        for index_list in attack_indices:
            return_scores.append(clf.score([data[i] for i in range(0, len(data)) if i in index_list],
                                           [actual_class[i] for i in range(0, len(actual_class)) if i in index_list]))

    else:
        return_scores.append(clf.score([data[i] for i in range(0, len(data)) if i in attack_indices],
                                       [actual_class[i] for i in range(0, len(actual_class)) if i in attack_indices]))

    return return_scores


# Normal classes with "not keep_separated" have not been tested yet
def get_specific_recall(clf, data, actual_classes, attack_types, keep_separated=False, is_nn=True):
    predictions = []
    recalls = []
    if not keep_separated:
        if "normal" in attack_types:
            attack_types.remove("normal")
            if is_nn:
                predictions.extend(np.squeeze(clf.predict_classes(
                    np.asarray([data[i] for i in range(len(actual_classes)) if actual_classes[i] == "normal"]))))
            else:
                predictions.extend(np.squeeze(clf.predict(
                    np.asarray([data[i] for i in range(len(actual_classes)) if actual_classes[i] == "normal"]))))

            # Invert predictions (because correct normal prediction is 0, but later we use 1's to count it
            predictions = [int(element == 0) for element in predictions]

        if is_nn:
            predictions.extend(np.squeeze(clf.predict_classes(np.asarray([data[i] for i in range(len(actual_classes)) if actual_classes[i] in attack_types]))))
        else:
            predictions.extend(np.squeeze(clf.predict(np.asarray([data[i] for i in range(len(actual_classes)) if actual_classes[i] in attack_types]))))

        # As we have separated by attack, all predictions should be one. Therefore we can find recall this way
        recalls.append(len([element for element in predictions if element == 1]) / len(predictions))
    else:
        #This should work when normals are separated, think it will when normals are combined too. but dont combine normals and attacks
        for attack in attack_types:
            if is_nn:
                predictions.append(np.squeeze(clf.predict_classes(np.asarray([data[i] for i in range(len(actual_classes)) if actual_classes[i] in attack]))))
            else:
                predictions.append(np.squeeze(clf.predict(np.asarray([data[i] for i in range(len(actual_classes)) if actual_classes[i] in attack]))))

            # Refactor later, this only works because you are giving normals separately
            if attack[0] in attacks.ALL_NORMALS.value:
                predictions[-1] = [int(element == 0) for element in predictions[-1]]

            recalls.append(len([element for element in predictions[-1] if element == 1]) / len(predictions[-1]))

    return recalls


def remove_nan_attributes(data, attributes):
    # assumes 2-dimensional data
    keep_indexes = {}
    # Last row is label, which we wanna handle separately
    for i in range(len(data[0]) - 1):
        keep_indexes[i] = False

    # We want to check for each column of the data, if it fits a numeric attribute or not. We return true if it is,
    # and false if not. But some values are empty, so it takes several rows before we figure out if they are or not.
    for row in data:

        for i in range(len(row) - 1):
            if row[i] == "":
                continue
            try:
                float(row[i])
                keep_indexes[i] = True
            except ValueError:
                pass

    return_data = []

    for row in data:
        return_data.append([row[i] for i in range(len(row) - 1) if keep_indexes[i] == True])
        return_data[-1].append(row[-1] if row[-1] == "normal" else "anomaly")

    for i in range(len(return_data)):
        new_row = [element if element != "" else "0" for element in return_data[i]]
        return_data[i] = new_row

    #Minus 1 cause we add the label attribute manually
    return_attributes = [(attributes[i], "REAL") for i in range(len(attributes) - 1) if keep_indexes[i] == True]

    #The class attribute needs to match the standard format (learned this the hard way)
    return_attributes.append(("class", ["normal", "anomaly"]))
    return return_data, return_attributes

def export_arff(data, attributes, filename, relation="Data", description=None):
    exported_arff = {
        'relation': relation,
        'description': description,
        'data': data,
        'attributes': attributes
    }

    arff.dump(exported_arff, open("Datasets/" + filename + ".arff", "w+"))


def export_attacks(attack_list, file):

    f = open(file, "w+")
    for attack in attack_list:
        f.write(attack + "\n")

    f.close()
    print("exported attacks")


# Currently only deals with one attack type
def packet_csv_to_arff(datafile_in, datafile_out, attack_type, normal_type, split=None, relation="Data", sampling=None):
    data = csv_read("Datasets/" + datafile_in + ".csv")
    attributes = data.pop(0)
    data, attributes = remove_nan_attributes(data, attributes)

    if sampling is not None:
        data = u.sample_data(data, sampling=sampling)

    # Refactor to arff method at some point
    if split is not None:
        train_data, test_data = train_test_split(data, test_size=0.2)
        export_arff(train_data, attributes, datafile_out + "Train", relation=relation + "_Train")
        export_arff(test_data, attributes, datafile_out + "Test", relation=relation + "_Test")

        export_attacks([attack_type if row[-1] == "anomaly" else normal_type for row in train_data], "Datasets/" + datafile_out + "Train_attacks")
        export_attacks([attack_type if row[-1] == "anomaly" else normal_type for row in test_data], "Datasets/" + datafile_out + "Test_attacks")

    else:
        export_arff(data, attributes, datafile_out, relation=relation)
        export_attacks([row[-1] for row in data], "Datasets/" + datafile_out + "_attacks")



print("scripts run apparently")
#create_filtered_dataset("KDDTest+", attacks.U2R.value + attacks.R2L.value)
# write_attack_column("KDDTrain+_20Percent")
#combine_datasets("combinedCoojas", "MitMKDDTrain", "UDPMitMKDDTrain")
#packet_csv_to_arff("MitM", "MitM", "MitM", "MitM_normal", 0.2, sampling="under")
