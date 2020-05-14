from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import numpy as np


# If no y_train is given, it is because it is a part of the dataset, kinda hacky overload
def sample_data(x_train, sampling="over"):
    y_train = [row.pop(-1) for row in x_train]
    x_train, y_train = sample(x_train, y_train, sampling_type=sampling)

    return [x_train[i] + [y_train[i]] for i in range(len(x_train))]


def sample_with_attacks(dataset, attacks, sampling_type="over"):
    for i in range(len(dataset)):
        dataset[i].insert(-1, attacks[i])

    dataset = sample_data(dataset, sampling_type)

    attacks = [row.pop(-2) for row in dataset]

    return dataset, attacks


# Refactor for better boolean control later
def sample(x_train, y_train, sampling_type="over"):
    print("Amount of anomalies before: ", len([1 for i in y_train if is_anomaly(i)]))
    print("Amount of normal occurences before: ", len([1 for i in y_train if not is_anomaly(i)]))

    if sampling_type == "over":
        ros = RandomOverSampler()
        x_train, y_train = ros.fit_resample(x_train, y_train)
    else:
        rus = RandomUnderSampler()
        x_train, y_train = rus.fit_resample(x_train, y_train)

    print("Amount of anomalies after: ", len([1 for i in y_train if is_anomaly(i)]))
    print("Amount of normal occurences after: ", len([1 for i in y_train if not is_anomaly(i)]))

    return x_train, y_train


def is_anomaly(label):
    if type(label) is str:
        return "normal" not in label
    elif type(label) is int:
        return label == 1
    else:
        print("unknown label type")
