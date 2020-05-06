from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import numpy as np


def sample(x_train, y_train, sampling_type="over"):
    print("Amount of anomalies before: ", len([1 for i in y_train if i == 1]))
    print("Amount of normal occurences before: ", len([1 for i in y_train if i == 0]))

    if sampling_type == "over":
        ros = RandomOverSampler()
        x_train, y_train = ros.fit_resample(x_train, y_train)
    else:
        rus = RandomUnderSampler()
        x_train, y_train = rus.fit_resample(x_train, y_train)

    print("Amount of anomalies after: ", len([1 for i in y_train if i == 1]))
    print("Amount of normal occurences after: ", len([1 for i in y_train if i == 0]))

    return x_train, y_train
