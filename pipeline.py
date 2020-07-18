from io import BufferedWriter, RawIOBase

import arff
import numpy as np
import dataset_manipulation as man
from NSL_KDD_attack_types import attack_types as types
import socket

data = arff.load(open("Datasets/KDDTest+_filtered.arff"))['data']
attacks = np.squeeze(man.csv_read("Datasets/KDDTest+_filtered_attacks"))
# data, attributes = man.remove_nan_attributes(file['data'], file['attributes'])
max = 0
for i in range(len(data)):
    if attacks[i] in types.APACHE2.value:
        print(data[i])

print("debugging point")
