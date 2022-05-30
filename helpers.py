import math
import os
from itertools import permutations

import numpy as np
import pandas as pd


def int_shap_exact(samples, linear_coefs, dist):
    return linear_coefs * (samples - dist.mean)


def int_shap_part_exact(samples, linear_coefs, dist):
    # Note: inefficient, you are recomputing the cond mean. Better use combinations definition of SHAP values, but this
    # is OK, since just for testing.
    nb_features = len(samples[0])
    nb_samples = len(samples)
    perms = permutations(range(nb_features))
    avg = np.zeros((nb_samples, nb_features))
    len_perms = math.factorial(nb_features)
    for ip, p in enumerate(perms):
        p = np.array(p)
        for i in range(nb_features):
            pos = np.where(p == i)[0][0]
            known_features = p[pos + 1:]
            unknown_features = np.sort(p[:pos + 1])  # ordered!

            for j in range(nb_samples):
                cond_means = dist.sample(features=known_features, feature_values=samples[j, known_features], n=1,
                                         return_moments=True)[1]
                avg[j, i] += cond_means[np.where(unknown_features == i)[0][0]]
        print(ip, '/', len_perms)

    return linear_coefs * (samples - avg / len_perms)


def get_file_path(file):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_path, file)


def accuracy(pred, test):
    pred = pred > 0.5
    return np.sum(pred == test) / len(test)


def load_algerian(return_X_y=False):
    data = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/00547/Algerian_forest_fires_dataset_UPDATE.csv',
        usecols=['Temperature', 'RH', 'Ws', 'Rain', 'Classes'], skiprows=[0, 124, 125, 126], sep=' *, *')
    data.Classes = data.Classes.str.strip()
    data.at[165, 'Classes'] = 'fire'
    data.Classes.replace(('fire', 'not fire'), (True, False), inplace=True)
    if return_X_y:
        return data.iloc[:, :-1].values, data.iloc[:, -1].values
    else:
        return data

