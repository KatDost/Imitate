import numpy as np
from sklearn.neighbors import NearestNeighbors
import random


def confidence_kNN_train_sized_coeff(training_set, size):
    if len(training_set) < 20 or size < 20:
        return 0.0001, 0

    size = min(size, len(training_set))
    sample = np.array(random.sample(training_set.tolist(), size))

    ''' compute average distance in the training set'''
    nbrs_t = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(sample)
    distances_t, _ = nbrs_t.kneighbors(sample)
    distances_t = np.array(distances_t)[:, 1:]
    avg_dist_per_point = np.average(distances_t, axis=1)
    random_dist = np.average(avg_dist_per_point)
    random_dist_std = np.std(avg_dist_per_point)

    return random_dist, random_dist_std


def confidence_kNN_rnd(data, kNN_rnd_dist, kNN_rnd_std):
    if len(data) < 10:
        return 0, 0, [[]]

    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(data)
    distances, _ = nbrs.kneighbors(data)
    distances = np.array(distances)[:, 1:]

    stays = np.average(distances, axis=1) <= kNN_rnd_dist + kNN_rnd_std  # 0.5*kNN_rnd_std
    data_remain = data[stays]
    confidence_before = 1 / np.average(distances)
    confidence_after = 0 if len(data_remain) <= 10 else 1 / np.average(distances[stays])
    return confidence_before, confidence_after, data_remain

