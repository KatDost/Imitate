from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import FastICA
import numpy as np


# Outlier removal: Local Outlier Factor
def remove_outliers_lof(data, k=10):
    k = min((len(data), k))
    lof = LocalOutlierFactor(n_neighbors=k)
    stays = lof.fit_predict(data)
    return np.array(data)[stays == 1]


# Alibi class; nothing happens here
class trafo_keep_axes:
    def __init__(self):
        pass

    def transform(self, data):
        return data

    def transform_back(self, points):
        return points


# Transformation via Independent Component Analysis (ICA)
class trafo_ica:
    def __init__(self, num_comp=0):
        self.num_comp = num_comp

    def transform(self, data):
        try:
            self.num_comp = len(data[0]) if self.num_comp == 0 else self.num_comp
            self.ica = FastICA(n_components=self.num_comp, max_iter=500, tol=0.01).fit(data)
        except:
            return data
        else:
            return self.ica.transform(data)

    def transform_back(self, points):
        try:
            return self.ica.inverse_transform(points)
        except:
            return points
