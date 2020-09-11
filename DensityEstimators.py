import numpy as np
from statsmodels.nonparametric.kde import KDEUnivariate


class DE_histogram:
    def __init__(self, num_bins):
        self.num_bins = num_bins

    def estimate(self, data, d_min, d_max, density=False):
        d_range = (d_min - 0.5 * (d_max - d_min), d_max + 0.5 * (d_max - d_min))
        self.values, self.grid = np.histogram(data, bins=self.num_bins, density=density, range=d_range)
        self.mids = self.grid[:-1] + np.diff(self.grid) / 2


class DE_kde:
    def __init__(self, num_bins, kernel='gau'):
        self.num_bins = num_bins
        self.kernel = kernel

    def estimate(self, data, d_min, d_max):
        d_range = (d_min - 0.5 * (d_max - d_min), d_max + 0.5 * (d_max - d_min))
        gridsize = (d_range[1] - d_range[0]) / self.num_bins
        bw = gridsize * self.num_bins / 100
        self.grid = [(d_range[0] + i * gridsize) for i in range(self.num_bins + 1)]
        self.mids = self.grid[:-1] + np.diff(self.grid) / 2

        try:
            kde = KDEUnivariate(data)
            kde.fit(bw=bw, kernel=self.kernel, fft=False)

            self.values = [kde.evaluate(i)[0] if kde.evaluate(i) > 0 else 0 for i in self.mids]
        except:
            print("KDE did not work, data length =", len(data), ", d_range =", d_range, ", gridsize =", gridsize)
            self.values = [0] * len(data[0])
