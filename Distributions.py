import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

'''Scaled normal distribution'''


class scaled_norm:
    def __init__(self, ends_zero=True):
        self.ends_zero = ends_zero

    def func(x, scale, mu, sigma):
        return scale * norm(mu, sigma).pdf(x)

    def weighted_dist(weights, points_x, points_y, params):
        return (((scaled_norm.func(points_x, *params) - points_y) ** 2) * weights).sum()

    def constraint(points_x, points_y, params):
        return 2 * points_y.sum() - scaled_norm.func(points_x, *params).sum()

    def fit(self, points_x, points_y, data):
        d_mean = points_x[np.argmax(points_y)]  # highest bin
        d_std = np.sqrt(np.sum((np.array(data) - d_mean) ** 2) / (len(data) - 1))
        d_scale = max(points_y) / max(scaled_norm.func(points_x, 1, d_mean, d_std))
        p0 = np.array([d_scale, d_mean, d_std])  # initial parameters
        weights = np.array(points_y) ** 2
        weights = [max(weights[i], 0.01 * max(points_y)) for i in range(len(weights))]
        optimize_me = lambda p: scaled_norm.weighted_dist(weights, points_x, points_y, p)
        if self.ends_zero:
            weights[0] = weights[-1] = max(points_y)
        try:
            bounds = [[0.01, 2 * d_scale], [points_x[0], points_x[-1]], [0, (points_x[-1] - points_x[0]) / 2]]
            constr = lambda p: scaled_norm.constraint(np.array(points_x), np.array(points_y), p)
            res = minimize(optimize_me, p0, method='SLSQP', bounds=bounds, constraints={'type': 'ineq', 'fun': constr})
            if constr(res.x) < 0:
                return np.array(points_y)
            return scaled_norm.func(points_x, *res.x)
        except:
            # print("pdf fitting with sigma was not successful")
            return np.array(points_y)
