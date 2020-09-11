import random
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import copy
import Bias, Transformations, Confidence, Distributions
from sklearn import svm


# Main Class for Imitate
class IMITATE:

    def __init__(self, num_hist_int, bias_gen, data_gen, DE, repeat=1, trafo=Transformations.trafo_keep_axes,
                 model=svm.SVC(kernel='linear')):
        self.num_hist_int = num_hist_int
        self.bias_gen = bias_gen
        self.data_gen = data_gen
        self.repeat = repeat
        self.model = svm.SVC(kernel='linear')
        self.reset(new_data=True)
        self.density_func = Distributions.scaled_norm(ends_zero=True)
        self.DE = DE
        self.trafo = trafo
        self.colors = [plt.cm.viridis(0), 'teal', 'goldenrod', 'deepskyblue']
        self.colormap = lambda x: [self.colors[self.D.labels.index(xi)] for xi in x]

    # Helper for experiments: Tests and compares different numbers of histogram/KDE bins
    def run(self, fill_up_plots=False, point_plots=False, result_plot=False, iterations=10, remove_outliers=True):

        err_acc = np.zeros((len(self.num_hist_int) + 1, self.repeat))
        confidence = np.zeros((self.D.num_classes, len(self.num_hist_int) + 1, self.repeat))
        num_added = np.zeros((self.D.num_classes, len(self.num_hist_int) + 1, self.repeat))

        for it in range(self.repeat):
            if it > 0:
                self.reset(new_data=True)
            err_acc[0][it] += self.D.acc_init

            for i in range(len(self.num_hist_int)):
                self.reset(new_data=False)
                conf = self.fill_up(self.num_hist_int[i], fill_up_plots=fill_up_plots,
                                    point_plots=point_plots, iterations=iterations, RO=remove_outliers)
                if result_plot:
                    self.plot_result(str(self.num_hist_int[i]))
                # evaluate result
                err_acc[i + 1][it] += self.D.accuracyBiased(self.added_points, self.added_labels)
                for l in range(self.D.num_classes):
                    confidence[l][i + 1][it] += conf[l]
                    num_added[l][i + 1][it] += list(self.added_labels).count(self.D.labels[l])

        self.err_acc = err_acc
        self.confidence = confidence
        self.num_added = num_added

        self.err_acc_mean = err_acc.sum(axis=1) / self.repeat
        self.confidence_mean = confidence.sum(axis=2) / self.repeat
        self.num_added_mean = num_added.sum(axis=2) / self.repeat

        self.plot_eval()

    # fill up the distribution for a certain number of histogram bins per dimension
    # This is IMITATE as presented in the paper!
    def fill_up(self, num_bins, iterations=10,
                fill_up_plots=False, point_plots=False, RO=True, t=1):

        # consider every label seperately
        label_confidence = []
        for label in self.D.labels:
            label_idx = self.D.labels.index(label)

            '''collect training data'''
            data = self.D.X_b_train[self.D.Y_b_train == label]

            '''remove outliers, rotate data'''
            if RO:
                data = Transformations.remove_outliers_lof(data)
            trafo = self.trafo()
            data = trafo.transform(data)

            cdfs_scaled = np.empty((len(data[0]), num_bins))
            fitted_cdf = np.empty((len(data[0]), num_bins))
            fitted_ = np.empty((len(data[0]), num_bins))
            num_fill_up = 0
            data_range = []

            DE_list = []

            if fill_up_plots:
                f, ax = plt.subplots(nrows=1, ncols=len(data[0]), figsize=(6, 2.5))

            # consider every dimension
            for line in range(len(data[0])):

                '''project onto line, determine borders'''
                d = data[:, line]
                d_min = min(d)
                d_max = max(d)
                data_range.append([d_min, d_max])

                '''define Density Estimator here!'''
                DE_list.append(self.DE(num_bins))
                DE_list[line].estimate(d, d_min, d_max)

                '''estimate distribution'''
                fitted = self.density_func.fit(DE_list[line].mids, DE_list[line].values, d)
                fitted_[line] = copy.deepcopy(fitted)
                fitted_cdf[line] = np.cumsum(fitted)
                fitted_cdf[line] = fitted_cdf[line] / fitted_cdf[line][-1]

                '''to be filled up: the differences between the distribution curve and the histogram'''
                diff = fitted - DE_list[line].values

                '''number of points to add'''
                num_points_line = (len(d) / sum(DE_list[line].values)) * sum(diff)
                num_fill_up = max(num_fill_up, num_points_line)

                '''probability distribution for the fill-up'''
                if sum(diff) == 0:
                    cdfs_scaled[line] = [0] * num_bins
                else:
                    diff = diff / sum(diff)
                    diff = [max(diff[i], 0) for i in range(len(diff))]
                    cdfs_scaled[line] = np.cumsum(diff)
                    cdfs_scaled[line] = (cdfs_scaled[line] / cdfs_scaled[line][-1]) * num_points_line

                if fill_up_plots:
                    barWidth = DE_list[line].mids[1] - DE_list[line].mids[0]
                    fill = fitted_[line] - DE_list[line].values
                    ax[line].bar(DE_list[line].mids, DE_list[line].values, label='data', color='teal',
                                 width=barWidth)
                    ax[line].bar(DE_list[line].mids, [max(fill[i], 0) for i in range(len(fill))],
                                 bottom=DE_list[line].values, label='fill up', color='goldenrod', width=barWidth,
                                 hatch="...", edgecolor="white")
                    ax[line].plot(DE_list[line].mids, fitted_[line], label='fitted', c='mediumvioletred', linewidth=2)
                    ax[line].get_xaxis().set_ticks([])
                    ax[line].get_yaxis().set_ticks([])

            if fill_up_plots:
                ax[-1].legend()
                plt.show()
                # f.savefig('Results/Example_cluster_distr.pdf', format='pdf', dpi=1200, bbox_inches='tight')

            # determine the number of added points in total: max over dimensions
            num_fill_up = int(num_fill_up)
            if num_fill_up == 0:
                label_confidence.append(0)
                continue

            # best out of 10: go for the result with the highest confidence
            best_conf = 0
            leftover_points = []
            # kNN_rnd_dist, kNN_rnd_std = confidence_kNN_rnd_coeff(data_range, num_fill_up)
            kNN_rnd_dist, kNN_rnd_std = Confidence.confidence_kNN_train_sized_coeff(data, num_fill_up)
            for it in range(iterations):
                points = np.empty((num_fill_up, 0))

                # generate points
                for line in range(len(data[0])):
                    '''adjust cdf (in case there have to be more points added because of other lines)'''
                    distr_scaled = fitted_cdf[line] * max((num_fill_up - cdfs_scaled[line][-1]), 0)
                    cdf = cdfs_scaled[line] + distr_scaled
                    cdf = cdf / cdf[-1]  # normalize

                    '''generate random values according to the cdf'''
                    values = np.random.rand(num_fill_up)
                    value_bins = np.searchsorted(cdf, values)
                    coords = np.array([random.uniform(DE_list[line].grid[value_bins[i]],
                                                      DE_list[line].grid[value_bins[i] + 1])
                                       for i in range(num_fill_up)]).reshape(num_fill_up, 1)
                    points = np.concatenate((points, coords), axis=1)

                '''compute the confidence of the result'''
                if len(points) < 20:
                    conf_b, conf_a, l_p = (0, 0, [[]])
                else:
                    conf_b, conf_a, l_p = Confidence.confidence_kNN_rnd(points, kNN_rnd_dist, t * kNN_rnd_std)

                # add the points to the data set
                if conf_a > best_conf:
                    best_conf = conf_a
                    leftover_points = copy.deepcopy(l_p)
                    # leftover_points = points

                if point_plots:
                    plt.figure(it)
                    plt.scatter(data[:, 0], data[:, 1], c=self.colors[label_idx], alpha=0.2, s=3)
                    plt.scatter(points[:, 0], points[:, 1], c='red', alpha=0.8, s=8)
                    if len(l_p) > 0 and len(l_p[0]) > 0:
                        plt.scatter(l_p[:, 0], l_p[:, 1], c=self.colors[label_idx], alpha=0.8, s=8)
                    plt.show()
                    if len(data[0]) > 2:
                        plt.figure(it * 100)
                        plt.scatter(data[:, 0], data[:, 2], c=self.colors[label_idx], alpha=0.2, s=3)
                        plt.scatter(points[:, 0], points[:, 2], c='red', alpha=0.8, s=8)
                        if len(l_p) > 0 and len(l_p[0]) > 0:
                            plt.scatter(l_p[:, 0], l_p[:, 2], c=self.colors[label_idx], alpha=0.8, s=8)
                        plt.show()

            '''remove the points with low confidence, discard the result entirely 
               if the confidence is too low. Transform back the leftover points'''
            if len(leftover_points) > 0:  # and 1 / best_conf <= kNN_rnd_dist + t*kNN_rnd_std:
                add_me = trafo.transform_back(leftover_points)
                self.added_points = np.concatenate((self.added_points, add_me))
                self.added_labels = np.append(self.added_labels, [label] * len(add_me))

            label_confidence.append(best_conf)
        if point_plots:
            plt.show()

        return label_confidence

    def reset(self, new_data=False):
        if new_data:
            # Draw a new data set
            self.D = Bias.BIASme(self.bias_gen, self.data_gen, model=self.model)
        self.added_points = np.empty((0, self.D.dims))
        self.added_labels = np.empty(0)

    def plot_result(self, title=""):
        # plt.title(title)
        plt.scatter(self.D.X_b_train[:, 0], self.D.X_b_train[:, 1], c=self.colormap(self.D.Y_b_train), alpha=0.2, s=3)
        plt.scatter(self.added_points[:, 0], self.added_points[:, 1],
                    c=self.colormap([self.D.labels.index(l) for l in self.added_labels]),
                    alpha=0.8, s=8)
        plt.show()

    def plot_eval(self):
        fig, acc = plt.subplots(nrows=1, ncols=1)
        x = np.concatenate(([0], self.num_hist_int))

        conf = acc.twinx()
        added = acc.twinx()
        acc.set_xticks(range(len(x)))
        acc.set_xticklabels(x)

        acc.set_xlabel("# Bins")
        # acc.set_ylabel("Acc_unb - Acc_b+add")
        acc.set_ylabel("Accuracy_Test")
        conf.set_ylabel("Confidence")
        added.set_ylabel("Added Points")

        color1 = plt.cm.viridis(0)
        color2 = plt.cm.viridis(0.5)
        color3 = plt.cm.viridis(.9)
        m = ['o', '^', '*', 'x']
        l = ['-', '--', '-.', ':']

        acc_line, = acc.plot(self.err_acc_mean, color=color1, label=acc.get_ylabel())
        for i in range(len(x)):
            acc.scatter([i] * self.repeat, self.err_acc[i], color=color1, alpha=0.2)
        conf_line = mlines.Line2D([], [], color=color2, label=conf.get_ylabel())
        added_line = mlines.Line2D([], [], color=color3, label=added.get_ylabel())
        lns = [acc_line, conf_line, added_line]
        for i in range(self.D.num_classes):
            conf.plot(self.confidence_mean[i], color=color2, linestyle=l[i], marker=m[i])
            added.plot(self.num_added_mean[i], color=color3, linestyle=l[i], marker=m[i])
            lns.append(mlines.Line2D([], [], color='black', linestyle=l[i], marker=m[i], label=str(self.D.labels[i])))

        acc.legend(handles=lns, loc='best')
        added.spines['right'].set_position(('outward', 60))

        acc.yaxis.label.set_color(acc_line.get_color())
        conf.yaxis.label.set_color(conf_line.get_color())
        added.yaxis.label.set_color(added_line.get_color())
        plt.show()
