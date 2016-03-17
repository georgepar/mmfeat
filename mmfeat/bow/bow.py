import random

import numpy as np

from scipy.cluster.vq import vq

from sklearn.cluster import MiniBatchKMeans

class BoW():
    def __init__(self, K, subsample=None, normalize=True, verbose=True, saveMeans=True):
        '''
        K:          number of dimensions
        subsample:  if not None, do not use all files for clustering but only a sample
        verbose:    verbosity
        normalize:  provide normalized vectors
        verbose:    be verbose
        saveMeans:  save the means (i.e., centroids) to the datadir as centroids.pkl
        '''
        self.K = int(K)
        self.subsample = subsample
        self.verbose = verbose
        self.normalize = normalize
        self.saveMeans = saveMeans

        self.centroids = None

    def load(self, datadir):
        raise BoWError('This method is not implemented for the BoW parent class.')

    def fit(self, data=None):
        '''
        data:       data dictionary {'filename': np.array(...), ...}
                    or a class with __getitem__() and keys()
        '''
        if data is not None:
            self.data = data

        if self.centroids is None:
            self.centroids = self.cluster()

        self.descriptors = self.quantize()

    def cluster(self):
        mbk = MiniBatchKMeans(n_clusters=self.K, batch_size=self.K*2, verbose=self.verbose, compute_labels=False)
        if self.subsample is None:
            data = np.vstack([self.data[k] for k in self.data.keys()])
            mbk.fit(data)
        else: # sample number of files
            fnames = self.data.keys()
            subset = random.sample(fnames, int(self.subsample * len(fnames)))
            subdata = np.vstack([self.data[k] for k in subset])
            mbk.fit(subdata)
        return mbk.cluster_centers_

    def quantize(self):
        clusters = range(self.centroids.shape[0] + 1)
        histograms = {}
        for fname in sorted(self.data.keys()):
            idx,_ = vq(self.data[fname], self.centroids)
            histograms[fname], _ = np.histogram(idx, bins=clusters, normed=self.normalize)
        return histograms

    def sequences(self):
        sequences = {}
        for fname in sorted(self.data.keys()):
            idx,_ = vq(self.data[fname], self.centroids)
            sequences[fname] = idx
        return sequences

    def means(self):
        means = {}
        for fname in sorted(self.data.keys()):
            means[fname] = np.mean(self.data[fname], axis=0)
        return means
