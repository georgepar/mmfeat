'''
Multi-modal models, with middle or late fusion and several combination methods.
'''

from .base import Space
from .sim import cosine, norm

import csv
import os
import sys

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats.stats import spearmanr
from scipy.stats.stats import pearsonr
import random

import warnings
warnings.filterwarnings("ignore")

class MMSpace(Space):
    def __init__(self, lingSpace, visSpace, modelType='middle', methodType='wmm', buildModel=False, alpha=0.5):
        self.reportMissing = True
        self.alpha = alpha
        self.lingSpace = lingSpace
        self.visSpace = visSpace
        self.modelType = modelType
        self.setMethodType(methodType)

        if modelType == 'middle':
            # do some checks on visspace
            example_key = self.visSpace.keys()[0]
            if isinstance(self.visSpace[example_key], dict):
                raise TypeError('Expecting vectors for the visual space, not dictionaries. Did you use AggSpace?')
            elif not isinstance(self.visSpace[example_key], np.ndarray):
                raise TypeError('Expecting numpy.ndarray')
            elif self.visSpace[example_key].ndim != 1:
                raise TypeError('Expecting tensors of rank 1 (vectors).')
            # and on lingspace
            example_key = self.lingSpace.keys()[0]
            if not isinstance(self.lingSpace[example_key], np.ndarray):
                raise TypeError('Expecting numpy.ndarray')

            if buildModel: # build model in-place
                self.space = {}
                for k in self.lingSpace.keys():
                    if k in self.visSpace.keys():
                        self.space[k] = self.concat(self.lingSpace[k], self.visSpace[k])
            else: # or override similarity function
                self.sim = self.midSimFunc
        elif modelType == 'late':
            self.sim = self.lateSimFunc

    def concat(self, u, v, alpha=None):
        if alpha is None:
            alpha = self.alpha

        u /= norm(u)
        v /= norm(v)
        return np.hstack((alpha * u, (1 - alpha) * v))

    def setMethodType(self, methodType):
        if methodType in ['dfmm', 'dwmm']:
            if not hasattr(self.visSpace, 'dispersions'):
                raise ValueError('Dispersion-dependent method selected but no dispersions loaded.')
            if methodType == 'dfmm':
                self.filterThreshold = np.median(self.visSpace.dispersions.values())
        self.methodType = methodType

    def midSimFunc(self, x, y, alpha=None):
        if self.methodType == 'wmm':
            mmx = self.concat(self.lingSpace[x], self.visSpace[x], alpha=alpha)
            mmy = self.concat(self.lingSpace[y], self.visSpace[y], alpha=alpha)
        elif self.methodType == 'dfmm':
            if self.visSpace.dispersions[x] > self.filterThreshold and self.visSpace.dispersions[x] > self.filterThreshold:
                mmx = self.concat(self.lingSpace[x], self.visSpace[x])
                mmy = self.concat(self.lingSpace[y], self.visSpace[y])
            else:
                mmx = self.lingSpace[x]
                mmy = self.lingSpace[y]
        elif self.methodType == 'dwmm':
            mmx = self.concat(self.lingSpace[x], self.visSpace[x], self.visSpace.dispersions[x])
            mmy = self.concat(self.lingSpace[y], self.visSpace[y], self.visSpace.dispersions[y])

        mmx /= norm(mmx)
        mmy /= norm(mmy)

        return cosine(mmx, mmy)

    def lateSimFunc(self, x, y, alpha=None):
        if alpha is None:
            alpha = self.alpha

        lingSim = self.lingSpace.sim(x, y)
        visSim = self.visSpace.sim(x, y)

        if self.methodType == 'wmm':
            ret = alpha * lingSim + (1 - alpha) * visSim
        elif self.methodType == 'dfmm':
            if self.visSpace.dispersions[x] > self.filterThreshold and self.visSpace.dispersions[x] > self.filterThreshold:
                ret = alpha * lingSim + (1 - alpha) * visSim
            else:
                ret = lingSim
        elif self.methodType == 'dwmm':
            ret = self.visSpace.dispersions[x] * lingSim + (1 - self.visSpace.dispersions[y]) * visSim
        return ret


class AVTSpace(Space):
    def __init__(self, lingSpace, visSpace, audioSpace, modelType='middle', methodType='wmm', alpha=0.33, beta=0.33):
        self.reportMissing = True
        self.alpha = alpha
        self.beta = beta
        self.lingSpace = lingSpace
        self.visSpace = visSpace
        self.audioSpace = audioSpace
        self.modelType = modelType
        self.setMethodType(methodType)

        if modelType == 'middle':
            # do some checks on visspace
            example_key = self.visSpace.keys()[0]
            if isinstance(self.visSpace[example_key], dict):
                raise TypeError('Expecting vectors for the visual space, not dictionaries. Did you use AggSpace?')
            elif not isinstance(self.visSpace[example_key], np.ndarray):
                raise TypeError('Expecting numpy.ndarray')
            elif self.visSpace[example_key].ndim != 1:
                raise TypeError('Expecting tensors of rank 1 (vectors).')
            # and on lingspace
            example_key = self.lingSpace.keys()[0]
            if not isinstance(self.lingSpace[example_key], np.ndarray):
                raise TypeError('Expecting numpy.ndarray')

            self.sim = self.midSimFunc
        elif modelType == 'late':
            self.sim = self.lateSimFunc

    def concat(self, u, v, w, alpha=None, beta=None):
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta

        u /= norm(u)
        v /= norm(v)
        w /= norm(w)
        return np.hstack((alpha * u,  beta * v, (1 - alpha - beta) * w))

    def midSimFunc(self, x, y, alpha=None, beta=None):
        mmx = self.concat(self.lingSpace[x], self.visSpace[x], self.audioSpace[x], alpha=alpha, beta=beta)
        mmy = self.concat(self.lingSpace[y], self.visSpace[y], self.visSpace[y], alpha=alpha, beta=beta)
        mmx /= norm(mmx)
        mmy /= norm(mmy)
        return cosine(mmx, mmy)

    def lateSimFunc(self, x, y, alpha=None, beta=None):
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta
        lingSim = self.lingSpace.sim(x, y)
        visSim = self.visSpace.sim(x, y)
        audioSim = self.audioSpace.sim(x, y)
        return self.alpha * lingSim + self.beta * visSim + (1 - self.alpha - self.beta) * audioSim


def MMEstimator(BaseEstimator):
    def __init__(mmspace, alpha=0.5, beta=None):
        self.mmspace = mmspace
        self.alpha = alpha
        self.actual_values = {}
        self.test_pairs = 0

    def loadDataset(self, datasetLocation='/home/geopar/projects/mmfeat/datasets', dataset='men'):
        if dataset == 'men':
            ds_file = os.path.join(datasetLocation, 'MEN_dataset_natural_form_full')
            with open(ds_file, 'r') as f:
                lines = [l.strip().split[' '] for l in f.readlines()]
                self.actual_values = {(l[0], l[1]): float(l[2]) for l in lines}
        elif dataset == 'simlex-999':
            ds_file = os.path.join(datasetLocation, 'Simlex-999.txt')
            with open(ds_file) as f:
                reader  = csv.DictReader(f,delimiter='\t')
                self.actual_values = {
                    (d['word1'],d['word2']):float(d['SimLex999'])
                    for d in reader
                }

    def fit(self, X=None, y=None):
        return self

    def _meaning(self, x):
        if x[0] in mmspace.space and x[1] in mmspace.space:
            if beta is None:
                return mmspace.space.sim(x[0], x[1], alpha=alpha)
            else:
                return mmspace.space.sim(x[0], x[1], alpha=alpha, beta=beta)
        else:
            return np.NaN

    def predict(self, X, y=None):
        return [self._meaning(x) for x in X]

    def score(self, X, y=None):
        y_pred = self.predict(X)
        predicted_dict = {tuple(X[i]): y_pred[i]
                            for i in range(len(y_pred))
                            if not np.isnan(y_pred[i])}
        predicted_values = np.array(predicted_dict.values())
        actual_values = np.array([self.actual_values[k]
                                for k in predicted_dict.keys()])
        self.test_pairs = len(predicted_values)
        return spearmanr(actual_values, predicted_values)
