'''
Multi-modal models, with middle or late fusion and several combination methods.
'''

from .base import Space
from .sim import cosine, norm

import csv
import os
import sys
import random

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats.stats import spearmanr
from scipy.stats.stats import pearsonr
import random

import cPickle as pickle
import json

from sklearn.decomposition import TruncatedSVD
import warnings
import math

warnings.filterwarnings("ignore")

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


class MMSpace(Space):
    def __init__(self, lingSpace, visSpace, modelType='middle', methodType='wmm', svd=None, alpha=0.5, modelDescription=''):
        self.modelDescription = modelDescription
        self.reportMissing = True
        self.alpha = alpha
        self.lingSpace = lingSpace
        self.visSpace = visSpace
        self.modelType = modelType
        self.setMethodType(methodType)
        self.setModelType(modelType, svd)

    def setParams(self, alpha=None):
        if alpha is not None:
            self.alpha = alpha

    def setModelType(self, modelType, svd=None):
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

            if svd is not None:
                self.space = {}
                y = []
                X = []
                for k in self.lingSpace.keys():
                    if k in self.visSpace.keys():
                        y.append(k)
                        mm_vector = self.concat(self.lingSpace[k], self.visSpace[k])
                        X.append(mm_vector)
                svd  = TruncatedSVD(n_components=svd)
                reduced = svd.fit_transform(X)
                for i in range(len(y)):
                    self.space[y[i]] = X[i]
            else: # or override similarity function
                self.sim = self.midSimFunc
        elif modelType == 'late':
            self.sim = self.lateSimFunc

    def __contains__(self, key):
        return key in self.lingSpace and key in self.visSpace

    def concat(self, u, v, alpha=None):
        if alpha is None:
            alpha = self.alpha

        u /= norm(u)
        v /= norm(v)
        concatenated = np.hstack((alpha * u, (1 - alpha) * v))
        return concatenated

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
    def __init__(self, lingSpace, visSpace, audioSpace, svd=None, modelType='middle', alpha=0.33, beta=0.33, sensicon=None, modelDescription=''):
        self.modelDescription = modelDescription
        self.reportMissing = True
        self.alpha = alpha
        self.beta = beta
        self.p = 0.1
        self.sensicon = None
        if sensicon is not None:
            if isinstance(sensicon, str):
                if sensicon.endswith('pkl'):
                    with open(sensicon, 'rb') as f:
                        self.sensicon = pickle.load(f)
                elif sensicon.endswith('json'):
                    with open(sensicon, 'r') as f:
                        self.sensicon = json.load(f)
            elif isinstance(sensicon, dict):
                self.sensicon = sensicon
            else:
                raise TypeError('Expecting file name or dictionary of Sensicon')

        self.lingSpace = lingSpace
        self.visSpace = visSpace
        self.audioSpace = audioSpace
        self.modelType = modelType
        self.setModelType(modelType, svd=None)

    def setParams(self, alpha=None, beta=None, p=None):
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if p is not None:
            self.p = p

    def setModelType(self, modelType, svd=None):
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
#            if svd is not None:
#                self.space = {}
#                y = []
#                X = []
#                for k in self.lingSpace.keys():
#                    if k in self.visSpace.keys():
#                        y.append(k)
##                        mm_vector = self.concat(self.lingSpace[k], self.visSpace[k])
#                        X.append(mm_vector)
#                svd  = TruncatedSVD(n_components=svd)
#                reduced = svd.fit_transform(X)
#                for i in range(len(y)):
#                    self.space[y[i]] = X[i]
#            else:
        elif modelType == 'late':
            self.sim = self.lateSimFunc

    def __contains__(self, key):
        return key in self.lingSpace and key in self.visSpace and key in self.audioSpace

    def adapt_weights(self, word, alpha):
        vis_coeff = self.sensicon[word]['visual_score']
        aud_coeff = self.sensicon[word]['audio_score']

        if aud_coeff < 0.001 and vis_coeff < 0.001:
            beta = 0.5 * (1 - alpha)
 
        if vis_coeff < 0.001:
            beta = 0
            return beta
 
        if aud_coeff < 0.001:
            beta = 1 - alpha
            return beta
        ratio_coeff = (float(vis_coeff) / aud_coeff) ** (1/2.0)
        beta = (ratio_coeff / (1 + ratio_coeff)) * (1 - alpha)
        return beta

    def adapt_weights_small(self, word1, alpha, p, word2=None):
        vis_coeff1 = self.sensicon[word1]['visual_score']
        aud_coeff1 = self.sensicon[word1]['audio_score']
        if word2 is not None:
            vis_coeff2 = self.sensicon[word2]['visual_score']
            aud_coeff2 = self.sensicon[word2]['audio_score']        
        else:
            vis_coeff2, aud_coeff2 = 1000, 1000

#        if aud_coeff1 + aud_coeff2 < 0.00001 and vis_coeff1 + aud_coeff2 < 0.00001:
#            beta = 0.5 * (1 - alpha)
# 
#        if vis_coeff1 + vis_coeff2 < 0.00001:
#            beta = 0
#            return beta
 
#        if aud_coeff1 + aud_coeff2 < 0.00001:
#            beta = 1 - alpha
#            return beta
        coeff = (1.0 - alpha) / 2.0
        beta = coeff * (1.0 + p) if min(aud_coeff1, aud_coeff2) < min(vis_coeff1, vis_coeff2) else coeff * (1.0 - p)
        return beta


    def adapt_weights_full(self, word1, word2=None, alpha=0.5):
        vis_coeff1 = self.sensicon[word1]['visual_score']
        aud_coeff1 = self.sensicon[word1]['audio_score']
        if word2 is not None:
            vis_coeff2 = self.sensicon[word2]['visual_score']
            aud_coeff2 = self.sensicon[word2]['audio_score']        
        else:
            vis_coeff2, aud_coeff2 = 0, 0
        beta = sigmoid((vis_coeff1 + vis_coeff2) / (aud_coeff1 + aud_coeff2))
        return beta

    def adapt_weights_late(self, word1, word2, alpha):
        vis_coeff1 = self.sensicon[word1]['visual_score']
        aud_coeff1 = self.sensicon[word1]['audio_score']
        vis_coeff2 = self.sensicon[word2]['visual_score']
        aud_coeff2 = self.sensicon[word2]['audio_score']        
        if (aud_coeff1 + aud_coeff2) < 0.001 and (vis_coeff1 + vis_coeff2) < 0.001:
            beta = 0.5 * (1 - alpha)
            return beta
 

        if aud_coeff1 < 0.000001 or aud_coeff2 < 0.000001:
            beta = 1 - alpha
            return beta
        ratio_coeff = (float(vis_coeff1 + vis_coeff2) / (aud_coeff1 + aud_coeff2)) ** (1/2.0)
        beta = (ratio_coeff / (1 + ratio_coeff)) * (1 - alpha)
        return beta

    def adapt_weights2(self, word, alpha):
        vis_coeff = self.sensicon[word]['visual_score']
        aud_coeff = self.sensicon[word]['audio_score']
        diff_coeff = float(vis_coeff) - aud_coeff
        if abs(diff_coeff) < 0.1:
            beta = 0.5 * (1 - alpha)
        elif diff_coeff > 0:
            beta = 1 - alpha
        else:
            beta = 0
        return beta

    def adapt_weights4(self, word, alpha):
        beta = random.uniform(0, 1)
        return beta


    def adapt_weights3(self, word, alpha, beta):
        vis_coeff = self.sensicon[word]['visual_score']
        aud_coeff = self.sensicon[word]['audio_score']
        beta = beta * vis_coeff
        gamma = (1 - alpha - beta) * aud_coeff
        return beta, gamma

    def concat(self, u, v, w, alpha=None, beta=None, gamma=None):
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta
        if gamma is None: 
            gamma = 1 - alpha - beta

        u /= norm(u)
        v /= norm(v)
        w /= norm(w)
        return np.hstack((alpha * u,  beta * v, gamma * w))

    def midSimFunc(self, x, y, alpha=None, beta=None, p=None):
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta
        if p is None:
            p = self.p
        if self.sensicon is not None:
            beta_x = self.adapt_weights(x, alpha)
            beta_y = self.adapt_weights(y, alpha)
            mmx = self.concat(self.lingSpace[x], self.visSpace[x], self.audioSpace[x], alpha=alpha, beta=beta_x)
            mmy = self.concat(self.lingSpace[y], self.visSpace[y], self.audioSpace[y], alpha=alpha, beta=beta_y)
        else:
            mmx = self.concat(self.lingSpace[x], self.visSpace[x], self.audioSpace[x], alpha=alpha, beta=beta)
            mmy = self.concat(self.lingSpace[y], self.visSpace[y], self.audioSpace[y], alpha=alpha, beta=beta)
        mmx /= norm(mmx)
        mmy /= norm(mmy)
        return cosine(mmx, mmy)

    def lateSimFunc(self, x, y, alpha=None, beta=None, p=None):
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta
        if p is None:
            p = self.p
        gamma = 1 - alpha - beta
        if self.sensicon is not None:
            beta = self.adapt_weights_late(x,y, alpha)
            gamma = 1 - alpha - beta
        lingSim = self.lingSpace.sim(x, y)
        visSim = self.visSpace.sim(x, y)
        audioSim = self.audioSpace.sim(x, y)
        return alpha * lingSim + beta * visSim + gamma * audioSim


class MMEstimator(BaseEstimator):
    def __init__(self, mmspace, alpha=0.5, beta=None):
        self.mmspace = mmspace
        self.alpha = alpha
        self.beta = beta
        self.actual_values = {}
        self.test_pairs = 0

    # TODO: Get this code out of here
    def loadDataset(self, datasetLocation='/home/geopar/projects/mmfeat/datasets', dataset='men'):
        if dataset == 'men':
            ds_file = os.path.join(datasetLocation, 'MEN_dataset_natural_form_full')
            with open(ds_file, 'r') as f:
                lines = [l.strip().split(' ') for l in f.readlines()]
                self.actual_values = {(l[0], l[1]): float(l[2]) for l in lines}
        elif dataset == 'men-reduced':
            ds_file = os.path.join(datasetLocation, 'MEN_dataset_natural_form_reduced')
            with open(ds_file, 'r') as f:
                lines = [l.strip().split(' ') for l in f.readlines()]
                self.actual_values = {(l[0], l[1]): float(l[2]) for l in lines}
        elif dataset == 'simlex-999':
            ds_file = os.path.join(datasetLocation, 'SimLex-999.txt')
            with open(ds_file) as f:
                reader  = csv.DictReader(f,delimiter='\t')
                self.actual_values = {
                    (d['word1'],d['word2']):float(d['SimLex999'])
                    for d in reader
                }
                print(self.actual_values.values())

    def fit(self, X=None, y=None):
        return self

    def _meaning(self, x):
        if x[0] in self.mmspace and x[1] in self.mmspace:
            if self.beta is None:
                return self.mmspace.sim(x[0], x[1], alpha=self.alpha)
            else:
                return self.mmspace.sim(x[0], x[1], alpha=self.alpha, beta=self.beta)
        else:
            return np.NaN

    def predict(self, X, y=None):
        return [self._meaning(x) for x in X]

    def score(self, X, y):
        ground_truth = [[X[i][0], X[i][1], y[i]] for i in range(len(y))]
        kwargs = {}
        if self.alpha is not None:
            kwargs['alpha'] = self.alpha
        if self.beta is not None:
            kwargs['beta'] = self.beta
        corr, _ = self.mmspace.spearman(ground_truth, **kwargs)
        return corr
