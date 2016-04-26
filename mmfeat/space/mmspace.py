'''
Multi-modal models, with middle or late fusion and several combination methods.
'''

from .base import Space
from .sim import cosine, norm

import numpy as np

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

    def midSimFunc(self, x, y):
        if self.methodType == 'wmm':
            mmx = self.concat(self.lingSpace[x], self.visSpace[x])
            mmy = self.concat(self.lingSpace[y], self.visSpace[y])
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

    def lateSimFunc(self, x, y):
        lingSim = self.lingSpace.sim(x, y)
        visSim = self.visSpace.sim(x, y)

        if self.methodType == 'wmm':
            ret = self.alpha * lingSim + (1 - self.alpha) * visSim
        elif self.methodType == 'dfmm':
            if self.visSpace.dispersions[x] > self.filterThreshold and self.visSpace.dispersions[x] > self.filterThreshold:
                ret = self.alpha * lingSim + (1 - self.alpha) * visSim
            else:
                ret = lingSim
        elif self.methodType == 'dwmm':
            ret = self.visSpace.dispersions[x] * lingSim + (1 - self.visSpace.dispersions[y]) * visSim
        return ret
