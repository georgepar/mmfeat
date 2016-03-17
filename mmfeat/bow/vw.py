'''
Bag of Visual Words (BoVW)
'''

from ..base import DataObject

from .bow import BoW
from .dsift import DsiftExtractor

import os

import numpy as np

from scipy.misc import imread
from scipy.io import loadmat

class BoVW(BoW):
    def loadFile(self, fname):
        if self.verbose: print('Loading %s' % fname)

        if self.cached:
            if not os.path.exists(fname + '-dsift.npy'):
                img = imread(fname)
                data = self.dsift.process_image(img)
                np.save(fname + '-dsift.npy', data)
            else:
                data = np.load(fname + '-dsift.npy')
        else:
            img = imread(fname)
            data = self.dsift.process_image(img)

        return data

    def loadMatlabFile(self, fname):
        if not self.cached: raise ValueError('Can only read cached Matlab files')

        if self.verbose: print('Loading %s' % fname)

        data = loadmat(fname + '-dsift.mat')['descrs'].T
        return data

    def load(self, data_dir, cached=True):
        self.data_dir = data_dir
        self.cached = cached
        self.dsift = DsiftExtractor()
        self.data = DataObject(data_dir, self.loadFile)
        self.idx = self.data.idx
