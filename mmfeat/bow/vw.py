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
        '''
        fname:      filename of the sound file we want to load
        '''
        if self.verbose: print('Loading %s' % fname)

        if self.cached:
            if not os.path.exists(fname + '-dsift.npy'):
                try:
                    img = imread(fname)
                    data = self.dsift.process_image(img)
                    np.save(fname + '-dsift.npy', data)
                except IOError:
                    return None
            else:
                try:
                    data = np.load(fname + '-dsift.npy')
                except:
                    return None
        else:
            try:
                img = imread(fname)
                data = self.dsift.process_image(img)
            except IOError:
                return None

        return data

    def loadMatlabFile(self, fname):
        '''
        fname:      filename of the sound file we want to load
        '''
        if not self.cached: raise ValueError('Can only read cached Matlab files')

        if self.verbose: print('Loading %s' % fname)

        try:
            data = loadmat(fname + '-dsift.mat')['descrs'].T
        except:
            return None

        return data

    def load(self, data_dir, cached=True):
        '''
        data_dir:   data directory containing an index.pkl file
        cached:     determines whether we cache MFCC descriptors to disk
        '''
        self.data_dir = data_dir
        self.cached = cached
        self.dsift = DsiftExtractor()
        self.data = DataObject(data_dir, self.loadFile)
        self.idx = self.data.idx
