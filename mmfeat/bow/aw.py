'''
Bag of Audio Words (BoAW)
'''

from ..base import DataObject

from .bow import BoW

import os

import numpy as np

try:
    import librosa
    from librosa.feature import mfcc
except ImportError:
    import warnings
    warnings.warn('Could not find librosa. You will not be able to use the BoAW model.')

class BoAW(BoW):
    def loadFile(self, fname):
        '''
        fname:      filename of the sound file we want to load
        '''
        if self.verbose: print('Loading %s' % fname)

        if self.cached:
            if not os.path.exists(fname + '-mfcc.npy'):
                y, sr = librosa.load(fname)
                data = mfcc(y=y, sr=sr).T
                np.save(fname + '-mfcc.npy', data)
            else:
                data = np.load(fname + '-mfcc.npy')
        else:
            y, sr = librosa.load(fname)
            # TODO: Add ability to filter by seconds/duration
            # seconds = y.size/sr
            data = mfcc(y=y, sr=sr).T

        return data

    def load(self, data_dir, cached=True):
        '''
        data_dir:   data directory containing an index.pkl file
        cached:     determines whether we cache MFCC descriptors to disk
        '''
        self.data_dir = data_dir
        self.cached = cached
        self.data = DataObject(data_dir, self.loadFile)
        self.idx = self.data.idx
