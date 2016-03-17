'''
Bag of Audio Words (BoAW)
'''

from ..base import DataObject

from .bow import BoW

import os

import numpy as np

import librosa
from librosa.feature import mfcc

class BoAW(BoW):
    def loadFile(self, fname):
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
            # TODO: Add ability to filter by seconds
            # seconds = y.size/sr
            data = mfcc(y=y, sr=sr).T

        return data

    def load(self, data_dir, cached=True):
        self.data_dir = data_dir
        self.cached = cached
        self.data = DataObject(data_dir, self.loadFile)
        self.idx = self.data.idx
