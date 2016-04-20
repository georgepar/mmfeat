'''
Convolutional Neural Network features
'''

from ..base import DataObject

import os, sys, traceback
import cPickle as pickle
from multiprocessing import Pool, Manager

import numpy as np

try:
    import caffe
except ImportError:
    import warnings
    warnings.warn('Could not find Caffe Python bindings. You will not be able to use the CNN models.')

mmfeat_caffe_net = None # needs to be global because of multiprocessing

class CNN(object):
    def __init__(self, caffe_root=None, modelType='alexnet', gpu=False, gpuid=0, verbose=True, n_workers=12):
        global mmfeat_caffe_net

        if caffe_root is None:
            local_caffe_root = os.getenv('CAFFE_ROOT_PATH')
            if local_caffe_root is None:
                print('Please set the CAFFE_ROOT_PATH environment variable to the Caffe root folder')
                print('For example:')
                print('$ export CAFFE_ROOT_PATH=/home/user/caffe')
                print('Or add it to your ~/.bashrc')
                quit()
            if ':' in local_caffe_root:
                # if we have multiple paths for Caffe, pick the first existing one
                dirs = local_caffe_root.split(':')
                for dir in dirs:
                    if os.path.exists(dir):
                        local_caffe_root = dir
            self.caffe_root = local_caffe_root
        else:
            self.caffe_root = caffe_root

        self.modelType = modelType
        self.useGPU = gpu
        self.gpuid = gpuid
        self.verbose = verbose
        self.n_workers = n_workers

        if self.useGPU:
            caffe.set_device(self.gpuid)
            caffe.set_mode_gpu()

        if modelType == 'alexnet':
            mmfeat_caffe_net = caffe.Net(self.caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                self.caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel', caffe.TEST)
        elif modelType == 'vgg':
            mmfeat_caffe_net = caffe.Net(self.caffe_root + 'models/vgg/VGG_ILSVRC_19_layers_deploy.prototxt',
               self.caffe_root + 'models/vgg/VGG_ILSVRC_19_layers.caffemodel', caffe.TEST)

        # standard imagenet data transformer
        transformer = caffe.io.Transformer({'data': mmfeat_caffe_net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_mean('data', np.load(self.caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2,1,0))

        if self.useGPU:
            if self.verbose: print('We are in GPU mode.')
            mmfeat_caffe_net.forward()
            self.descriptors = {}
        else:
            manager = Manager()
            self.descriptors = manager.dict()

        self.transformer = transformer

        self.useLayer = 'fc7'

    def load(self, datadir):
        self.data = DataObject(datadir, self.loadFile)
        self.idx = self.data.idx

    def loadFile(self, fname):
        #if self.verbose: print('Loading %s' % fname)
        image = caffe.io.load_image('%s' % fname)
        return self.transformer.preprocess('data', image)

    # forward pass in the network
    def forward(self, data, fname):
        global mmfeat_caffe_net
        try:
            # load image
            mmfeat_caffe_net.blobs['data'].data[...] = data
            # forward pass
            out = mmfeat_caffe_net.forward()
            # extract relevant layer (default 'fc7')
            layer = mmfeat_caffe_net.blobs[self.useLayer].data[0].flatten()
            # add to descriptors
            self.descriptors[fname] = layer.copy()
            if self.verbose: print('%s - done' % fname)
        except:
            traceback.print_exc(file=sys.stdout)

    def fit(self, data=None):
        if data is not None:
            self.data = data

        if self.useGPU:
            for fname in self.data.keys():
                self.forward(self.data[fname], fname)
        else:
            p = Pool(processes=self.n_workers)
            for fname in self.data.keys():
                p.apply_async(self.forward, args=(self.data[fname], fname, ))
            p.close()
            p.join()

    def toLookup(self):
        lkp = {}
        for key in self.idx:
            lkp[key] = {}
            for fname in self.idx[key]:
                fname = fname.split('/')[-1]
                if fname not in self.descriptors: continue
                lkp[key][fname] = self.descriptors[fname]
        return lkp
