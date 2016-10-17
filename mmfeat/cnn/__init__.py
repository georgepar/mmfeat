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
    def __init__(self, caffe_root=None, modelType='alexnet', modelLocation=None, gpu=False, gpuid=0, verbose=True, n_workers=12):
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
        self.modelLocation = modelLocation
        if self.modelType == 'custom' and modelLocation is None:
            raise ValueError('Selected custom model typed by failed to specify its location')

        self.useGPU = gpu
        self.gpuid = gpuid
        self.verbose = verbose
        self.n_workers = n_workers

        if self.useGPU:
            caffe.set_device(self.gpuid)
            caffe.set_mode_gpu()

        if modelType == 'caffenet':
            mmfeat_caffe_net = caffe.Net(self.caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                self.caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel', caffe.TEST)
            self.useLayer = 'fc7'
        elif modelType == 'alexnet':
            mmfeat_caffe_net = caffe.Net(self.caffe_root + 'models/bvlc_alexnet/deploy.prototxt',
                self.caffe_root + 'models/bvlc_alexnet/bvlc_alexnet.caffemodel', caffe.TEST)
            self.useLayer = 'fc7'
        elif modelType == 'vgg':
            mmfeat_caffe_net = caffe.Net(self.caffe_root + 'models/vgg/VGG_ILSVRC_19_layers_deploy.prototxt',
               self.caffe_root + 'models/vgg/VGG_ILSVRC_19_layers.caffemodel', caffe.TEST)
            self.useLayer = 'fc7'
        elif modelType == 'googlenet':
            mmfeat_caffe_net = caffe.Net(self.caffe_root + 'models/bvlc_googlenet/deploy.prototxt',
               self.caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel', caffe.TEST)
            self.useLayer = 'pool5/7x7_s1'
        if modelType == 'custom':
            # assume we have a deploy in the same dir and that we want fc7
            mmfeat_caffe_net = caffe.Net(self.caffe_root + os.path.dirname(self.modelLocation) + '/deploy.prototxt',
                self.caffe_root + self.modelLocation, caffe.TEST)
            self.useLayer = 'fc7'

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

    def load(self, datadir):
        self.data = DataObject(datadir, self.loadFile)
        self.idx = self.data.idx

    def loadFile(self, fname):
        #if self.verbose: print('Loading %s' % fname)
        if self.modelType == 'custom':
            if not os.path.exists(fname):
                fname = fname[:-3] + 'png'

        try:
            image = caffe.io.load_image('%s' % fname)
            return self.transformer.preprocess('data', image)
        except (IOError, ValueError) as e:
            if self.verbose: print('%s - error' % fname)
            return None

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

    def toLookup(self, n_files=None):
        lkp = {}
        for key in self.idx:
            lkp[key] = {}

            # in case we only want to take a subset
            fnames = self.idx[key]
            if n_files is not None:
                fnames = fnames[:n_files]

            for fname in fnames:
                if fname is None: continue
                fname = str(fname).split('/')[-1]
                if fname not in self.descriptors: continue
                lkp[key][fname] = self.descriptors[fname]
        return lkp
