'''
This demo shows how you can download images from ImageNet and then get
their representations for use in multi-modal models.

This demo requires NLTK (http://www.nltk.org/index.html) and the WordNet
corpus (see http://www.nltk.org/data.html).
'''

import json
import os
import sys

import cPickle as pickle

sys.path.append('../..')

from mmfeat.miner import *
from mmfeat.bow import *
from mmfeat.cnn import *
from mmfeat.space import *
from mmfeat.space.sim import *

if __name__ == '__main__':
    useGPU = True # change to False if you want to use the GPU

    data_dir = 'demo-data-imagenet'

    words = ['falcon-N', 'owl-N', 'television-N']

    if not os.path.exists(data_dir):
        miner = ImageNetMiner(data_dir, '../../miner.yaml')
        miner.getResults(words, 50)
        miner.save()
    else:
        print('Image directory already exists..')

    model = CNN(modelType='alexnet', gpu=useGPU, n_workers=32)
    model.load(data_dir)
    model.fit()

    print('Converting to lookup..')
    lkp = model.toLookup(n_files=10)

    print('Building space..')
    vs = AggSpace(lkp, 'mean')

    print('A falcon is more similar to an owl than to a television')
    print('Cosine(falcon, owl) = %.4f' % cosine(vs['falcon-N'], vs['owl-N']))
    print('Cosine(falcon, television) = %.4f' % cosine(vs['falcon-N'], vs['television-N']))
