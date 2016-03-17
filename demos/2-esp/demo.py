'''
This demo does the following:

    1. download and unpack the ESP Game dataset
        c.f. https://en.wikipedia.org/wiki/ESP_game
    2. build an index from the labels and thumbnails
    3. learn image representations
    4. save visual representations that you can use wherever you like
        (i.e., the means of image representations associated with label)

The purpose of the demo is to show you that you don't need to use a miner.
All you need is a directory of images and associated labels, from which
you can build an index file in the correct format.
'''

import json
import os
import sys

sys.path.append('../..')

from mmfeat.bow import *
from mmfeat.cnn import *
from mmfeat.space import *

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: python %s {bovw|cnn}'
        quit()

    method = sys.argv[1]

    #
    # 1. Fetch the ESP Game dataset
    #
    print('Fetching the ESP Game dataset')
    os.system('wget http://server251.theory.cs.cmu.edu/ESPGame100k.tar.gz')
    os.system('tar xvzf ESPGame100k.tar.gz')

    data_dir = '../../../data/ESPGame100k/'

    #
    # 2. If the index doesn't exist, build it and save it to the data_dir
    #
    if not os.path.exists('%s/thumbnails/index.pkl' % data_dir):
        print('Building index, this might take a while..')
        lookup = {}
        files = os.listdir('%s/labels/' % data_dir)
        for file in files:
            labels = open('%s/labels/%s' % (data_dir, file)).read().splitlines()
            for label in labels:
                if label not in lookup: lookup[label] = []
                # store the filename, minus the '.desc' at the end
                lookup[label].append('%s/thumbnails/%s' % (data_dir, file[:-5]))
        pickle.dump(lookup, open('%s/thumbnails/index.pkl' % data_dir, 'wb'))

    #
    # 3. Build the model and get image representations
    #
    if method == 'bovw':
        model = BoVW(50, subsample=0.01)
    elif method == 'cnn':
        model = CNN()

    print('Loading data..')
    model.load('%s/thumbnails/' % data_dir)

    print('Fitting..')
    model.fit()

    print('Building visual lookup')
    lkp = {}
    for key in model.idx:
        lkp[key] = {}
        for fname in model.idx[key]:
            fname = fname.split('/')[-1]
            if fname not in model.descriptors: continue
            lkp[key][fname] = model.descriptors[fname]

    #
    # 4. Build and save visual representations by taking the
    #    mean of the image representations associated with each
    #    label.
    #
    print('Loading visual space')
    vs = AggSpace(lkp, 'mean')

    print('Saving visual representations in descriptors.pkl')
    pickle.dump(vs.space, open('%s/descriptors.pkl' % data_dir, 'wb'))
