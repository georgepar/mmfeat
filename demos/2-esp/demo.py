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
    esp_dir = '../../../data/'
    data_dir = '../../../data/ESPGame100k/'

    #
    # 1. Fetch the ESP Game dataset
    #
    if not os.path.exists('esp.tgz'):
        print('Fetching the ESP Game dataset')
        sources = ['http://server251.theory.cs.cmu.edu/ESPGame100k.tar.gz',
            'http://hunch.net/~learning/ESP-ImageSet.tar.gz']
        for source in sources:
            success = os.system('wget %s -O esp.tgz' % (source))
            print success
            if success == 0: # exit code 0 = no problems occurred
                break
        if success != 0:
            raise ValueError('Could not download ESP Game dataset - please find another source for the data')

    print('Unpacking ESP game dataset')
    os.system('tar xvzf esp.tgz -C %s/..' % (esp_dir)) # unpacking one up from actualy data dir

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
    lkp = model.toLookup()

    #
    # 4. Build and save visual representations by taking the
    #    mean of the image representations associated with each
    #    label.
    #
    print('Loading visual space')
    vs = AggSpace(lkp, 'mean')

    print('Saving visual representations in descriptors.pkl')
    pickle.dump(vs.space, open('%s/descriptors.pkl' % data_dir, 'wb'))
