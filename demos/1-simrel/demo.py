'''
This demo does the following:

    1. fetch images from the specified search engine
    2. get image representations using the specified method
    3. build visual space from image representations
    4. build multi-modal space
    5. evaluate on MEN and SimLex datasets

'''

import json
import os
import sys

sys.path.append('../..')

from mmfeat.miner import *
from mmfeat.bow import *
from mmfeat.cnn import *
from mmfeat.space import *

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Usage: python %s {google|bing} {bovw|cnn}'
        quit()

    engine = sys.argv[1]
    method = sys.argv[2]

    useGPU = False # change to True if you want to use the GPU

    data_dir = './demo-data-%s' % engine

    #
    # 0. Set up evaluation data
    #
    men = json.load(open('men.json'))
    simlex = json.load(open('simlex.json'))
    words = []
    for [w1, w2, _] in men + simlex:
        words.extend([w1,w2])
    unique_words = list(set(words))

    #
    # 1. Fetch images
    #
    print('Fetching images..')
    if not os.path.exists(data_dir):
        if engine == 'bing':
            miner = BingMiner(data_dir, '../../miner.yaml')
        elif engine == 'google':
            miner = GoogleMiner(data_dir, '../../miner.yaml')
        elif engine == 'freesound':
            miner = FreeSoundMiner(data_dir, '../../miner.yaml')
        miner.getResults(unique_words, 10)
        miner.save()
    else:
        print 'Image directory already exists..'

    #
    # 2. Get image representations
    #
    if method == 'bovw':
        model = BoVW(100, subsample=0.1)
    elif method == 'cnn':
        model = CNN(modelType='alexnet', gpu=useGPU)

    print('Loading data..')
    model.load(data_dir)
    print('Fitting..')
    model.fit()

    #
    # 3. Build visual and linguistic spaces
    #
    print('Building visual lookup')
    lkp = model.toLookup()

    print('Loading visual space')
    vs = AggSpace(lkp, 'mean')

    print('Loading linguistic space')
    ls = Space('simrel-ling.pkl')

    #
    # 4. Construct multi-modal space
    #
    print('Building multi-modal space')
    mm = MMSpace(ls, vs)

    #
    # 5. Run evaluations
    #
    print 'Evaluating using engine %s, method %s. Results:' % (engine, method)
    for name, dataset in {'MEN': men, 'SimLex': simlex}.items():
        r_s, p = ls.spearman(dataset)
        print 'Linguistic %s spearman=%.4f (p=%.4f)' % (name, r_s, p)

        r_s, p = vs.spearman(dataset)
        print 'Visual %s spearman=%.4f (p=%.4f)' % (name, r_s, p)

        r_s, p = mm.spearman(dataset)
        print 'Multi-modal %s spearman=%.4f (p=%.4f)' % (name, r_s, p)
