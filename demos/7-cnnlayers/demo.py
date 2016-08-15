'''
This demo is an adaptation of 1-simrel and does the following:

    1. fetch images from the specified search engine
    2. get CNN image representations at different levels
    3. build visual spaces for each
    4. build multi-modal space
    5. evaluate on MEN and SimLex datasets

This is for demo purposes only. In practice it would be much
more efficient to transfer the relevant layers after a forward
pass all at the same time, instead of one by one.
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

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: python %s {google|bing}'
        quit()

    engine = sys.argv[1]

    useGPU = True # change to False if you want to use the GPU

    data_dir = '../1-simrel/demo-data-%s' % engine

    #
    # 0. Set up evaluation data
    #

    men = json.load(open('../1-simrel/men.json'))
    simlex = json.load(open('../1-simrel/simlex.json'))
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
        miner.getResults(unique_words, 10)
        miner.save()
    else:
        print('Image directory already exists..')

    #
    # 2. Get layer representations
    #
    model = CNN(modelType='alexnet', gpu=useGPU)
    print('Loading data..')
    model.load(data_dir)

    lkp = {}
    for layer in ['pool1', 'norm1', 'pool2', 'norm2', 'pool5', 'fc6', 'fc7']:
        if os.path.exists('%s/%s.pkl' % (data_dir, layer)):
            print('%s.pkl already exists - loading' % layer)
            lkp[layer] = pickle.load(open('%s/%s.pkl' % (data_dir, layer), 'rb'))
            continue

        print('Using layer %s' % layer)
        model.useLayer = layer

        print('Fitting..')
        model.fit()

        print('Building and saving visual lookup')
        lkp[layer] = model.toLookup()
        pickle.dump(lkp[layer], open('%s/%s.pkl' % (data_dir, layer), 'wb'))

    #
    # 3. Evaluate
    #
    print('Loading linguistic space')
    ls = Space('../1-simrel/simrel-wikipedia.pkl')
    for layer in ['pool5', 'fc6', 'fc7', 'pool5+fc6', 'pool5+fc6+fc7', 'fc6+fc7']:
        if layer not in lkp and '+' in layer:
            print('Computing %s' % layer)
            lkp[layer] = {}
            for term in model.idx:
                for l in layer.split('+'):
                    vs = AggSpace(lkp[l], 'max')
                    if term not in lkp[layer]:
                        lkp[layer][term] = (vs.space[term] / np.linalg.norm(vs.space[term]))
                    else:
                        lkp[layer][term] = np.hstack((lkp[layer][term], vs.space[term] / np.linalg.norm(vs.space[term])))
            pickle.dump(lkp[layer], open('%s/%s.pkl' % (data_dir, layer), 'wb'))
            vs = Space(lkp[layer]) # already have the means here
        else:
            vs = AggSpace(lkp[layer], 'max')

        print('Building multi-modal space')
        mm = MMSpace(ls, vs, modelType='middle', methodType='wmm', buildModel=True)

        print 'Evaluating using layer %s, engine %s. Results:' % (layer, engine)
        for name, dataset in {'MEN': men, 'SimLex': simlex}.items():
            r_s, p = ls.spearman(dataset)
            print 'Linguistic %s spearman=%.6f (p=%.4f)' % (name, r_s, p)

            r_s, p = vs.spearman(dataset)
            print 'Visual %s spearman=%.6f (p=%.4f)' % (name, r_s, p)

            r_s, p = mm.spearman(dataset)
            print 'Multi-modal %s spearman=%.6f (p=%.4f)' % (name, r_s, p)
