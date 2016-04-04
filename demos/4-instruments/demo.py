'''
This demo shows how to get sound files for instruments and cluster them.

The example file lists two classes of instruments, so we cluster with k=2 after
having obtain BoAW representations for each of the instruments.
'''

from sklearn.metrics import v_measure_score
from sklearn.cluster import MiniBatchKMeans
from scipy.cluster.vq import vq

import numpy as np

import sys, os
sys.path.append('../..')
from mmfeat.bow import BoAW
from mmfeat.miner import FreeSoundMiner
from mmfeat.space import AggSpace

if __name__ == '__main__':
    data_dir = './sound_files'

    print('Loading eval data')
    instclass = {l[0]:l[1].rstrip('\n') for l in [line.split() for line in open('list_of_instruments.txt')]}
    instruments, classes = list(set(instclass.keys())), list(set(instclass.values()))

    print('Mining sound files')
    miner = FreeSoundMiner(data_dir, '../../miner.yaml')
    miner.getResults(instruments, 20)
    miner.save()

    print('Loading data and getting representations')
    model = BoAW(100)
    model.load(data_dir)
    model.fit()

    print('Building lookup')
    lkp = model.toLookup()

    print('Loading auditory space')
    asp = AggSpace(lkp, 'mean')

    the_data, labels_true = [], []
    for instrument in instruments:
        the_data.append(asp.space[instrument])
        labels_true.append(instclass[instrument])
    the_data = np.array(the_data)

    mbk = MiniBatchKMeans(n_clusters=len(classes), batch_size=2, verbose=True, compute_labels=True, max_iter=10000, n_init=25)
    mbk.fit(the_data)
    centroids = mbk.cluster_centers_
    #labels_pred, _ = vq(the_data, centroids)
    score = v_measure_score(labels_true, mbk.labels_)
    print 'V-measure:', score
    for instrument, label in zip(instruments, mbk.labels_):
        print('Instrument=%s,\tcluster=%d' % (instrument, label))
