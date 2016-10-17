'''
Tool for extracting multi-modal features from data folders obtained using one of the miners
'''
import argparse
import sys, os
import json, csv
import cPickle as pickle

from mmfeat.bow import *
from mmfeat.cnn import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', nargs=1, \
        help='type of model to use for feature extraction', choices=['boaw', 'bovw', 'cnn'])
    parser.add_argument('data_dir', nargs=1, \
        help='location of directories')
    parser.add_argument('out_file', nargs=1, \
        help='file to store extracted features in (default will be a Python pickle, overwrite with -o)')
    parser.add_argument('-gpu', action='store_true', \
        help='use GPU for CNNs (default False)', default=False)
    parser.add_argument('-storedescriptors', action='store_true', \
        help='store descriptors as separate pickle for later use', default=False)
    parser.add_argument('-gpuid', type=int, action='store', \
        help='GPU id for CNNs (default 0)', default=0)
    parser.add_argument('-k', type=int, action='store', \
        help='number of dimensions in descriptors for bag-of-words (default 100)', default=100)
    parser.add_argument('-c', '--centroids', action='store', \
        help='pre-load centroids for bag-of-words (default None)', default=None)
    parser.add_argument('-o', '--output', action='store', \
        help='type of output file (default pickle)', choices=['pickle', 'json', 'csv'], default='pickle')
    parser.add_argument('-s', '--sample_files', type=float, action='store', \
        help='fraction of files to sample for clustering bag-of-words (default None, range 0-1)', default=None)
    parser.add_argument('-n', '--n_files', type=int, action='store', \
        help='number of files to use when creating lookup')
    parser.add_argument('-m', '--modelType', action='store', \
        help='type of CNN model to use (default alexnet)', default='alexnet', choices=['vgg', 'alexnet', 'caffenet', 'googlenet', 'custom'])
    parser.add_argument('-ml', '--modelLocation', action='store', \
        help='location of CNN model if custom modelType')
    parser.add_argument('-v', '--verbose', action='store_true', \
        help='verbosity (default True)', default=True)
    args = parser.parse_args()

    args.model = args.model[0]
    args.data_dir = args.data_dir[0]
    args.out_file = args.out_file[0]

    if not os.path.exists(args.data_dir): raise ValueError('Data directory does not exist')
    if os.path.exists(args.out_file): raise ValueError('Out file already exists - please remove manually')

    if args.model == 'boaw':
        model = BoAW(args.k, subsample=args.sample_files)
    elif args.model == 'bovw':
        model = BoVW(args.k, subsample=args.sample_files)
    elif args.model == 'cnn':
        model = CNN(modelType=args.modelType, modelLocation=args.modelLocation, gpu=args.gpu, gpuid=args.gpuid)

    print('Loading..')
    model.load(args.data_dir)

    if args.centroids is not None:
        print('Pre-loading centroids.. from %s' % args.centroids)
        model.centroids = pickle.load(open(args.centroids, 'rb'))

    print('Fitting..')
    model.fit()

    if args.storedescriptors:
        print('Saving descriptors to %s-descriptors.pkl' % args.out_file)
        pickle.dump(model.descriptors, open(args.out_file + '-descriptors.pkl', 'wb'))

    print('Building lookup')
    lkp = model.toLookup(n_files=args.n_files)

    print('Saving.. to %s' % args.out_file)
    if args.output == 'pickle':
        pickle.dump(lkp, open(args.out_file, 'wb'))
    elif args.output == 'json':
        json.dump(lkp, open(args.out_file, 'wb'))
    elif args.output == 'csv':
        with open(args.out_file, 'wb') as csvfile:
            csvw = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            for key in lkp:
                for fname in lkp[key]:
                    csvw.writerow([key, fname] + lkp[key][fname].tolist())

    if args.model in ['boaw', 'bovw']:
        print('Saving centroids.. to %s/centroids.pkl' % model.data_dir)
        pickle.dump(model.centroids, open(model.data_dir + '/centroids.pkl', 'wb'))
