'''
This demo illustrates how you can interface with Matlab by generating local
descriptors using e.g. VLFeat and saving them with the appropriate path,
which can then be used in BoVW.

See run_dsift.m for Matlab example code to get descriptors. In this demo
we assume that you have already run this on the images in the example directory.
'''

import sys
sys.path.append('../..')
from mmfeat.bow import *
from mmfeat.space import AggSpace

if __name__ == '__main__':
    model = BoVW(25)
    model.loadFile = model.loadMatlabFile

    data_dir ='./exampledir'

    print('Loading data..')
    model.load(data_dir)

    print('Fitting..')
    model.fit()

    for (fn, descriptor) in model.descriptors.items():
        print(fn, descriptor)

    print('Building visual lookup')
    lkp = model.toLookup()

    print('Loading visual space')
    vs = AggSpace(lkp, 'mean')

    print(vs.space)
