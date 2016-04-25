'''
Base
'''

import os
import cPickle as pickle

import copy_reg
import types

'''
Make it easier to share objects when multiprocessing
TODO: Check if this is still necessary
'''
def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    if func_name.startswith('__') and not func_name.endswith('__'):
        #deal with mangled names
        cls_name = cls.__name__.lstrip('_')
        func_name = '_%s%s' % (cls_name, func_name)
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    if obj and func_name in obj.__dict__:
        cls, obj = obj, None # if func_name is classmethod
    for cls in cls.__mro__:
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

'''
Simple data object wrapper which can be used instead of the normal dictionary,
with some extra functionality
'''
class DataObject(object):
    def __init__(self, datadir, loadFunc):
        self.datadir = datadir
        if datadir[-1] == '/': self.datadir = self.datadir[:-1]

        idxf = datadir + '/index.pkl'
        if not os.path.exists(idxf):
            raise ValueError('Index pickle not found')

        idx = pickle.load(open(idxf, 'rb'))
        fnames = set([])
        for k in idx:
            for fname in idx[k]:
                ### Backward compat with earlier versions of this code:
                if isinstance(fname, (long, int)):
                    fname = '%d.ogg' % (fname)
                if fname is None: continue
                fname = fname.split('/')[-1]
                ###

                fnames.add(fname)

        self.idx = idx
        self.fnames = list(fnames)
        self.loadFunc = loadFunc

    def keys(self):
        return self.fnames

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, fname):
        if fname not in self.fnames: raise IndexError('%s does not exist' % fname)
        return self.loadFunc('%s/%s' % (self.datadir, fname))
