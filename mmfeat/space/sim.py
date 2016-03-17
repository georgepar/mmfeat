'''
Similarity methods
'''

import numpy as np

def norm(x):
    return np.linalg.norm(np.nan_to_num(x))
def cosine(u, v):
    return np.dot(np.nan_to_num(u), np.nan_to_num(v)) / (norm(u) * norm(v))
