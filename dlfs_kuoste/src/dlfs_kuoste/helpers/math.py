import numpy as np
from scipy import special

def assert_same_shape(array: np.ndarray, array_grad: np.ndarray):
    assert array.shape == array_grad.shape, \
    '''
    Two ndarrays should have the same shape;
    instead, first ndarray's shape is {0}
    and second ndarray's shape is {1}.
    '''.format(tuple(array_grad.shape), tuple(array.shape))
    return None

def softmax(x: np.ndarray, axis=None) -> np.ndarray:
    return np.exp(x - special.logsumexp(x, axis=axis, keepdims=True))

def normalize(a: np.ndarray):
    other = 1 - a
    return np.concatenate([a, other], axis=1)

def unnormalize(a: np.ndarray):
    return a[np.newaxis, 0]

def permute_data(X: np.ndarray, y: np.ndarray):
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]