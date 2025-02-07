import numpy as np
from numpy import ndarray
from scipy import special

def assert_same_shape(array: np.ndarray, array_grad: np.ndarray):
    assert array.shape == array_grad.shape, \
    '''
    Two ndarrays should have the same shape;
    instead, first ndarray's shape is {0}
    and second ndarray's shape is {1}.
    '''.format(tuple(array_grad.shape), tuple(array.shape))
    return None

def mae(y_true: ndarray, y_pred: ndarray):
    '''
    Compute mean absolute error for a neural network.
    '''    
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true: ndarray, y_pred: ndarray):
    '''
    Compute root mean squared error for a neural network.
    '''
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))

def softmax(x: np.ndarray, axis=None) -> np.ndarray:
    return np.exp(x - special.logsumexp(x, axis=axis, keepdims=True))

def normalize(a: np.ndarray):
    '''
    If a = np.array([[0.2], [0.5], [0.8]])

    then return np.array([[0.2, 0.8], 
                          [0.5, 0.5], 
                          [0.8, 0.2]])
    '''
    # Because probabilities [0,1], this creates a mirrored array
    other = 1 - a
    return np.concatenate([a, other], axis=1)

def unnormalize(a: np.ndarray):
    '''
    Reverse normalize method
    '''
    return a[np.newaxis, 0]

def permute_data(X: np.ndarray, y: np.ndarray):
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]