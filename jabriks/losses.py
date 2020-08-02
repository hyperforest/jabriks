import numpy as np

def _check_shape(y_pred, y_true):
    assert(y_pred.ndim == 2)
    assert(y_pred.shape[1] == 1)    
    assert(y_true.ndim == 2)
    assert(y_true.shape[1] == 1)
    assert(y_pred.shape[0] == y_pred.shape[0])

def sum_squared_error(y_pred, y_true, grad=False):
    _check_shape(y_pred, y_true)
    
    if grad:
        return (y_pred - y_true)
    return 0.5 * np.sum(np.square(y_pred - y_true))

def binary_crossentropy(y_pred, y_true, grad=False):
    _check_shape(y_pred, y_true)
    assert(y_pred.shape[0] == 1)
    
    if grad:
        return ((1 - y_true) / (1 - y_pred + 1e-9)) - (y_true / (y_pred + 1e-9))
    return - y_true * np.log(y_pred + 1e-9) - (1 - y_true) * np.log(1 - y_pred + 1e-9)

def categorical_crossentropy(y_pred, y_true, grad=False):
    _check_shape(y_pred, y_true)
    
    if grad:
        return (- y_true / (y_pred + 1e-9))
    return - np.sum(y_true * np.log(y_pred + 1e-9))
