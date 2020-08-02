import numpy as np

def accuracy(y_pred, y_true):
    assert(y_true.shape == y_pred.shape)
    result = np.equal(y_true, y_pred)
    result = np.all(result, axis=1)
    return np.mean(result)
