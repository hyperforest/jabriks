import numpy as np

def linear(x, grad=False):
    if grad:
        return np.full_like(x, 1.)
    return x

def sigmoid(x, grad=False):
    if grad:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1.0 / (1 + np.exp(-x))

def tanh(x, grad=False):
    if grad:
        return 1 - np.square(tanh(x))
    return np.tanh(x)

def relu(x, grad=False):
    if grad:
        res = x.copy()
        res[res >= 0] = 1.
        res[res < 0] = 0.
        return res
    res = x.copy()
    res[res < 0] = 0.
    return res

def softmax(x, grad=False):
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)