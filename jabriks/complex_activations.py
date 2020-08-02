from .real_activations import sigmoid as sig
from .real_activations import tanh, relu

import numpy as np

def amin_murase2(z, grad=False):
    x = z.real
    y = z.imag
    
    if grad:
        res = 2 * (sig(x) - sig(y))
        res = res * (sig(x, grad=True) - 1.0j * sig(y, grad=True))
        return res

    return np.square(sig(x) - sig(y))

def split(f, z, grad=False):
    return f(z.real, grad) + 1.0j * f(z.imag, grad)

def split_tanh(z, grad=False):
    return split(tanh, z, grad)

def split_relu(z, grad=False):
    return split(relu, z, grad)