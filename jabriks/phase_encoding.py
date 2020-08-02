from .utils import cis
from .real_activations import sigmoid
import numpy as np

def linear(x):
    assert(x.ndim == 1)
    return x

def amin_murase(x):
    assert(x.ndim == 1)
    a = np.min(x)
    b = np.max(x)
    phi = np.full_like(x, 0)
    if a != b:
        phi = np.pi * (x - a) / (b - a)
    return cis(phi)

def erlangga(x):
    assert(x.ndim == 1)
    return cis(2 * np.pi * sigmoid(x))
