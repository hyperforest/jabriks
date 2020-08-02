import numpy as np
import pandas as pd
import pickle

def cis(x):
    assert(x.ndim == 1)
    return np.cos(x) + 1.0j * np.sin(x)

def star_product(a, b):
    assert(a.shape == b.shape)
    return a.real * b.real + 1.0j * a.imag * b.imag

def vectorize(X):
    return X.reshape(X.shape + (1, ))

def to_categorical(y, n_unique):
    assert(y.ndim == 2)
    n_samples = y.shape[0]
    res = np.zeros((n_samples, n_unique))
    for i in range(n_samples):
        res[i, y[i]] = 1.0
    return res

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def hist_to_csv(epoch_index, hist, filename):
    df = pd.DataFrame(hist)
    epoch = pd.DataFrame(data=epoch_index, columns=['epoch'])
    df = pd.concat([epoch, df], axis=1)
    df.to_csv(filename, index=False)
    
class ModelCheckpoint:
    def __init__(self, model, monitor, filepath, suffix='model'):
        self.model = model
        self.monitor = monitor
        self.filepath = filepath
        self.suffix = suffix
        
        if 'loss' in monitor:
            self.monitor_op = np.less
            self.best = np.Inf
        elif 'acc' in monitor:
            self.monitor_op = np.greater
            self.best = -np.Inf
    
    def __call__(self, epoch, logs):
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if self.monitor_op(current, self.best):
            filepath = '{}/{}_{:04d}_{:.4f}.pkl'.format(
                    self.filepath, self.suffix, epoch, current)
            
            print('\nEpoch %05d: %s improved from %0.5f to %0.5f'
                  '\nSaving model to %s' %
                  (epoch, self.monitor, self.best, current, filepath))
            
            self.best = current
            self.model.save(filepath)
        else:
            print('\nEpoch %05d: %s did not improve from %0.5f' %
                  (epoch, self.monitor, self.best))