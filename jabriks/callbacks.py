import numpy as np

class Callback:
    def __init__(self):
        pass
    
class ModelCheckpoint(Callback):
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
            filepath = '{}/{}_{:03d}_{:.4f}'.format(
                    self.filepath, self.suffix, epoch, current)
            
            self.best = current
            
            print('\nEpoch %05d: %s improved from %0.5f to %0.5f'
                  ' saving model to %s' %
                  (epoch, self.monitor, self.best, current, filepath))
            
            self.model.save(filepath)
        else:
            print('\nEpoch %05d: %s did not improve from %0.5f' %
                  (epoch, self.monitor, self.best))