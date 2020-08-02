import numpy as np

class Optimizer:
    def __init__(self):
        pass

class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr
        
    def __call__(self, model, batch_data):
        db = [np.full_like(db, 0) for db in model.db[1:]]
        dW = [np.full_like(dW, 0) for dW in model.dW[1:]]
        X, y = batch_data
        n_samples = X.shape[0]
        
        for i in range(n_samples):
            model.feed_forward(X[i])
            model._compute_gradient(y[i])
            
            db = list(map(lambda x : x[0] + x[1], list(zip(db, model.db[1:]))))
            dW = list(map(lambda x : x[0] + x[1], list(zip(dW, model.dW[1:]))))
            
        db = [None] + list(map(lambda x : x / n_samples, db))
        dW = [None] + list(map(lambda x : x / n_samples, dW))
        
        for l in range(1, len(model.layers)):
            model.layers[l].b = model.layers[l].b - self.lr * db[l]
            model.layers[l].W = model.layers[l].W - self.lr * dW[l]