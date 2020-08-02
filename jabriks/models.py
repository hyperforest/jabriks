from .complex_activations import amin_murase2
from .real_activations import sigmoid, tanh, softmax

from .layers import RInput, RDense, CInput, CDense
from .losses import sum_squared_error
from .losses import binary_crossentropy, categorical_crossentropy

from .metrics import accuracy
from .optimizers import Optimizer, SGD
from .phase_encoding import linear, amin_murase, erlangga
from .utils import star_product, to_categorical, vectorize

import numpy as np
import pickle
from time import time

class BaseModel:
    def __init__(self, seed=None):
        self.layers = []
        self.n_layers = 0
        self.input_shape = None
        self.output_shape = None
        self.seed = seed
        self.built = False
        self.db = []
        self.dW = []
        
    def add(self, layer):
        if len(self.layers) == 0:
            assert(isinstance(layer, RInput))
            self.layers.append(layer)
            self.input_shape = layer.units
            self.db.append(None)
            self.dW.append(None)
        else:
            assert(isinstance(layer, RDense))
            shape = (self.layers[-1].units, layer.units)
            self.layers.append(layer)
            self.layers[-1].init_weights(shape, seed=self.seed)
            self.output_shape = layer.units
            self.n_layers += 1
            
            self.db.append(np.zeros((layer.units, 1)))
            self.dW.append(np.zeros(shape))

    def build(self, loss, optimizer='sgd'):
        if loss == 'sum_squared_error':
            self.loss_func = sum_squared_error
        elif loss == 'binary_crossentropy':
            self.loss_func = binary_crossentropy
        elif loss == 'categorical_crossentropy':
            self.loss_func = categorical_crossentropy
            
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        elif isinstance(optimizer, str):
            if optimizer == 'sgd':
                self.optimizer = SGD()
        
        self.built = True

    def _check_X_y(self, X, y):
        assert(X.ndim == 2)
        assert(X.shape[1] == self.input_shape)
        
        assert(y.ndim == 2)
        assert(y.shape[1] == self.output_shape)
        
        assert(X.shape[0] == y.shape[0])

    def get_weights(self):
        if self.n_layers == 0:
            return []
        
        result = []
        for layer in self.layers[1:]:
            weights, bias = layer.get_weights()
            result.append(weights.copy())
            result.append(bias.copy())
        return result

    def set_weights(self, weights):
        weights_now = self.get_weights()
        assert(len(weights_now) == len(weights))
        
        for i in range(len(weights_now)):
            assert(weights_now[i].shape == weights[i].shape)
        
        W, b = weights[::2], weights[1::2]
        for i in range(1, self.n_layers + 1):
            self.layers[i].set_weights(W[i - 1], b[i - 1])

    def _compute_gradient(self, y):
        for l in reversed(range(1, self.n_layers + 1)):
            if l == self.n_layers:
                if self.layers[l].f == softmax:
                    self.db[l] = self.layers[l].x - y
                else:
                    grad_loss = self.loss_func(self.layers[l].x, y, grad=True)
                    grad_func = self.layers[l].f(self.layers[l].s, grad=True)
                    self.db[l] = grad_loss * grad_func
            else:
                grad_next = self.layers[l + 1].W @ self.db[l + 1]
                grad_func = self.layers[l].f(self.layers[l].s, grad=True)
                self.db[l] = grad_next * grad_func
            
            self.dW[l] = self.layers[l - 1].x @ self.db[l].T

    def feed_forward(self, input_vector):
        assert(input_vector.ndim == 2)
        assert(input_vector.shape[1] == 1)
        
        x = input_vector.copy()
        for layer in self.layers:
            x = layer(x)
        return x

    def predict(self, X): 
        if X.ndim == 2:
            X = vectorize(X)
        
        result = np.array(list(map(self.feed_forward, X)))
        return result

    def evaluate(self, X, y):
        if X.ndim == 2:
            X = vectorize(X)
        if y.ndim == 2:
            y = vectorize(y)
            
        y_pred = self.predict(X)
        assert(y.shape == y_pred.shape)
        
        zipped = list(zip(y_pred, y))
        loss = list(map(lambda x : self.loss_func(x[0], x[1]), zipped))
        loss = np.mean(loss)
        
        return loss

    def fit(self, X, y, epochs, batch_size=1, val_data=None,
            verbose=2, shuffle=True, initial_epoch=1):
        pass
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f, protocol=-1)
    
    def summary(self):
        print()
        print('==============================================================')
        print('                         Model Summary                        ')
        print('==============================================================')
        
        print('- Type\t\t: Real')
        print('- #. of layers\t:', self.n_layers)      

        if self.built:
            print('- Loss function\t:', self.loss_func.__name__)
            print('- Optimizer\t: SGD, lr =', self.optimizer.__dict__['lr'])
        
        print('- Seed\t\t:', self.seed)
        
        print('- Layers detail\t:')
        print('--------------------------------------------------------------')
        print('No.\tLayer type\tUnits\tActivations\t#. of params')
        print('--------------------------------------------------------------')
        
        total_params = 0
        for i in range(len(self.layers)):
            text = '[{}]'.format(str(i))
            text += ('\tRInput' if i == 0 else '\tRDense')
            text += ('\t\t' + str(self.layers[i].units))
            
            if 'f' in self.layers[i].__dict__.keys():
                text += ('\t' + self.layers[i].f.__name__)
            
            if i > 0:
                nparams = (self.layers[i-1].units + 1) * self.layers[i].units
                total_params += nparams
                text += ('\t\t' + str(nparams))
                
            print(text)
        
        print('--------------------------------------------------------------')
        print('Total parameters:', total_params)
        print('==============================================================')
        print()

class Classifier(BaseModel):
    def __init__(self, seed=None):
        super().__init__(seed=seed)

    def inference(self, y_pred):
        pass
    
    def evaluate(self, X, y):
        if X.ndim == 2:
            X = vectorize(X)
        if y.ndim == 2:
            y = vectorize(y)
            
        y_pred = self.predict(X)
        assert(y.shape == y_pred.shape)
        
        zipped = list(zip(y_pred, y))
        loss = list(map(lambda x : self.loss_func(x[0], x[1]), zipped))
        loss = np.mean(loss)
        
        c_pred = self.inference(y_pred)
        assert(c_pred.shape == y.shape)
        acc = accuracy(y, c_pred)
        
        return (loss, acc)

class RClassifier(Classifier):
    def __init__(self, seed=None):
        super().__init__(seed=seed)
    
    def inference(self, y_pred):
        f = self.layers[-1].f
        if f == sigmoid:
            c_pred = (y_pred >= 0.5) * 1
        elif f == tanh:
            c_pred = (y_pred >= 0) * 1
        elif f == softmax:
            c_pred = y_pred.argmax(axis = 1)
            c_pred = to_categorical(c_pred, self.output_shape)
            c_pred = c_pred.reshape(c_pred.shape + (1, ))
        return c_pred

    def fit(self, X, y, epochs, batch_size=1, val_data=None,
            verbose=2, shuffle=True, initial_epoch=1, checkpointer=None):
        
        assert(self.built)
        self._check_X_y(X, y)
        
        X_train = vectorize(X)
        y_train = vectorize(y)
        hist = {'time':[], 'loss':[], 'acc':[]}
        
        if val_data is not None:
            X_val, y_val = val_data
            self._check_X_y(X_val, y_val)
            
            X_val = vectorize(X_val)
            y_val = vectorize(y_val)
            
            hist['val_loss'] = []
            hist['val_acc'] = []
        
        time_start_train = time()
        final_epoch = initial_epoch + epochs - 1
        epoch_index = list(range(initial_epoch, final_epoch + 1))
        
        n_samples = X.shape[0]
        batch_index = np.arange(n_samples)
        if shuffle:
            if self.seed is not None:
                np.random.seed(self.seed)
            np.random.shuffle(batch_index)
        
        for epoch in epoch_index:
            time_start_epoch = time()
            
            for i in range(0, n_samples, batch_size):
                index = batch_index[i : i + batch_size]
                batch_data = (X_train[index], y_train[index])
                self.optimizer(self, batch_data)
            
            loss, acc = self.evaluate(X_train, y_train)
            hist['loss'].append(loss)
            hist['acc'].append(acc)
            report = 'loss: {:.4f} - acc: {:.4f}'.format(loss, acc)
            
            if val_data is not None:
                loss, acc = self.evaluate(X_val, y_val)
                hist['val_loss'].append(loss)
                hist['val_acc'].append(acc)
                report += ' - val_loss: {:.4f} - val_acc: {:.4f}'.format(loss, acc)
            
            total_time_epoch = time() - time_start_epoch
            header = ('[{}/{}] - time: {:.2f} s. - '.format(
                    epoch, final_epoch, total_time_epoch))
            report = header + report
            hist['time'].append(total_time_epoch)
            
            if checkpointer is not None:
                logs = {k:v[-1] for k, v in zip(hist.keys(), hist.values())}
                checkpointer(epoch, logs)
                
            if verbose >= 2:
                print(report)
    
        if verbose > 0:
            elapsed = time() - time_start_train
            print('\nTotal training time: {:.2f} s.'.format(elapsed))
            print('Average: {:.2f} s./epoch.\n'.format(elapsed / epochs))
        
        return (epoch_index, hist)

class CClassifier(Classifier):
    def __init__(self, phase_encoder=None, seed=None):
        super().__init__(seed=seed)
        self.phase_encoder = None
       
        if phase_encoder is None:
            self.phase_encoder = linear
        else:
            if phase_encoder == 'amin_murase':
                self.phase_encoder = amin_murase
            elif phase_encoder == 'erlangga':
                self.phase_encoder = erlangga
   
    def add(self, layer):
        if len(self.layers) == 0:
            assert(isinstance(layer, CInput))
            self.layers.append(layer)
            self.input_shape = layer.units
            self.db.append(None)
            self.dW.append(None)
        else:
            assert(isinstance(layer, CDense))
            shape = (self.layers[-1].units, layer.units)
            self.layers.append(layer)
            self.layers[-1].init_weights(shape, seed=self.seed)
            self.output_shape = layer.units
            self.n_layers += 1
            
            self.db.append(np.zeros((layer.units, 1)))
            self.dW.append(np.zeros(shape))
   
    def _compute_gradient(self, y):
        for l in reversed(range(1, self.n_layers + 1)):
            if l == self.n_layers:
                if self.layers[l].f == amin_murase2:
                    grad_loss = self.loss_func(self.layers[l].x, y, grad=True)
                    grad_func = self.layers[l].f(self.layers[l].s, grad=True)
                    self.db[l] = grad_loss * grad_func
            else:
                if 'split' in self.layers[l].f.__name__:
                    grad_next = self.layers[l + 1].W.conj() @ self.db[l + 1]
                    grad_func = self.layers[l].f(self.layers[l].s, grad=True)
                    self.db[l] = star_product(grad_next, grad_func)
            
            self.dW[l] = self.layers[l - 1].x.conj() @ self.db[l].T
            
    def inference(self, y_pred):
        f = self.layers[-1].f
        if f == amin_murase2:
            c_pred = y_pred.argmax(axis = 1)
            c_pred = to_categorical(c_pred, self.output_shape)
            c_pred = vectorize(c_pred)
            return c_pred
    
    def phase_encode(self, data):
        data_copy = data.copy()
        data_copy = list(map(self.phase_encoder, data_copy))
        return np.array(data_copy)
    
    def fit(self, X, y, epochs, batch_size=1, val_data=None, verbose=2,
            shuffle=True, initial_epoch=1, phase_encode=True, checkpointer=None):
        
        assert(self.built)
        self._check_X_y(X, y)
        
        if X.dtype.kind != 'c':
            X = X.astype('float')
        if phase_encode:
            X = self.phase_encode(X)
        
        X_train = vectorize(X)
        y_train = vectorize(y)
        
        hist = {'time':[], 'loss':[], 'acc':[]}
        
        if val_data is not None:
            X_val, y_val = val_data
            self._check_X_y(X_val, y_val)
            
            if X_val.dtype.kind != 'c':
                X_val = X_val.astype('float')
            if phase_encode:
                X_val = self.phase_encode(X_val)
            
            X_val = vectorize(X_val)
            y_val = vectorize(y_val)
            
            hist['val_loss'] = []
            hist['val_acc'] = []
        
        time_start_train = time()
        final_epoch = initial_epoch + epochs - 1
        epoch_index = list(range(initial_epoch, final_epoch + 1))

        n_samples = X.shape[0]
        batch_index = np.arange(n_samples)
        if shuffle:
            if self.seed is not None:
                np.random.seed(self.seed)
            np.random.shuffle(batch_index)
        
        for epoch in epoch_index:
            time_start_epoch = time()
            
            for i in range(0, n_samples, batch_size):
                index = batch_index[i : i + batch_size]
                batch_data = (X_train[index], y_train[index])
                self.optimizer(self, batch_data)
            
            loss, acc = self.evaluate(X_train, y_train)
            hist['loss'].append(loss)
            hist['acc'].append(acc)
            report = 'loss: {:.4f} - acc: {:.4f}'.format(loss, acc)
            
            if val_data is not None:
                loss, acc = self.evaluate(X_val, y_val)
                hist['val_loss'].append(loss)
                hist['val_acc'].append(acc)
                report += ' - val_loss: {:.4f} - val_acc: {:.4f}'.format(loss, acc)
            
            total_time_epoch = time() - time_start_epoch
            header = ('[{}/{}] - time: {:.2f} s. - '.format(
                    epoch, final_epoch, total_time_epoch))
            report = header + report
            hist['time'].append(total_time_epoch)
            
            if checkpointer is not None:
                logs = {k:v[-1] for k, v in zip(hist.keys(), hist.values())}
                checkpointer(epoch, logs)
            
            if verbose >= 2:
                print(report)
    
        if verbose > 0:
            elapsed = time() - time_start_train
            print('\nTotal training time: {:.2f} s.'.format(elapsed))
            print('Average: {:.2f} s./epoch.\n'.format(elapsed / epochs))
        
        return (epoch_index, hist)
    
    def summary(self):
        print()
        print('==============================================================')
        print('                         Model Summary                        ')
        print('==============================================================')
        
        print('- Type\t\t: Complex')
        print('- Phase-encoding:', self.phase_encoder.__name__)
        print('- #. of layers\t:', self.n_layers)      

        if self.built:
            print('- Loss function\t:', self.loss_func.__name__)
            print('- Optimizer\t: SGD, lr =', self.optimizer.__dict__['lr'])
        
        print('- Seed\t\t:', self.seed)
        
        print('- Layers detail\t:')
        print('--------------------------------------------------------------')
        print('No.\tLayer type\tUnits\tActivations\t#. of params')
        print('--------------------------------------------------------------')
        
        total_params = 0
        for i in range(len(self.layers)):
            text = '[{}]'.format(str(i))
            text += ('\tCInput' if i == 0 else '\tCDense')
            text += ('\t\t' + str(self.layers[i].units))
            
            if 'f' in self.layers[i].__dict__.keys():
                text += ('\t' + self.layers[i].f.__name__)
            
            if i > 0:
                nparams = (self.layers[i - 1].units + 1) * self.layers[i].units
                nparams *= 2
                total_params += nparams
                text += ('\t\t' + str(nparams))
                
            print(text)
        
        print('--------------------------------------------------------------')
        print('Total parameters:', total_params)
        print('==============================================================')
        print()