from .real_activations import sigmoid, tanh, relu, softmax
from .complex_activations import amin_murase2, split_relu, split_tanh
from .initializers import RInitializer, GlorotUniform, GlorotNormal
from .initializers import CInitializer, CGlorotUniform, CGlorotNormal
from .initializers import Zeros, Ones

class Layer:
    def __init__(self, units):
        self.units = units
        self.x = None
    
    def __call__(self, x):
        pass
    
class RInput(Layer):
    def __init__(self, units):
        super().__init__(units)
        
    def __call__(self, x):
        self.x = x.copy()
        return self.x
        
class CInput(Layer):
    def __init__(self, units):
        super().__init__(units)
        
    def __call__(self, x):
        self.x = x.copy()
        self.x = self.x.astype('complex')
        return self.x
    
class RDense(Layer):
    def __init__(self, units, activations,
                 weights_initializer='glorot_uniform',
                 bias_initializer='zeros'):
        
        super().__init__(units)
        
        self.W = None
        self.b = None
        self.s = None
        self.x = None
        
        if activations == 'sigmoid':
            self.f = sigmoid
        elif activations == 'tanh':
            self.f = tanh
        elif activations == 'relu':
            self.f = relu
        elif activations == 'softmax':
            self.f = softmax
            
        if isinstance(weights_initializer, RInitializer):
            self.weights_initializer = weights_initializer
        elif isinstance(weights_initializer, str):
            if weights_initializer == 'glorot_uniform':
                self.weights_initializer = GlorotUniform()
            elif weights_initializer == 'glorot_normal':
                self.weights_initializer = GlorotNormal()
            elif weights_initializer == 'zeros':
                self.weights_initializer = Zeros()
            elif weights_initializer == 'ones':
                self.weights_initializer = Ones()
    
        if isinstance(bias_initializer, RInitializer):
            self.bias_initializer = bias_initializer
        elif isinstance(bias_initializer, str):
            if bias_initializer == 'glorot_uniform':
                self.bias_initializer = GlorotUniform()
            elif bias_initializer == 'glorot_normal':
                self.bias_initializer = GlorotNormal()
            elif bias_initializer == 'zeros':
                self.bias_initializer = Zeros()
            elif bias_initializer == 'ones':
                self.bias_initializer = Ones()
    
    def __call__(self, x):
        self.s = self.W.T @ x + self.b
        self.x = self.f(self.s)
        return self.x
    
    def init_weights(self, shape, seed=None):
        n_in, n_out = shape
        W = self.weights_initializer(n_in, n_out, seed=seed)
        b = self.bias_initializer((n_out, 1), seed=seed)
        self.W, self.b = W, b
    
    def get_weights(self):
        return (self.W.copy(), self.b.copy())
    
    def set_weights(self, weights, bias):
        self.W = weights
        self.b = bias
        
class CDense(Layer):
    def __init__(self, units, activations,
                 weights_initializer='glorot_uniform',
                 bias_initializer='zeros'):
        
        super().__init__(units)
        
        self.W = None
        self.b = None
        self.s = None
        self.x = None
        
        if activations == 'amin_murase2':
            self.f = amin_murase2
        elif activations == 'split_relu':
            self.f = split_relu
        elif activations == 'split_tanh':
            self.f = split_tanh
        
        if isinstance(weights_initializer, CInitializer):
            self.weights_initializer = weights_initializer
        elif isinstance(weights_initializer, str):
            if weights_initializer == 'glorot_uniform':
                self.weights_initializer = CGlorotUniform()
            elif weights_initializer == 'glorot_normal':
                self.weights_initializer = CGlorotNormal()
            elif weights_initializer == 'zeros':
                self.weights_initializer = Zeros(dtype='complex')
            elif weights_initializer == 'ones':
                self.weights_initializer = Ones(dtype='complex')
        
        if isinstance(bias_initializer, CInitializer):
            self.bias_initializer = bias_initializer
        elif isinstance(bias_initializer, str):
            if bias_initializer == 'glorot_uniform':
                self.bias_initializer = CGlorotUniform()
            elif bias_initializer == 'glorot_normal':
                self.bias_initializer = CGlorotNormal()
            elif bias_initializer == 'zeros':
                self.bias_initializer = Zeros(dtype='complex')
            elif bias_initializer == 'ones':
                self.bias_initializer = Ones(dtype='complex')
        
    def __call__(self, x):
        self.s = self.W.T @ x + self.b
        self.x = self.f(self.s)
        return self.x
    
    def init_weights(self, shape, seed=None):
        n_in, n_out = shape
        W = self.weights_initializer(n_in, n_out, seed=seed)
        b = self.bias_initializer((n_out, 1), seed=seed)
        self.W, self.b = W, b
    
    def get_weights(self):
        return (self.W.copy(), self.b.copy())
    
    def set_weights(self, weights, bias):
        self.W = weights
        self.b = bias
