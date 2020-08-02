import numpy as np

class Initializer:
    def __init__(self):
        pass

class RInitializer(Initializer):
    def __init__(self):
        pass

class CInitializer(Initializer):
    def __init__(self):
        pass
        
class GlorotUniform(RInitializer):
    def __init__(self):
        super().__init__()
    
    def __call__(self, n_in, n_out, seed=None):
        np.random.seed(seed)
        result = np.random.random((n_in, n_out))
        a = -np.sqrt(6 / (n_in + n_out))
        b = -a
        result = (b - a) * result + a
        return result

class CGlorotUniform(CInitializer):
    def __init__(self):
        super().__init__()
        
    def __call__(self, n_in, n_out, seed=None):
        g = GlorotUniform()
        
        result_real = g(n_in, n_out, seed=seed)
        if seed is not None:
            seed += 1
        result_imag = g(n_in, n_out, seed=seed)
        
        result = result_real + result_imag * 1.0j
        return result
    
class GlorotNormal(RInitializer):
    def __init__(self):
        super().__init__()
    
    def __call__(self, n_in, n_out, seed=None):
        np.random.seed(seed)
        mean = 0
        std = np.sqrt(2 / (n_in + n_out))
        result = np.random.normal(loc=mean,
                                  scale=std,
                                  size=(n_in, n_out))
        return result

class CGlorotNormal(CInitializer):
    def __init__(self):
        super().__init__()
        
    def __call__(self, n_in, n_out, seed=None):
        g = GlorotNormal()
        
        result_real = g(n_in, n_out, seed=seed)
        if seed is not None:
            seed += 1
        result_imag = g(n_in, n_out, seed=seed)
        
        result = result_real + result_imag * 1.0j
        return result
    
class Zeros(Initializer):
    def __init__(self, dtype='float'):
        super().__init__()
        self.dtype = dtype
        
    def __call__(self, shape, seed=None):
        result = np.zeros(shape, dtype=self.dtype)
        return result

class Ones(Initializer):
    def __init__(self, dtype='float'):
        super().__init__()
        self.dtype = dtype
        
    def __call__(self, shape, seed=None):
        result = np.ones(shape, dtype=self.dtype)
        return result
