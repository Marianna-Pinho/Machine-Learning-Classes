import numpy as np

class Normalize(object):
    
    def __init__(self):
        self.x_min = []
        self.x_max = []
        
    def fit(self, X):
        
        n_cols = X.shape[1]
        
        for i in range(n_cols):
            self.x_min.append(np.min(X[:, i]))
            self.x_max.append(np.max(X[:, i]))
    
    def transform(self, X):
        X_norm = np.copy(X)
        n_cols = X.shape[1]
        
        for i in range(n_cols):
            X_norm[:, i] = (X[:, i] - self.x_min[i])/(self.x_max[i] - self.x_min[i])
            
        return X_norm
    
    
class Standardize(object):
    
    def __init__(self):
        self.x_mean = []
        self.x_std = []
        
    def fit(self, X):
        
        n_cols = X.shape[1]
        
        for i in range(n_cols):
            self.x_mean.append(np.mean(X[:, i]))
            self.x_std.append(np.std(X[:, i]))
    
    def transform(self, X):
        X_std = np.copy(X)
        n_cols = X.shape[1]
        
        for i in range(n_cols):
            X_std[:, i] = (X[:, i] - self.x_mean[i])/ self.x_std[i]
            
        return X_std