import numpy as np

def standardization(x_np):
    return (x_np - x_np.mean()) / x_np.std()

def minMaxScaling(x_np):
    return (x_np - x_np.min()) / (x_np.max() - x_np.min())
