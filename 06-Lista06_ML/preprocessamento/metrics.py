import numpy as np

def accuracy(y_test, y_pred):
    return ((y_test.reshape(y_test.shape[0],)) == y_pred).sum()/y_test.shape[0]