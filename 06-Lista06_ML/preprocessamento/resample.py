import numpy as np
import pandas as pd

def split_stratified_train_test(X,y, perc_train, seed):
    #Shuffle dos dados
    tam = len(y)
    rs = np.random.RandomState(seed)
    shuffled_indices = rs.permutation(tam)
    
    X = X[shuffled_indices]
    y = y[shuffled_indices]

    #Discover the labels that exists
    labels, counts = np.unique(y, return_counts=True)

    #Calculate the probabilities of each label in the original dataset
    probabilitites = counts/tam

    #Generate the amount of each label in the training dataset
    sequence = np.random.choice(labels, round(tam*perc_train), p = probabilitites)

    #Get the amount of each label in the training dataset
    unique, labels_train_qtd = np.unique(sequence, return_counts=True)

    idx_train = []
    idx_test = []

    for i in range(tam):
        for j in range(len(labels)):
            if(y[i] == labels[j]):
                if(labels_train_qtd[j] > 0):
                    labels_train_qtd[j] -= 1
                    idx_train.append(i)
                else:
                    idx_test.append(i)
                    
    return X,y,idx_train, idx_test