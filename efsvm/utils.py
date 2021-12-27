import numpy as np
import pandas as pd
import cvxopt
from cvxopt import matrix as cvxopt_matrix
from sklearn.model_selection import train_test_split

def Entropy(p):
    if 0 < p < 1:
        return -p*np.log(p)-(1-p)*np.log(1-p)
    else:
        return 0.0

def KNN_(X:np.array,
         y:np.array,
         k:int):
    
    # dist[i][j]: distance between X[i] and X[j]
    dist = np.zeros((X.shape[0],X.shape[0]))
    for i in range(X.shape[0]):
        dist[i,:] = np.sqrt(np.sum((X-X[i,:])**2, axis= 1))
    neigh= [np.argsort(dist[i])[1:k+1] for i in range(X.shape[0])]
    # neigh[i]= [indice of k neighbors]
    # calculate num_+i, num_-1, p_+i, p_-i
    # numm, nump, pp, pm, H    
    entropy = []
    for i in range(X.shape[0]):
        numm= np.sum(y[neigh[i]]!=1) # The number of minus-labels
        nump= k-numm # The number of the plus labels
        pp= nump/k # proportion of the plus labels
        pm= numm/k # proportion of the minus labels
        ent= Entropy(pp)
        entropy.append(ent)        

    return neigh, entropy 

def Kernel_(x, y, params = 0, type='rbf'):
    if type=='linear':
        return x@y.T 
    elif type== 'rbf':
        return np.exp(-params*(np.sum(x**2,axis=1).reshape(-1,1) + np.sum(y**2,axis=1).reshape(1,-1) - 2*x@y.T))
