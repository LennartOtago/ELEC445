import numpy as np
import random as rd
def mcgaus(mu, sig, x0, N, w):
    X = [None] * N
    X[0] = x0

    for k in range(0,N-1):
    #xp = X[k] + w * (2* random() -1) # uniform window

        xp = X[k] + w * np.random.normal() #normal window variance w**2
        #print(xp)
        alpha = min(1, np.exp( (-(xp-mu)**2+ (X[k]-mu)**2)/(2 * sig**2) ))
        #print(alpha)
        if np.random.uniform(0,1,1) < alpha:
            X[k+1] = xp
        else:
            X[k+1] = X[k]

    return X


def mcexp(n,w):
    X = [None] * n  # initialize output vector
    X[0] = 1

    for k in range(0, n - 1):
        #print(X[k])
        xp = X[k] + w * (rd.random() - 1 / 2)
        if xp < 0:
            alpha = 0
        else:
            alpha = np.exp(-xp) / np.exp(-X[k])
            # alpha = np.exp(-xp + X[k])

        if rd.random() < alpha:
            X[k + 1] = xp
        else:
            X[k + 1] = X[k]


    return X