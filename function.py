import numpy as np
import random as rd

# Return N sa
def mcgaus(mu, sig, x0, N, w):
    """
    Return N samples of N(mu,sig^2) using a
    RWM with window w, starting at x0
    :param mu:
    :param sig:
    :param x0:
    :param N:
    :param w:
    :return:
    """
    X = [None] * N
    X[0] = x0

    for k in range(0,N-1):
    #xp = X[k] + w * (2* random() -1) # uniform window
        #generate new number with X[k] mean and w as width
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
    """
    implement MCMC for exponential distribution, unit mean
    :param n:
    :param w:
    :return:
    """
    X = [None] * n  # initialize output vector
    X[0] = 1

    for k in range(0, n - 1):
        #print(X[k])
        #generate new number
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

#def dmcmc

def putgood(a,x,y,r):
    """
    put good cell in image a

    """

    m,n = a.shape
    #print(n)
    mm, nn = np.meshgrid(np.linspace(1,m,m), np.linspace(1,n,n))
    d2 = (mm -x)**2 + (nn -y)**2
    a[(mm -y)**2 + (nn -x)**2 <= r**2] = 0.5
    a[(mm - y) ** 2 + (nn - x) ** 2 <= (r/2)**2] = 0
    return a

def putbad(a,x,y,r):
    """
    put bad cell in image a

    """

    m,n = a.shape
    #print(n)
    mm, nn = np.meshgrid(np.linspace(1,m,m), np.linspace(1,n,n))
    d2 = (mm -x)**2 + (nn -y)**2
    a[(mm -y)**2 + (nn -x)**2 <= r**2] = 0.5
    a[(mm - y) ** 2 + (nn - x) ** 2 <= (r/2)**2] = 1
    return a