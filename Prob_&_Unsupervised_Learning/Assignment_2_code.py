# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:56:56 2019

@author: callum
"""

import numpy as np
from matplotlib import pyplot as plt

def initialiseP(K, D):
    """
    Initialise (KxD) probability matrix, using a uniform distribution for each pkd and enforcing
    that the sum of the probabilities over the k mixtures for the dth component is 1.
    
    Keyword arguments:
        K -- number of mixtures
        D -- dimension of dataset
    
    Returns:
        P matrix
    """
    P = np.ndarray(shape=(K, D))
    for j in range(D):
        temp = [np.random.uniform(0.25,0.75) for i in range(K)]
        for i in range(K):
            P[i][j] = temp[i]/np.sum(temp)
    return P

def initialiseWeights(K):
    """
    Initialise mixing weights as a (Kx1) vector
    
    Keyword arguments:
        K -- number of components
    
    Returns:
        R -- (Kx1) vector with initialised mixing weights, where each weight is given an equal value
    """
    R = np.ndarray(shape=(K,))
        
    for i in range(K):
        R[i] = 1/K
    return R

def getBernoulliLogLike(data, mixBernoulliEst):
    """
    Function to return the MAP data-log-like for the Bernoulli Mixture. The ML data-log-likelihood is
    computed first. The MAP correction, using a Beta Prior over the mixing and P parameters (with
    prior variables set equal to alpha and beta) is then added (the log of the prior distribution is taken
    and the constant normalising terms are ignored).
    
    Keyword arguments:
        data -- input
        mixBernoullieEst -- the current estimates for the parameters in a dictionary
        
    Return:
        loglike -- the MAP log likelihood
    """
    
    loglike = 0
    
    nData, nDims = data.shape
    
    weights = mixBernoulliEst['weight']
    PT = mixBernoulliEst['prob']
    resp = mixBernoulliEst['resp']
    
    # Prior variables for the Dirichlet distribution
    alpha = 1 + 1.0e-10
    beta = 1 + 1.0e-10
    
    # Computes ML joint data log likelihood
    for n in range(nData):
        thisX = data[n,:]
        for k in range(mixBernoulliEst['k']):
            temp_term_1 = np.log(weights[k]) 
            temp_term_2 = np.sum(np.dot(thisX, np.log(PT[:,k])))
            temp_term_3 = np.sum(np.dot((1-thisX), np.log(1-PT[:,k])))
            loglike += (resp[k][n])*(temp_term_1 + temp_term_2 + temp_term_3)
            
            
#   MAP correction to log likelihood 
    loglike += (alpha-1)*np.sum(np.log(PT)) + (beta-1)*np.sum(np.log(1-PT))
            
    return np.asscalar(loglike)

def multBernoulli(x, mixBernoulliEst, k):
    """
    Function to calculate the bernoulli probability of a sample x, under mixture k.
    
    Keyword Arguments:
        x -- a single sample from the data
        mixBernoulliEst -- dictionary containing the weights, hidden responsibilities, and
        current estimate of the probabilities for pixel d under mixture k to be equal to 1
        k -- variable indicating which mixture to use
    
    Returns:
        total --
    """
    total = 1
    prob = mixBernoulliEst['prob'][:,k]
    for d in range(np.size(x)):
        total *= np.power(prob[d], x[d])*np.power(1-prob[d], 1-x[d])

    return total

def updateResponsibilities(data, mixBernoulliEst):
    """
    Function to perform the estimation step of the EM algorithm.
    
    Keyword Arguments:
        data -- all of the n sample data points
        mixBernoulliEst -- dictionary containing the weights, hidden responsibilities, and
        current estimate of the probabilities for pixel d under mixture k to be equal to 1
    
    Returns:
        nothing -- updates the responsibilities of all the k mixtures for the dth picxel
    """
    
    weights = mixBernoulliEst['weight']
    resp = mixBernoulliEst['resp']
    nData, nDims = data.shape  

    for n in range(nData):
        thisX = data[n,:]

        for k in range(mixBernoulliEst['k']):
            resp[k][n] = weights[k]*multBernoulli(thisX, mixBernoulliEst, k)
    
    for n in range(nData):
        denominator = 0
        for k in range(mixBernoulliEst['k']):
            denominator += resp[k][n]
        for k in range(mixBernoulliEst['k']):
            resp[k][n] = resp[k][n]/denominator
    return 0
         
        
def updateParams(data, mixBernoulliEst):
    """
    Function to perform the maximisation step of the EM algorithm, using the MAP log likelihood.
    
    Keyword Arguments:
        data -- all of the n sample data points
        mixBernoulliEst -- dictionary containing the weights, hidden responsibilities, and
        current estimate of the probabilities for pixel d under mixture k to be equal to 1
    
    Returns:
        nothing -- updates the values of the mixture probabilities, and probabilities of pixel
        d to take value 1 under the kth mixture.
    """
    
    weights = mixBernoulliEst['weight']
    resp = mixBernoulliEst['resp']
    prob = mixBernoulliEst['prob'].T
    nData, nDims = data.shape
    
    
    alpha = 1 + 1.0e-10
    beta = 1 + 1.0e-10
    eps = 1.0e-30
     
    for k in range(mixBernoulliEst['k']):
        weights[k] = (np.sum(resp[k,:]))/(nData)
    
        
    for k in range(mixBernoulliEst['k']):
        sum_resp = np.sum(resp[k,:])
        for d in range(nDims):
            prob[k][d] = (np.sum(resp[k,:]*data[:,d])+alpha-1)/(sum_resp + alpha + beta - 2)
#            prob[k][d] = (np.sum(resp[k,:]*data[:,d])+eps)/(sum_resp)

def drawCluster(cluster):
    """
    Function to draw the clusters, using a threshold of 0.35 to produce the 
    binary values.
    
    Keyword Arguments:
        cluster -- (Dx1) vector containing the probability estimates that pixel d takes value 1. 
    
    Returns:
        nothing -- reshapes the vector back into an (8x8) image, and plots.
    """
    temp = np.zeros(np.size(cluster))
    for i in range(np.size(cluster)):
        if(cluster[i] <= 0.35):
            temp[i] = 1
        else:
            temp[i] = 0
            
    cluster = np.reshape(cluster, (8,8))
    temp = np.reshape(temp, (8,8))

    
    plt.figure()
    plt.imshow(temp,
               interpolation="None",
               cmap='gray',
               vmin=0., vmax=1.)
    plt.axis('off')
    plt.show()
    
def EMAlgorithm(X, K, nIter):
        # load the data set
    N, D = X.shape 
    mixBernoulliEst = dict()
    mixBernoulliEst['d'] = D
    mixBernoulliEst['k'] = K
    mixBernoulliEst['weight'] = initialiseWeights(K)
    mixBernoulliEst['prob'] = initialiseP(K, D).T
    mixBernoulliEst['resp'] = np.zeros(shape=(K, N))
    print(mixBernoulliEst['resp'])
    for i in range(K):
        for j in range(N):
            mixBernoulliEst['resp'][i][j] = (1/K)
    
    logLike = getBernoulliLogLike(X, mixBernoulliEst)
    prev = abs(logLike)
    print('Performing EM for {}-mixture bernoulli...'.format(K))
#    print(logLike)
    for cIter in range(nIter):
        # =============================================================================
        #   Expectation Step: update responsibilities
        # =============================================================================
        updateResponsibilities(X, mixBernoulliEst)
            
        # =============================================================================
        #   Maximisation Step: update parameter values
        # =============================================================================
        updateParams(X, mixBernoulliEst)
            
        # =============================================================================
        #   Calculate current loglikehood for the model and print
        # =============================================================================
        logLike = getBernoulliLogLike(X, mixBernoulliEst)
#        print(logLike)
            
        if(abs(prev - logLike) < 1e-10):
            break
        else:
            prev = logLike
    
    # display the K clusters
    for cluster in range(K):
        drawCluster(mixBernoulliEst['prob'][:,cluster])
    
    # return the final log likelihood
    return [logLike, mixBernoulliEst]    

def main():
    
    loglikearray = list()
    nIter = 50
    X = np.loadtxt('binarydigits.txt')
    for i in [2,3,4,7,10]:
        arr = EMAlgorithm(X, i, nIter)
        loglikearray.append(arr[0])
        print('Log Likelihood is {} \n'.format(arr[0]))
#    print(loglikearray)
    
    return 0    

if __name__ == "__main__":
    main()
    
