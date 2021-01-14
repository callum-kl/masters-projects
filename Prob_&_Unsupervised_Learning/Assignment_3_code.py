# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 12:26:46 2019

@author: callu
"""

import numpy as np
from matplotlib import pyplot as plt

d_temp = np.loadtxt("co2.txt")
nData, nDim = d_temp.shape
d = np.zeros(shape=(nData, nDim-1))

for cIter in range(nData):
    d[cIter,:] = [d_temp[cIter,0] + (d_temp[cIter,1] - 1)/12, d_temp[cIter,2]]

Y = d[:,1].reshape((1,nData))
X = np.array([np.ones(nData),d[:,0]])
C = np.array([[100**2, 0],[0, 10**2]])
C_inv = np.linalg.inv(C)

sigma_w = np.linalg.inv(X @ X.T + C_inv)
mean_w = sigma_w @ (C_inv @ np.array([[360], [0]]) + X @ Y.T)

a_map = mean_w[1]
b_map = mean_w[0]
g_obs = Y - (a_map*X[1,:] + b_map) 
plt.scatter(X[1,:],g_obs[0], marker='x', color='r')
plt.plot(X[1,:], g_obs[0])
plt.xlabel('time')
plt.ylabel('residuals')
plt.show()
print('mean: {}, var = {}'.format(np.mean(g_obs), np.var(g_obs)))

# =============================================================================
# Justification
# - mean of 0 justification correct
# - normality looks correct
# - variance appears to be incorrect, much higher at 3.3
# - i.i.d seems to be incorrect, would see no correlation between mean and year
# but clearly see mean alternates between high and low for different periods of years.
# =============================================================================


def kernel(k, h, x_1, x_2):
    """
    Returns the kernel covariance given by input points x_1, x_2.
    -- k is the kernel function
    -- h is a list of hyperparameters
    """
    n_1 = x_1.shape[0]
    n_2 = x_2.shape[0]
    K = np.zeros(shape=(n_1, n_2))
    for i in range(n_1):
        for j in range(n_2):
            K[i][j] = k(x_1[i,:], x_2[j,:], h)
    return K
    
# theta=2, tau=1, sigma=2, phi=22, eta=28, ups=0.15
def k_1(s, t, h):
    """
    Definition of the kernel given in the question
    -- h is a list containing the hyperparameters
    """
    theta, tau, sigma, phi, eta, ups = h[0], h[1], h[2], h[3], h[4], h[5] 
    term_1 = np.exp(-2*np.sin(np.pi*(s-t)/tau)**2/sigma**2)
    term_2 = phi**2*np.exp(-(s-t)**2/(2*eta**2))
    term_3 = 0
    if(s == t):
        term_3 = ups**2
    return theta**2*(term_1 + term_2) + term_3

def GP_sample(k, h, x, s):
    """
    Generates s samples from drawn from a GP with zero mean.
    -- K is the kernel matrix evaluated at the points x, given the kernel k
    -- L is the cholesky decomposition
    -- u is a set of s random normal vectors
    -- f is the function evaluated at input points x
    """
    n, d = x.shape
    K = kernel(k, h, x, x)
    L = np.linalg.cholesky(K)
    u = np.random.normal(size=(n,s))
    f = L @ u
    
    return f

# =============================================================================
# Term_1:
# Theta = overall amplitude
# Tau = periodicity -> large tau = less periodic: want tau = 1 to correspond to one year
# Sigma = Gives the smoothness of the periodic component
    
# Term_2: = long term trend smooth using squared exponential term
# Phi -> is the amplititude of this term
# Eta is the scale 
# 
# Term_3:
# ups -> independent noise component
    
# =============================================================================

# =============================================================================
# Testing the function for samples
# =============================================================================
n = 100
x_temp = np.linspace(-25, 25, n).reshape(-1,1)

def sample_GP(k, x, s):
    
    #ETA
    xlim = (-25, 25)
    ylim = (-5, 5)
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(x, GP_sample(k, [1,1,100000,1,0.1,0.01], x, s))
    ax[1].plot(x, GP_sample(k, [1,1,100000,1,1,0.01], x, s))
    ax[2].plot(x, GP_sample(k, [1,1,100000,1,10,0.01], x, s))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle('Samples from GP prior, \u03C3 = 10000 \u03B7  = 0.1, 1, 10,')
    plt.setp(ax, xlim=xlim, ylim=ylim)
    plt.show()
    
    return 0 

#Relabel for ease of understnading:
x_train = X[1,:][np.newaxis].T
x_test = np.array([2007 + x/12 for x in range(8, 14*12)])[np.newaxis].T
y_train = g_obs.T

def GP_posterior(x_train, x_test, y_train, k, h):
    """
    Function to compute the posterior mean and variance of the GP
    -- x_train, y_train is the training data
    -- x_test is the set of x points we wish to evaluate the GP on.
    -- k is the given kernel function
    -- h is a list of the chosen hyperparameter values
    """
    
    kernel_11 = kernel(k, h, x_train, x_train)
    
    kernel_12 = kernel(k, h, x_train, x_test)
    
    kernel_22 = kernel(k, h, x_test, x_test)
    
    n = x_train.shape[0]
    
    kernel_solve = kernel_12.T @ np.linalg.pinv(kernel_11 + np.eye(n))
    
    post_mean = kernel_solve @ y_train
    
    post_var = kernel_22 - kernel_solve @ kernel_12
    
    return post_mean, post_var

h_chosen = [2,1,1,1,3.75,0.01]
p_m, p_v = GP_posterior(x_train, x_test, y_train, k_1, h_chosen)
plt.plot(x_train, GP_sample(k_1, h_chosen, x_train, 3))
plt.title("3 GP samples using chosen parameters")
plt.show()

# One standard deviation
stdv = np.sqrt(np.diag(p_v))

# Computing the function f(t)
f_t = a_map*x_test + b_map + p_m

# Plotting g(t)
plt.plot(x_train, y_train, label='g_obs(t)')
plt.plot(x_test, p_m, label = 'g(t)')
plt.gca().fill_between(x_test.flat, p_m.flat-stdv, p_m.flat+stdv, color="#dddddd")
plt.title('GP posterior')
plt.legend(loc='upper left')
plt.show()
#
plt.plot(x_train, Y.T, label='f_obs(t)')
plt.plot(x_test, f_t, label = 'f(t)')
plt.gca().fill_between(x_test.flat, f_t.flat -stdv, f_t.flat +stdv, color="#dddddd")
plt.title('Plot of extrapolated CO_2 values')
plt.legend(loc='upper left')
plt.show()



    
    