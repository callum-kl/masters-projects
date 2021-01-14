import math
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns; sns.set()
from mpl_toolkits.mplot3d.axes3d import Axes3D



d = np.zeros(shape=(506,14), dtype='float')
with open('boston.txt', 'r') as f:
    for j in range(506):       
        array = [float(x) for x in next(f).split()]
        for k in range(11):
            d[j][k] = array[k]
        
        array = [float(x) for x in next(f).split()]
        for k in range(3):
            d[j][11+k] = array[k]


# =============================================================================
# Exctracting Training and Test data in 2/3 1/3 splits as in 4
# =============================================================================

def extractTrainingSet(d):
    # Generate indices used for separating data in training and test data
    allData = np.arange(0,506)
#    trainingSample = random.sample(range(0, 506), 337)
    trainingSample = np.sort(np.random.choice(506, 337, replace=False))
    testSample = [x for x in allData if x not in trainingSample]
    
    # Initialise training and test data arrays
    trainingX = np.zeros(shape=(337, 13), dtype='float')
    trainingY = np.zeros(shape=(337, 1), dtype = 'float')
    testX = np.zeros(shape=(169, 13), dtype='float')
    testY = np.zeros(shape=(169, 1), dtype = 'float')
    
    for i in range(337):
        for j in range(13):
            trainingX[i][j] = d[trainingSample[i]][j]
        trainingY[i][0] = d[trainingSample[i]][13]
    
    for i in range(169):
        for j in range(13):
            testX[i][j] = d[testSample[i]][j]
        testY[i][0] = d[testSample[i]][13]
        
    return trainingX, trainingY, testX, testY

trainingX, trainingY, testX, testY = extractTrainingSet(d)

g = [2**-(40-x) for x in range(15)]
s = [2**(7 + x/2) for x in range(13)]

def kernelMatrix(X, sigma):
    """
    Produces the kernel matrix used in the Ridge Regression for estimating alpha
    
    Keyword Arguments:
        X-- the input data
        sigma-- the chosen sigma value
        
    Returns:
        K-- the kernel matrix as in the formula
    """
    m = np.shape(X)[0]
    K = np.zeros(shape=(m, m))
    for i in range(m):
        for j in range(m):
            # using kernel trick
            K[i][j] = np.exp(-np.dot(X[i] - X[j], X[i] - X[j])/(2*sigma**2))
    return K

def regressionCoefficients(K, Y, gamma):
    """
    Computes the coefficients alpha used for fitting the regression curve.
    
    Keywords:
        K-- the kernel matrix
        Y-- the training output
        gamma-- the chosen gamma value
    
    Returns:
        alpha-- regression coefficients
    """
    m = np.shape(K)[0]
    alpha = np.linalg.solve((K + gamma*m*np.identity(m)), Y)
    return alpha

def kFoldSplits(X, Y, k):
    """
    Function to split data into k training/test folds.
    
    Keyword Arguments:
        X-- the input data
        Y-- the output data
        k-- the requested number of folds
    
    Returns:
        xTrainTestSet-- an array containing the k permumations of the training/test 
        set X data
        yTrainTestSet-- an array containing the k permumations of the training/test 
        set Y data
    
    """
    
    indices = np.array([x for x in range(np.size(X[:,0]))])
    random.shuffle(indices)
    indices = np.array_split(indices, k)    
    xTrainTestSet = list()
    yTrainTestSet = list()
    xCopy = np.copy(X)
    yCopy = np.copy(Y)
    yCopy = np.hstack(yCopy)
    for i in range(k):
        xFolds, yFolds = trainAndTestArrays(xCopy, yCopy, indices, i)
        xTrainTestSet.append(xFolds)
        yTrainTestSet.append(yFolds)
    
    return xTrainTestSet, yTrainTestSet

def trainAndTestArrays(X, Y, l, index):
    """
    Helper function to split the data into k fold splits
    """
    x_test = [X[i] for i in l[index]]
    y_test = [Y[i] for i in l[index]]

    l_copy = np.hstack((np.delete(l, index)))

    
    x_training = [X[i] for i in l_copy]
    y_training = [Y[i] for i in l_copy]
    
    return [x_training, x_test], [y_training, y_test]
        

def regressionFunction(xTrain, xtest, alpha, s):
    """
    Function to predict y values (output) given input x.
    
    Keyword Arguments:
        alpha-- regression coefficients
        s-- sigma value
    
    Returns:
        ypredict-- the predicted output values
    """
    m = np.shape(xTrain)[0]
    ypredict = 0
    for i in range(m):
        exponent = np.linalg.norm(xTrain[i] - xtest)**2
        K = np.exp(-exponent/(2*s**2))
        ypredict += alpha[i]*K
    
    return ypredict
    

def computeTotal(xTrain, yTrain, sigma, gamma):
    """
    Main function to complete question (a) - performs k folds cross validation to
    find the best combination of gamma and sigma to minimise MSE. On each fold 
    combination, it performs kernel ridge regression and computes the MSE between
    the real output values and the predicted output values. 
    Prints the cumulative average MSE for each combination of gamma and sima
    
    Keyword Arguments:
        xTrain, yTrain-- the training x and y data
        sigma, gamma-- arrays of sigma and gamma values as defined
    
    Returns:
        errors-- the MSE for all combinations of gamma and sigma
        sigma[index]-- the optimal sigma value
        gamma[index]-- the optimal gamma value
    """
    
    xfold, yfold = kFoldSplits(xTrain, yTrain, 5)
    errors = np.zeros(shape=(len(sigma),len(gamma)))
    
    # 5-fold cross validation on the kernel ridge regression for different values of
    # gamma and sigma
    for k in range(5):
        x_train = xfold[k][0]
        x_test = xfold[k][1]
        y_train = yfold[k][0]
        y_test = yfold[k][1]
    
        for i in range(len(sigma)):
           K = kernelMatrix(x_train, sigma[i])
           for j in range(len(gamma)):
               alpha = regressionCoefficients(K, y_train, gamma[j])
               errors[i][j] += (computeMSE(x_train, x_test, y_test, alpha, sigma[i]))/5
               print(errors[i][j])
    
    # Selecting the best sigma, gamma values
    b0 = errors[0][0]
    indexSigma = 0
    indexGamma = 0
    for i in range(len(sigma)):
        for j in range(len(gamma)):
            if(errors[i][j] < b0):
                b0 = errors[i][j]
                indexSigma = i
                indexGamma = j
                
    return errors, sigma[indexSigma], gamma[indexGamma], indexSigma, indexGamma

def computeMSE(xTrain, xTest, yTest, alpha, sigma):
    """
    Function to compute MSE
    """
    error = 0
    N = np.shape(xTest)[0]
    for i in range(N):
        ypredicted = regressionFunction(xTrain, xTest[i], alpha, sigma)
        error = error + np.power((yTest[i] - ypredicted),2)
    return error/N

def retrain(d, sigma, gamma):
    """
    Function to compute the predictor, by retraining on the entire training test set
    using the optimal gamma and sigma values
    
    Returns:
        training_error-- the training MSE
        test_error-- the test MSE
    """
    training_error = np.zeros(20)
    test_error = np.zeros(20)
    stdTrainingError = 0
    stdTestError = 0
    for i in range(20):
        trainingX, trainingY, testX, testY = extractTrainingSet(d)
        K = kernelMatrix(trainingX, sigma)
        alpha = regressionCoefficients(K, trainingY, gamma)
        training_error[i] = computeMSE(trainingX, trainingX, trainingY, alpha, sigma)
        test_error[i] = computeMSE(trainingX, testX, testY, alpha, sigma)
    
    stdTrainingError = np.sqrt(np.sum((training_error - np.sum(training_error)/20)**2)/19)
    stdTestError = np.sqrt(np.sum((test_error - np.sum(test_error)/20)**2)/19)
    return np.sum(training_error)/20, np.sum(test_error)/20, stdTrainingError, stdTestError

       
# =============================================================================
# Question 5 (a)
# =============================================================================
mse, sig, gam, indSigma, indGamma = computeTotal(trainingX, trainingY, s, g)
print('best sigma: {}, index: {}'.format(sig, indSigma))
print('best gamma: {}, index: {}'.format(gam, indGamma))
print(s[indSigma], g[indGamma])

# =============================================================================
# Question 5 (b) Plotting the cross-validation error
# =============================================================================
xs, ys = np.meshgrid(s, g)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(np.log2(xs), np.log2(ys), mse)
plt.title('Cross-Validation Error vs index of Sigma, Gamma')
plt.xlabel('Sigma')
plt.ylabel('Gamma')
plt.show()

ax = sns.heatmap(np.log(mse).T, xticklabels=np.log2(s), yticklabels=np.log2(g))
ax.set_xlabel("Log Sigma")
ax.set_ylabel("Log Gamma")


# =============================================================================
# Question 5 (c)
# =============================================================================
train_err, test_err, train_std, test_std = retrain(d, 4096, g[0])
print('Training MSE: ', train_err)
print('Training MSE standard deviation: ', train_std)
print('Test MSE: ', test_err)
print('Test MSE standard deviation: ', test_std)


#     

