import math
import numpy as np
import matplotlib.pyplot as plt
import random

# =============================================================================
# Extracting boston data
# =============================================================================
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
# Separating boston data into random 2/3rd 1/3rd training and test splits.
# =============================================================================
def extractTrainingSet(d):
    """ Function to Separating boston data into random 2/3rd 1/3rd training 
    and test splits.
    
    Keyword Arguments:
        d-- the boston data in raw form
    
    Returns:
        trainingX, testX-- the training/test input data (with bias term added)
        trainingY, testY-- the training/test output data    
    """
    # Generate indices used for separating data in training and test data
    allData = np.arange(0,506)
    trainingSample = random.sample(range(0, 506), 337)
    testSample = [x for x in allData if x not in trainingSample]
    
    # Initialise training and test data arrays
    trainingX = np.zeros(shape=(337, 14), dtype='float')
    trainingY = np.zeros(shape=(337, 1), dtype = 'float')
    testX = np.zeros(shape=(169,14), dtype='float')
    testY = np.zeros(shape=(169, 1), dtype = 'float')
    
    for i in range(337):
        for j in range(14):
            if(j == 0):
                # adding bias term
                trainingX[i][j] = 1
            else:
                trainingX[i][j] = d[trainingSample[i]][j-1]
        trainingY[i][0] = d[trainingSample[i]][13]
    
    for i in range(169):
        for j in range(14):
            if(j == 0):
                # adding bias term
                testX[i][j] = 1
            else:
                testX[i][j] = d[testSample[i]][j-1]
        testY[i][0] = d[testSample[i]][13]
        
    return trainingX, trainingY, testX, testY


# =============================================================================
# Question 4 (a) and (b)
# =============================================================================
def naiveRegression(trainingX, trainingY, testX, testY):
    """
    Function to perform naive regression, i.e just fitting a constant line
    
    Returns:
        trainingMSE-- the MSE computed from the regression fit from the training data
        testMSE-- the MSE computed between the regression fit and the test data
    """
    # The ones correspond to the design matrix X with just an intercept.
    onesTraining = np.ones(np.size(trainingX[:,0]))
    onesTest = np.ones(np.size(testX[:,0]))
    
    # Solving the normal equation to find the constant function w.
    XTX = np.dot(np.transpose(onesTraining), onesTraining)
    XTY = np.dot(np.transpose(onesTraining), trainingY)
    w = (1/XTX)*XTY
    
    # Question 4(b)
    # This constant term w is simply the estimate (sampled from training data) 
    # of the mean of the median house prices.

    # Calculating both the training and test mean squared error.
    trainingMSE = np.sum((trainingY - w)**2)/(np.size(trainingY))
    testMSE = np.sum((testY - w)**2)/(np.size(testY))   
    return trainingMSE, testMSE

def averageNaiveRegression(d):
    """
    Simply performs naive regression over 20 runs with different training/test samples
    over each run
    
    Returns:
        Average training MSE--
        Average test MSE--
        stdTrainingMSE-- the standard deviation of the training MSE
        stdTestMSE-- the standard deviation of the test MSE
    """
        
    trainingMSE = np.zeros(20)
    testMSE = np.zeros(20)
    stdTrainingMSE = 0
    stdTestMSE = 0
    
    for i in range(20):
        trainingX, trainingY, testX, testY = extractTrainingSet(d)
        t1, t2 = naiveRegression(trainingX, trainingY, testX, testY)
        trainingMSE[i] = t1
        testMSE[i] = t2
    
    stdTrainingMSE = np.sqrt(np.sum((trainingMSE - np.sum(trainingMSE)/20)**2)/19)
    stdTestMSE = np.sqrt(np.sum((testMSE - np.sum(testMSE)/20)**2)/19)
    
    return np.sum(trainingMSE)/20, np.sum(testMSE)/20, stdTrainingMSE, stdTestMSE
        
             
averageNaiveTrainingMSE, averageNaiveTestMSE, stdTrainingMSE, stdTestMSE = averageNaiveRegression(d)
print('\nAverage Naive training MSE: ', averageNaiveTrainingMSE)
print('Standard deviation of Naive training MSE: ', stdTrainingMSE)
print('Average Naive testing MSE: ', averageNaiveTestMSE)
print('Standard deviation of Naive test MSE: ', stdTestMSE)
      

# =============================================================================
# Question 4(c)
# =============================================================================
def performRegression(attribute, trainingY):
    
    # Use least squares to perform regression for an attribute with bias term.
    XTX = attribute.T @ attribute
    XTY = attribute.T @ trainingY
    w = np.linalg.solve(XTX, XTY)
    return w

def singleRegression(trainingX, trainingY, testX, testY):
    """
    Function to perform regression on all the single attributes.
    """
    wCoef = np.zeros(shape=(13, 2))
    N1 = np.size(trainingY)  
    N2 = np.size(testY)

    trainingMSE = np.zeros(13)
    testMSE = np.zeros(13)
    
    
    for i in range(1, np.size(trainingX[0])):
        
        # Extract colums 0 and i to form data matrix X for this regression.
        x_training = trainingX[:,[0,i]]
        x_test = testX[:,[0,i]]
        
        # Use least squares method to perform regression on X, Y to find 
        # coefficients w.
        wTemp = performRegression(x_training, trainingY)
        wCoef[i-1][0], wCoef[i-1][1] = wTemp[0], wTemp[1] 
        
        # Calculate XB for the regression equation and find the MSE for the
        # training set and test set
        y_training_predict = np.dot(x_training, wTemp)
        y_test_predict = np.dot(x_test, wTemp)
        trainingMSE[i-1] = np.sum((trainingY - y_training_predict)**2)/N1
        testMSE[i-1] = np.sum((testY - y_test_predict)**2)/N2
        
    return wCoef, trainingMSE, testMSE

def averageSingleRegression(d):
    """
    Function to run single Regression 20 times and take the average MSEs and standard deviations.
    """
    weights = np.ndarray(shape=(13,2))
    trainingMSE = np.zeros(shape=(13,20))
    testMSE = np.zeros(shape=(13,20))
    stdTrainingMSE = np.zeros(shape=(13,))
    stdTestMSE = np.zeros(shape=(13,))
    for i in range(20):
        trainingX, trainingY, testX, testY = extractTrainingSet(d)
        w, trainingMSEsample, testMSEsample = singleRegression(trainingX, trainingY, testX, testY)
        weights += w
        trainingMSE[:,i] = trainingMSEsample
        testMSE[:,i] = testMSEsample
    
    for j in range(13):
        stdTrainingMSE[j] = np.sqrt(np.sum((trainingMSE[j,:]-np.sum(trainingMSE[j,:])/20)**2)/(19))
        stdTestMSE[j] = np.sqrt(np.sum((testMSE[j,:]-np.sum(testMSE[j,:])/20)**2)/(19))


        
        
    return weights/20, np.sum(trainingMSE,axis=1)/20, np.sum(testMSE,axis=1)/20, stdTrainingMSE, stdTestMSE
    
averageWeights, averageTrainingMSE, averageTestMSE, stdTrainingMSE, stdTestMSE = averageSingleRegression(d)

print('\nArray of average coefficients of w for each attribute with bias term:')
print(averageWeights)
print('\nArray of average training MSE for each attribute used in regression: ')
print(averageTrainingMSE)
print('\nArray of training standard deviation MSE for each attribute used in regression: ')
print(stdTrainingMSE)
print('\nArray of average test MSE for each attribute used in regression: ')
print(averageTestMSE)
print('\nArray of test standard deviation MSE for each attribute used in regression: ')
print(stdTestMSE)


# =============================================================================
# Question 4 (d)
# =============================================================================
def fullRegression(trainingX, trainingY, testX, testY):
    """
    Performs Regression using all the attributes.
    """
    w = performRegression(trainingX, trainingY)
    trainingMSE = np.sum((trainingY - np.dot(trainingX, w))**2)/(np.size(trainingY))
    testMSE = np.sum((testY - np.dot(testX, w))**2)/(np.size(testY))
    
    return trainingMSE, testMSE

def averageFullRegression(d):
    trainingMSE = np.ndarray(shape=(20,))
    testMSE = np.ndarray(shape=(20,))
    stdTrainingMSE = 0
    stdTestMSE = 0
    
    for i in range(20):
        trainingX, trainingY, testX, testY = extractTrainingSet(d)
        t1, t2 = fullRegression(trainingX, trainingY, testX, testY)
        trainingMSE[i] = t1
        testMSE[i] = t2
    stdTrainingMSE = np.sqrt(np.sum((trainingMSE - np.sum(trainingMSE)/20)**2)/19)
    stdTestMSE = np.sqrt(np.sum((testMSE - np.sum(testMSE)/20)**2)/19)
    
    return (np.sum(trainingMSE)/20), np.sum(testMSE)/20, stdTrainingMSE, stdTestMSE
        

train_mse_full, test_mse_full, stdTrainingMSE, stdTestMSE = averageFullRegression(d)
print('\nAverage Training MSE using all attributes: ', train_mse_full)
print('Standard Deviation of Training MSE using all attributes: ', stdTrainingMSE)
print('Average Test MSE using all attributes: ', test_mse_full)
print('Standard Deviation of Test MSE using all attributes: ', stdTestMSE)

