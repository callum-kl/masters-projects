import math
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Question 2
# =============================================================================

# =============================================================================
# Generating training data
# =============================================================================
sigma = 0.07
eps = np.random.normal(0, sigma, 30)
xpoints = np.array(sorted(np.random.uniform(0,1,30)))
ypoints = np.sin(2*np.pi*xpoints)**2+eps

# =============================================================================
# Helper Functions
# =============================================================================

def phiMatrix(points, degree):
    """ 
    Function to produce a k dimensional polynomial basis for the data points
    
    Keyword Arguments:
        points -- the x coordinates of the training data
        degree -- the chosen dimension of the basis
    
    Returns:
        phi -- the phi matrix in the polynomial basis
    """
    
    phi = np.zeros((np.size(points), degree))
    for i in range(np.size(points)):
        for j in range(degree):
            phi[i][j] = np.power((points[i]),j)

    return phi

def phiMatrixSin(points, degree):
    """ 
    Function to produce a k dimensional sinusoidal basis for the data points
    
    Keyword Arguments:
        points -- the x coordinates of the training data
        degree -- the chosen dimension of the basis
    
    Returns:
        phi -- the phi matrix in the sin basis
    """
    phi = np.zeros((np.size(points), degree))
    for i in range(np.size(points)):
        for j in range(degree):
            phi[i][j] = np.sin((j+1)*np.pi*points[i])
    return phi

def ls_estimate(phi, ypoints):
    """
    Function to provide the least-squares estimate of the coefficients
    
    Keyword Arguments:
        phi -- the matrix of basis functions
        ypoints -- the y coordinates of the training data
    
    Returns:
        w -- the least squares estimate of the regression coefficients
    """
    
    XTX = np.dot(phi.T, phi)
    XTY = np.dot(phi.T, ypoints.reshape((np.size(ypoints),1)))
    w = np.flip(np.linalg.solve(XTX, XTY))
#   w1 = np.flip((np.linalg.lstsq(phi, ypoints))[0])
    return w

def fitPolynomials(k, xpoints, ypoints):
    """
    Function to fit the polynomial curves using a polynomial basis (phiMatrix)
    and least squares (ls_estimate).
    
    Keyword Arguments:
        k -- degree of polynomial
        xpoints -- training x data
        ypoints -- training y data
        
    Returns:
        wCoef -- a list of the the coefficients of the fitted polynomial curves 
        up to degree k
    """
    
    wCoef = []
    phi = phiMatrix(xpoints, k)
    for i in range(1,k+1):
        phiK = phi[:,:i]
        wK = ls_estimate(phiK, ypoints)
        wCoef.append(wK)
    return wCoef

def fitSinBasis(k, xpoints, ypoints):
    """
    Same as fitPolynomials, except for the sin basis
    """
    
    wCoef = []
    phi = phiMatrixSin(xpoints, k)
    for i in range(1,k+1):
        phiK = phi[:,:i]
        wK = ls_estimate(phiK, ypoints)
        wCoef.append(wK)
    return wCoef
    


def plotPolynomialFits(xpoints, ypoints):
    """
    Function to plot the fitted polynomials for k= 2,5,10,14,18
    """
    x_range = np.linspace(0,1,100)
    plt.plot(xpoints, ypoints, 'o')
    plt.xlim(0,1)
    plt.ylim(-0.2,1.8)
    for k in [2,5,10,14,18]:
        phi = phiMatrix(xpoints, k)
        w = ls_estimate(phi, ypoints)
        y_predicted = np.polyval(w, x_range)
        plt.plot(x_range, y_predicted, label='k={}'.format(k))
    
    plt.legend(loc='upper right')
    plt.show()
    
def plotSinFits(xpoints, ypoints):
    """
    Function to plot the fitted polynomials for k= 2,5,10,14,18
    """
    x_range = np.linspace(0,1,100)
    plt.plot(xpoints, ypoints, 'o')
    plt.xlim(0,1)
    plt.ylim(-0.2,1.8)
    for k in [2,5,10,14,18]:
        phi = phiMatrixSin(xpoints, k)
        w = ls_estimate(phi, ypoints)
        y_predicted = sinEval(w, x_range)
        plt.plot(x_range, y_predicted, label='k={}'.format(k))
    
    plt.legend(loc='upper right')
    plt.show()
    
    
def calculateMeanSquareError(wCoef, xpoints, ypoints , indicator=1):
    """
    Function to calculate the MSE of all the fitted polynomials.
    
    Keyword Arguments:
        wCoef -- a list of the coefficients for the fitted polynomial curves
        xpoints, ypoints -- training data
    
    Returns:
        meanSquareError -- an array of the MSE for all the polynomial fits
        """
    k = np.size(wCoef)
    meanSquareError = np.zeros(k)
    N = np.size(xpoints)
    if(indicator):
        for i in range(k):
            meanSquareError[i] = (np.sum((np.polyval(wCoef[i],xpoints) - ypoints)**2)/(N))
    else:
        for i in range(k):
            meanSquareError[i] = (np.sum((sinEval(wCoef[i],xpoints) - ypoints)**2)/(N))
        
    return meanSquareError

def generateDataSet(size):
    """
    Function to generate the required data set 
    -- gσ(x) := sin^2(2πx) + eps
    
    Keyword Arguments:
        size -- number of points in data set requested
    """
    sigma = 0.07
    eps = np.random.normal(0, sigma, size)
    xpoints = np.array(np.random.uniform(0,1,size))
    ypoints = (np.sin(2*np.pi*xpoints))**2+eps
    return xpoints, ypoints


def calculateTestError(wMatrix, numberOfTests, indicator=1):
    """
    Function to calculate the test error for Question 2 (c)
    
    Keyword Arguments:
        wMatrix -- the matrix of least squares estimates of the coefficients
        numberOfTests -- number of test points requested
        
    Returns
        nothing -- prints the Log of the Test Error against the Dimension of the Basis used.
    """
    
    testX, testY = generateDataSet(numberOfTests)
    testError = np.log(calculateMeanSquareError(wMatrix, testX, testY, indicator))
    kInts = [i+1 for i in range(18)]
    plt.xlabel('Basis Dimension')
    plt.ylabel('Log of Test Error')
#    plt.title('(c)')
    plt.plot(kInts, testError)
    plt.show()

# =============================================================================
# Specific function to calculate the average training and test errors for 2d.)
# =============================================================================
def averageTrainingAndTestError(numberOfRuns, indicator=1):
    """
    Function to calculate the average training and test error over a certain numberOfRuns.
    Generates 30 training points and a 1000 test points, and for each run, estimates
    the polynomial coefficients and computes the training and test MSE
    
    Keyword Arguments:
        numberOfRuns --
        
    Returns:
        Average training and test error over numberOfRuns.
    """
    trainingError = np.zeros(18)
    testError = np.zeros(18)
    xTraining, yTraining = generateDataSet(30)
    xTest, yTest = generateDataSet(1000)
    
    if(indicator):
        for i in range(numberOfRuns):
            w = fitPolynomials(18, xTraining, yTraining)
            trainingError = trainingError + calculateMeanSquareError(w, xTraining, yTraining)
            testError = testError + calculateMeanSquareError(w, xTest, yTest)
            
    else:
        for i in range(numberOfRuns):
            w = fitSinBasis(18, xTraining, yTraining)
            trainingError = trainingError + calculateMeanSquareError(w, xTraining, yTraining, 0)
            testError = testError + calculateMeanSquareError(w, xTest, yTest, 0)
    
        
        
    trainingError = np.log(trainingError/numberOfRuns)
    testError = np.log(testError/numberOfRuns)
    return trainingError, testError

# =============================================================================
# Produces plots for Question 2 (i) and (ii)
# =============================================================================
x_range = np.linspace(0,1,100)
plt.plot(xpoints, ypoints,'o')
plt.plot(x_range, np.sin(2*np.pi*x_range)**2)
plt.title('2 (b)')
plt.show()

plotPolynomialFits(xpoints, ypoints)


# =============================================================================
# Produces plot for Question 2 (b)
# =============================================================================
w = fitPolynomials(18, xpoints, ypoints)
trainingError = np.log(calculateMeanSquareError(w, xpoints, ypoints))
kInts = [i+1 for i in range(18)]
plt.plot(kInts, trainingError)
plt.title('2 (a)(ii)')
plt.xlabel('Degree of Polynomial')
plt.ylabel('Log of training error')
plt.show()


# =============================================================================
# Produces plot for Question 2 (c)
# =============================================================================
calculateTestError(w, 1000)

# =============================================================================
# Producs plot for Question 2 (d)
# =============================================================================
training, test = averageTrainingAndTestError(100)
k = [i+1 for i in range(18)]
plt.plot(k, training, label='Training Error')
plt.plot(k, test, label='Test Error')
plt.title('2 (d)')
plt.xlabel('Degree of Polynomial')
plt.ylabel('Log of Average MSE')
plt.legend(loc='upper right')
plt.show()

# =============================================================================
# Question 3 - repeating (b) - (d) with sin basis.
# =============================================================================
def sinEval(w, x):
    """
    Function to evaluate the predicted values, using the coefficients w and the sin basis.
    
    Keyword Arguments:
        w -- the estimated regression coefficients found using a sin basis
        x -- the training or test points
    
    Returns:
        ypredicted -- the predicted value of y given points x and regression coefficients w.
    """
    ypredicted = 0
    w = np.flip(w)
    for i in range(0, np.size(w)):
        ypredicted += w[i]*np.sin((i+1)*np.pi*x)
    return ypredicted

w_2 = fitSinBasis(18, xpoints, ypoints)
trainingError_2 = np.log(calculateMeanSquareError(w_2, xpoints, ypoints, 0))
kInts = [i+1 for i in range(18)]
plt.plot(kInts, trainingError_2)
plt.title('3 (b)')
plt.xlabel('Basis Dimension')
plt.ylabel('Log of training error')
plt.show()

calculateTestError(w_2, 1000, 0)

#plotSinFits(xpoints, ypoints)
training_2, test_2 = averageTrainingAndTestError(100, 0)
k = [i+1 for i in range(18)]
plt.plot(k, training_2, label='Training Error')
plt.plot(k, test_2, label='Test Error')
#plt.title('3 (d)')
plt.xlabel('Basis Dimension')
plt.ylabel('Log of Average MSE')
plt.legend(loc='upper right')
plt.show()
