import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# =============================================================================
# Question 1
# =============================================================================
points = np.array([(1,3),(2, 2),(3, 0),(4, 5)])
x = points[:,0]
y = np.asmatrix(points[:,1]).transpose()
yReal = points[:,1]

# producing the phi matrices (using basis functions to transform data set X)
phi4 = np.matrix([[1,1,1,1], [1,2,4,8], [1,3,9,27], [1,4,16,64]])
phi3 = np.matrix([[1,1,1], [1,2,4], [1,3,9], [1,4,16]])
phi2 = np.matrix([[1,1], [1,2], [1,3], [1,4]])
phi1 = np.matrix([[1], [1], [1], [1]])

# using normal equations to find coefficients.
w4 = np.flip((np.linalg.inv((phi4.transpose() * phi4))*(phi4.transpose())*y).A1)
w3 = np.flip((np.linalg.inv((phi3.transpose() * phi3))*(phi3.transpose())*y).A1)
w2 = np.flip((np.linalg.inv((phi2.transpose() * phi2))*(phi2.transpose())*y).A1)
w1 = np.flip((np.linalg.inv((phi1.transpose() * phi1))*(phi1.transpose())*y).A1)

# evaluating the polynomial with w coefficients for x in some range
x_points = np.linspace(0, 5, 50)
y4_points = np.polyval(w4, x_points)
y3_points = np.polyval(w3, x_points)
y2_points = np.polyval(w2, x_points)
y1_points = np.polyval(w1, x_points)

# part (a)
plt.plot(x,y,'o')
plt.plot(x_points, y4_points, label='k=4')
plt.plot(x_points, y3_points, label='k=3')
plt.plot(x_points, y2_points, label='k=2')
plt.plot(x_points, y1_points, label='k=1')
plt.legend(loc='upper left')
plt.xlim([0, 5])
plt.ylim([-5, 15])
plt.show()

# part (b)
print("Equations corresponding to curves for k = 1, 2, 3, 4: ")
print("{:.2f} + {:.2f}x + {:.2f}x^2 + {:.2f}x^3" .format(w4[3], w4[2], w4[1], w4[0]))
print("{:.2f} + {:.2f}x + {:.2f}x^2" .format(w3[2], w3[1], w3[0]))
print("{:.2f} + {:.2f}x" .format(w2[1], w2[0]))
print("{:.2f}" .format(w1[0]))

# part (c)
MSE4 = np.sum((np.polyval(w4,x) - yReal)**2)/4
MSE3 = np.sum((np.polyval(w3,x) - yReal)**2)/4
MSE2 = np.sum((np.polyval(w2,x) - yReal)**2)/4
MSE1 = np.sum((np.polyval(w1,x) - yReal)**2)/4

print("\nMSE for different k values: ")
print(round(MSE4,3), '  MSE for k = 4')
print(round(MSE3,3), '  MSE for k = 3')
print(round(MSE2,3), ' MSE for k = 2')
print(round(MSE1,3), ' MSE for k = 1')
