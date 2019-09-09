#  Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     warmUpExercise.py
#     plotData.py
#     gradientDescent.py
#     computeCost.py
#     gradientDescentMulti.py
#     computeCostMulti.py
#     featureNormalize.py
#     normalEqn.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s
#

## Initialization

# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces

from sklearn.linear_model import LinearRegression

import warmUpExercise
import plotData
import computeCost
import gradientDescent

# ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.py
print('Running warmUpExercise ... ')
print('5x5 Identity Matrix: \n')
print(warmUpExercise.identityMatrix())

# ======================= Part 2: Plotting =======================
print('\nPlotting Data ...\n')
# Read comma separated data
data = np.loadtxt('C:\Octave\ex1py\ex1data1.txt', delimiter=',')
X, y = data[:, 0], data[:, 1]

m = y.size  # number of training examples

# Plot Data
# Note: You have to complete the code in plotData.py
plt.figure('Data Points and  Hypothesis')
plotData.myPlot(X, y)

# =================== Part 3: Gradient descent ===================
print('Running Gradient Descent ...\n')

X = np.stack([np.ones(m), X], axis=1)
# initialize fitting parameters
theta=np.zeros(X.shape[1])

# Some gradient descent settings

iterations = 1800
alpha = 0.01


# compute and display initial cost
J = computeCost.jOfTheta(X, y, theta)
print('Initial Cost :  ', J)

# run gradient descent
[theta, J_history] = gradientDescent.CalGradientDescent(X, y, theta, alpha, iterations);

# print theta to screen
print('Theta found by gradient descent: ', theta)

# plot the linear fit
#plotData.myPlot(X[:, 1], y)
plt.plot(X[:, 1], np.dot(X, theta), '-', label = 'Linear regression (Gradient Descent)')

# Compare with Scikit-learn Linear regression (Optional)
regr = LinearRegression()
regr.fit(X[:,1].reshape(-1,1), y.ravel())
plt.plot(X[:, 1], regr.intercept_+regr.coef_*X[:, 1], label='Linear regression (Scikit-learn GLM)')

plt.legend()
plt.show()


# plot cost functin versus Iteration Number
plt.figure('Cost Function vs Number of Iterations')
plt.plot(J_history, label = 'Convergence of CostJ')
plt.legend()
plt.ylabel('Cost J')
plt.xlabel('Iterations')
plt.show()


# Predict values for population sizes of 35,000 and 70,000
predict1 = theta.T.dot([1, 3.5])*10000
predict2 = theta.T.dot([1, 7])*10000

print('For population = 35,000, we predict a profit of ', predict1)
print('For population = 70,000, we predict a profit of ', predict2)

#  ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...\n')

# Grid over which we will calculate J
theta0_vals = np.linspace(-30, 30, 100)
theta1_vals = np.linspace(-1, 4, 100)


# initialize J_vals to a matrix of 0's
J_vals = np.zeros((theta0_vals.size, theta1_vals.size))

# Fill out J_vals
for i, theta0 in enumerate(theta0_vals):
    for j, theta1 in enumerate(theta1_vals):
        t = [theta0, theta1]
        J_vals[i, j] = computeCost.jOfTheta(X, y, t)


# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.T;
# Surface plot
fig = plt.figure(num='Surface Plot for Cost Function',figsize=(12, 5))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
ax.set_zlabel('J of theta / Cost Function')
plt.title('Surface Plot for Cost Function')
plt.show()

# Contour plot
#figure;
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
ax = pyplot.subplot(122)
pyplot.contour(theta0_vals, theta1_vals, J_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
pyplot.xlabel('theta0')
pyplot.ylabel('theta1')
pyplot.plot(theta[0], theta[1], 'ro', ms=10, lw=2)
pyplot.title('Contour, Showing Minimum Cost Function')
plt.show()
pass
