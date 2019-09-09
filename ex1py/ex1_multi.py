## Machine Learning Online Class
#  Exercise 1: Linear regression with multiple variables
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear regression exercise.
#
#  You will need to complete the following functions in this
#  exericse:
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
#  For this part of the exercise, you will need to change some
#  parts of the code below for various experiments (e.g., changing
#  learning rates).
#

## Initialization

# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
# from matplotlib import pyplot
import matplotlib.pyplot as plt

import featureNormalize
import computeCostMulti
import gradientDescentMulti
import normalEqnMulti
## ================ Part 1: Feature Normalization ================

print('Loading data ...\n')

data = np.loadtxt('ex1data2.txt', delimiter=',')
#data = np.loadtxt('housing.data')

examples, features = data.shape

X = data[:, :(features - 1)]
y = data[:, (features - 1)]

m = y.size  # number of training examples

# print out some data points
print('First 10 examples from the dataset: ')
print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
print('-'*26)

for i in range(10):
    print('{:10.6f}{:10.6f}{:10.6f}'.format(X[i, 0], X[i, 1], y[i]))

# Scale features and set them to zero mean
print('Normalizing Features ...')

[X, mu, sigma ] = featureNormalize.featureScaling(X)
print('Computed mean:', mu)
print('Computed standard deviation:', sigma)

## ================ Part 2: Gradient Descent ================

# ====================== YOUR CODE HERE ======================
# Instructions: We have provided you with the following starter
#               code that runs gradient descent with a particular
#               learning rate (alpha).
#
#               Your task is to first make sure that your functions -
#               computeCost and gradientDescent already work with
#               this starter code and support multiple variables.
#
#               After that, try running gradient descent with
#               different values of alpha and see which one gives
#               you the best result.
#
#               Finally, you should complete the code at the end
#               to predict the price of a 1650 sq-ft, 3 br house.
#
# Hint: By using the 'hold on' command, you can plot multiple
#       graphs on the same figure.
#
# Hint: At prediction, make sure you do the same feature normalization.
#


print('Running gradient descent ...\n');

#X = np.c_[np.ones((X.shape[0],1)), X[:,0:2]]
X = np.concatenate([np.ones((m, 1)), X], axis=1)

for i in range(10):
    print('{:10.6f}{:10.6f}{:10.6f}'.format(X[i, 0], X[i, 1], X[i, 2]))

# initialize fitting parameters
theta=np.zeros(X.shape[1])

# compute and display initial cost
J = computeCostMulti.jOfThetaMulti(X, y, theta)

# Init Theta and Run Gradient Descent
# Choose some alpha value
alpha = 0.1
num_iters = 50

# Init Theta and Run Gradient Descent
#n = size(X,2);


# run gradient descent
[theta, J_history] = gradientDescentMulti.CalGradientDescentMuti(X, y, theta, alpha, num_iters)

# print theta to screen
print('Theta computed from gradient descent: ')
print(theta)

# plot cost functin versus Iteration Number
plt.figure('Cost Function vs Number of Iterations')
plt.plot(J_history, label = 'Convergence of CostJ')
plt.legend()
plt.ylabel('Cost J')
plt.xlabel('Iterations')
plt.show()


# Estimate the price of a 1650 sq-ft, 3 br house

# ====================== YOUR CODE HERE ======================
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.

X_array = [1, 1650, 3]
#X_array = [1, 0.08014, 0.00, 5.960, 0, 0.4990, 5.8500, 41.50, 3.9342, 5, 279.0, 19.20, 396.90, 8.77]

predict = np.copy(X_array)

X_array[1:features] = (X_array[1:features] - mu) / sigma

price = np.dot(X_array, theta)
# ============================================================

print('Predicted price of a 1650 sq-ft, 3 br house using gradient descent : ${:0,.6f}'.format(price))

## ================ Part 3: Normal Equations ================

print('Solving with normal equations...\n');

# ====================== YOUR CODE HERE ======================
# Instructions: The following code computes the closed form
#               solution for linear regression using the normal
#               equations. You should complete the code in
#               normalEqn.m
#
#               After doing so, you should complete this code
#               to predict the price of a 1650 sq-ft, 3 br house.
#

# Load data

data = np.loadtxt('ex1data2.txt', delimiter=',')
#data = np.loadtxt('housing.data')

examples, features = data.shape

X = data[:, :(features - 1)]
y = data[:, (features - 1)]

# Add intercept term to X
X = np.concatenate([np.ones((m, 1)), X], axis=1)

m = y.size

# Calculate the parameters from the normal equation
theta = normalEqnMulti.normalEqn(X, y)

# Display normal equation's result
print('Theta computed from the normal equations: \n')
print(theta)


# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
price = np.dot(predict, theta)

# ============================================================

print('Predicted price of a 1650 sq-ft, 3 br house using normal equations:  ${:0,.6f}'.format(price))
