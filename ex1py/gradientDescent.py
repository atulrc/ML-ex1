# Scientific and vector computation for python
import numpy as np
import computeCost

def CalGradientDescent(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn `theta`. Updates theta by taking `num_iters`
    gradient steps with learning rate `alpha`.

    Parameters
    ----------
    X : array_like
        The input dataset of shape (m x n+1).

    y : arra_like
        Value at given features. A vector of shape (m, ).

    theta : array_like
        Initial values for the linear regression parameters.
        A vector of shape (n+1, ).

    alpha : float
        The learning rate.

    num_iters : int
        The number of iterations for gradient descent.

    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).

    J_history : list
        A python list for the values of the cost function after each iteration.

    Instructions
    ------------
    Peform a single gradient step on the parameter vector theta.

    While debugging, it can be useful to print out the values of
    the cost function (computeCost) and gradient here.
    """
    # Initialize some useful values
    m = y.shape[0]  # number of training examples
    print('Number of Training Examples: ', m)

    J_history = np.zeros(num_iters) # initialize J(theta) history

    # make a copy of theta, to avoid changing the original array, since numpy arrays
    # are passed by reference to functions
    theta = theta.copy()

    J_history = [] # Use a python list to save cost in every iteration

    for i in np.arange(num_iters):
        # ==================== YOUR CODE HERE =================================
        htheta = X.dot(theta)
        theta = theta - alpha * (1/ m) * (X.T.dot(htheta - y))
        # =====================================================================
        # save the cost J in every iteration
        J_history.append(computeCost.jOfTheta(X, y, theta))

    return theta, J_history
