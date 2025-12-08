"""
File: logistic.py
Description: A logistic regression model.
"""

import numpy as np


def sigmoid(z):
    """
    sigmoid function -> sigma(z) = 1 / (1 + exp(-z)).
    :param z: scalar or numpy array of shape (n_samples, ).
    :returns: same shape as z, values in (0,1).
    """
    # modified to avoid overflow when testing differing initial conditions
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def logistic_loss(w, X, y):
    """
    logistic regression loss function.
    :param w: weight vector, shape (n_features, ).
    :param X: data matrix, shape (n_samples, n_features).
    :param y: labels vector, shape (n_samples, ).
    """

    # linear scores: z_i = w^T x_i
    # one score per data point
    z = X @ w               # shape (n_samples, )

    # predicted probabilities: p_i = sigmoid(z_i)
    # one probability per data point
    p = sigmoid(z)          # shape (n_samples, )

    # small epsilon to avoid log(0)
    eps = 1e-9

    # cross-entropy:
    # J(w) = -1/n * sum [ y_i log(p_i) + (1 - y_i) log(1 - p_i) ]
    loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

    return loss

def logistic_grad(w, X, y):
    """
    gradient of the logistic loss function with respect to w (weights).
    :param w: weight vector, shape (n_features, ).
    :param X: data matrix, shape (n_samples, n_features).
    :param y: vector of true labels, shape (n_samples, ).
    :return: gradient: shape (n_features, ).
    """

    # linear scores
    z = X @ w               # (n_samples, )

    # predicted probabilities vector
    p = sigmoid(z)          # (n_samples, )

    # error vector: p_i - y_i
    error = p - y           # (n_samples, )

    # gradient formula: (1/n) * X^T (p-y)
    # dividing by len(y) gets the average
    grad = X.T @ (error / len(y))       # (n_features, )

    return grad

def logistic_hess(w, X, y):
    """
    Hessian of the logistic loss with respect to w (weights).
    :param w: weight vector, shape (n_features, ).
    :param X: data matrix, shape (n_samples, n_features).
    :param y: vector of true labels, shape (n_samples, ).
    :return H: Hessian matrix, shape (n_features, n_features).
    """

    # linear scores
    z = X @ w               # (n_samples, )

    # predicted probabilities
    p = sigmoid(z)          # (n_samples, )

    # si = pi * (1-pi)
    s = p * (1 - p)         # (n_samples, )

    # want H = (1/n) X^T diag(s) X
    # use broadcasing instead of building diagonals
    # X^T * s has shape (n_features, n_samples)
    n = len(y)
    H = (X.T * s) @ X / n   # (n_features, n_features)

    return H