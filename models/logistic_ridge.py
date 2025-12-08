"""
File: logistic_ridge.py
Description: A ridge logistic regression model.
"""

import numpy as np

from models.logistic import logistic_loss, logistic_grad, logistic_hess

def logistic_ridge_loss(w, X, y, alpha=1.0):
    """
    Ridge logistic regression loss function -> logistic loss function with a penalty term.
    equation: J(w) = logistic_loss(w) + (alpha/2) * ||w_no_bias||^2
    logistic_loss(w) -> J(w) = -1/n * sum [ y_i log(p_i) + (1 - y_i) log(1 - p_i) ]

    :param w: weight vector, shape (n_features, ).
    :param X: data matrix, shape (n_samples, n_features).
    :param y: labels vector, shape (n_samples, ).
    :param alpha: penalty term.
    :return: base + penalty term.
    """

    # splitting the weights -> bias and the rest
    # bias (intercept): is not penalized
    # feature_weights: remaining weights (w1, w2, ... wd) are penalized
    bias = w[-1]                # last weight, added for clarity
    feature_weights = w[: -1]   # the rest

    base = logistic_loss(w, X, y)

    # ridge penalty: (alpha/2) * sum(w_i ^2)
    # alpha/2 -> by dividing alpha by 2 it simplifies the math for optimization
    penalty = (alpha / 2) * np.dot(feature_weights, feature_weights)

    return base + penalty

def logistic_ridge_grad(w, X, y, alpha=1.0):
    """
    Gradient of the ridge logistic regression.
    :param w: weight vector, shape (n_features, ).
    :param X: data matrix, shape (n_samples, n_features).
    :param y: labels vector, shape (n_samples, ).
    :param alpha: ridge penalty term.
    :return: grad
    """

    # splitting the weights
    bias = w[-1]            # added for clarity
    feature_weights = w[: -1]

    # gradient of the base logistic model
    grad = logistic_grad(w, X, y)

    # add alpha * wi to every penalized weight
    grad[ : -1] += alpha * feature_weights

    return grad

def logistic_ridge_hess(w, X, y, alpha=1.0):
    """
    Hessian of the ridge logistic regression.
    :param w: weight vector, shape (n_features, ).
    :param X: data matrix, shape (n_samples, n_features).
    :param y: labels vector, shape (n_samples, ).
    :param alpha: ridge penalty term.
    :return H: Hessian matrix, shape (n_features, n_features).
    """

    H = logistic_hess(w, X, y)

    # add alpha * I to the penalized weights (all except bias)
    d = H.shape[0]
    H[ :-1, :-1] += alpha * np.eye(d-1)

    return H