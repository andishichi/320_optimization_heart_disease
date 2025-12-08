"""
File: gradient_descent.py
Description: A gradient descent algorithm.
"""

import numpy as np

def gradient_descent(w0, loss_fn, grad_fn, X, y, lr=0.01, iters=1000, **loss_kwargs):
    """
    Basic gradient descent algorithm.
    :param w0: initial weight
    :param loss_fn: function(w, X, y) -> scalar loss (passed in from models)
    :param grad_fn: function(w, X, y) -> vector gradient (passed in from models)
    :param X: data matrix
    :param y: labels vector, shape (n_samples, ).
    :param lr: learning rate
    :param iters: number of iterations
    :param loss_kwargs: python convention for passing in an arbitrary argument
                        (logistic_ridge has an additional alpha parameter)
    :return w: final weights vector, shape (n_samples, )
    :return history: list of loss values
    """
    w = w0.copy()   # copy so we don't change the original weight

    history = []    # save loss values

    for _ in range(iters):
        grad = grad_fn(w, X, y)
        w -= lr * grad
        history.append(loss_fn(w, X, y))

    return w, history