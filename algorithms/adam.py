"""
File: adam.py
Description: Adam algorithm.
"""

import numpy as np

def adam(w0, loss_fn, grad_fn, X, y, lr=0.01, iters=1000, beta1=0.9, beta2=0.999, eps=1e-8, **loss_kwargs):
    """
    Adam algorithm.

    :param w0: initial weight, shape (n_samples, ).
    :param loss_fn: loss_fn: function(w, X, y) -> scalar loss (passed in from models)
    :param grad_fn: grad_fn: function(w, X, y) -> vector gradient (passed in from models)
    :param X: data matrix
    :param y: labels vector, shape (n_samples, ).
    :param lr: learning rate
    :param iters: number of iterations
    :param beta1: exponential decay rate for first moment (mean of gradients)
    :param beta2: exponential decay rate for second moment (mean of squared gradients)
    :param eps: small constant to prevent division by zero
    :param loss_kwargs: python convention for passing in an arbitrary argument
                        (logistic_ridge has an additional alpha parameter)
    :return w: final weights vector, shape (n_samples, )
    :return history: list of loss values
    """

    w = w0.copy().astype(float)

    m = np.zeros_like(w)    # first moment (mean of gradients)
    v = np.zeros_like(w)    # second moment (mean of squared gradients)

    history = []

    for t in range(1, iters+1):
        # gradient at current w
        g = grad_fn(w, X, y, **loss_kwargs)

        # update biased first and second moment estimates
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)

        # bias-corrected estimates
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # parameter update
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)

        # track loss
        loss = loss_fn(w, X, y, **loss_kwargs)
        history.append(loss)

    return w, history