"""
File: newton.py
Description: Newton's method for optimization.
"""

import numpy as np

def newton(w0, loss_fn, grad_fn, hess_fn, X, y, iters = 20, tol = 1e-8, **loss_kwargs):
    """
    Newton method algorithm.
    w_(k+1) = w_k - H(w_k)^-1 grad(w_k)
    :param w0: initial weight
    :param loss_fn: function(w, X, y) -> scalar loss (passed in from models)
    :param grad_fn: function(w, X, y) -> vector gradient (passed in from models)
    :param hess_fn:
    :param X: data matrix
    :param y: labels vector, shape (n_samples, ).
    :param iters: number of iterations
    :param tol: tolerance for convergence, stop if ||grad(w_k)|| < tol
    :param loss_kwargs: python convention for passing in an arbitrary argument
                        (logistic_ridge has an additional alpha parameter)
    :return w: final weights vector, shape (n_samples, ):
    :return history: list of loss values
    """
    w = w0.copy().astype(float)

    history = []

    for k in range(iters):
        loss = loss_fn(w, X, y, **loss_kwargs)
        g = grad_fn(w, X, y, **loss_kwargs)
        H = hess_fn(w, X, y, **loss_kwargs)

        history.append(loss)

        grad_norm = np.linalg.norm(g)
        if grad_norm < tol:
            break

        # Solve H p = g
        try:
            p = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            # if Hessian is singular fall back to gradient step
            p = g

        w = w - p

    return w, history