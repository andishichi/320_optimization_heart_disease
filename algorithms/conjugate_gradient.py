"""
File: conjugate_gradient.py
Description: A conjugate gradient algorithm.
"""

import numpy as np

def conjugate_gradient(w0, loss_fn, grad_fn, X, y, iters=100, tol=1e-6, c1=1e-6, backtracking_beta=0.5, **loss_kwargs):
    """
    Nonlinear conjugate gradient solver.
    d_0 = -g_0
    w_(k+1) = w_k + t_k d_k
    beta_k = max(0, (g_(k+)}^T (g_(k+1) - g_k)) / (g_k^T g_k))
    d_(k+1) = -g_(k+1) + beta_k d_k
    :param w0: initial weight
    :param loss_fn: function(w, X, y) -> scalar loss (passed in from models)
    :param grad_fn: function(w, X, y) -> vector gradient (passed in from models)
    :param X: data matrix
    :param y: labels vector, shape (n_samples, ).
    :param iters: number of iterations
    :param tol: tolerance for convergence, stop if ||grad(w_k)|| < tol
    :param c1: Armijo parameter for line search (small positive)
    :param backtracking_beta: shrink factor for step size in backtracking (0 < beta < 1)
    :param loss_kwargs: python convention for passing in an arbitrary argument
                        (logistic_ridge has an additional alpha parameter)
    :return w: final weights vector, shape (n_samples, ):
    :return history: list of loss values
    """

    w = w0.copy().astype(float)

    # initial loss and gradient
    loss = loss_fn(w, X, y, **loss_kwargs)
    g = grad_fn(w, X, y, **loss_kwargs)

    # initial search direction: steepest descent
    d = -g

    history = []

    for k in range(iters):
        history.append(loss)

        grad_norm = np.linalg.norm(g)
        if grad_norm < tol:
            break

        # Armijo backtracking line search along d
        t = 1.0
        g_dot_d = g.dot(d)

        # if not descent direction, reset to steepest descent
        if g_dot_d >= 0:
            d = -g
            g_dot_d = g.dot(d)

        while True:
            w_new = w + t * d
            loss_new = loss_fn(w_new, X, y, **loss_kwargs)

            # Armijo condition: f(w + t d) <= f(w) + c1 t g^T d
            if loss_new <= loss + c1 * t * g_dot_d:
                break

            t *= backtracking_beta
            if t < 1e-8:
                # if step size too small, don't update
                break

        # update position
        s = w_new - w
        w = w_new

        # new gradient
        g_new = grad_fn(w, X, y, **loss_kwargs)

        # conjugate gradient beta
        y_vec = g_new - g
        denom = g.dot(g)
        if denom > 0:
            beta = max(0.0, g_new.dot(y_vec) / denom)
        else:
            beta = 0.0

        # new search direction
        d = -g_new + beta * d

        # move grad/loss forward
        g = g_new
        loss = loss_new

    return w, history