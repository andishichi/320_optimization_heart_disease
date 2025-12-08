"""
File: bfgs.py
Description: A BFGS algorithm.
"""

import numpy as np

def bfgs(w0, loss_fn, grad_fn, X, y, iters = 100, tol=1e-6, c1=1e-4, backtracking_beta=0.5, **loss_kwargs):
    """
    BFGS quasi-Newton method algorithm.
    w_(k+1) = w_k - t_k * p_k
    p_k = -H_k * grad_k
    BFGS formula will update H_k
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
    n_features = w.shape[0]

    # start with identity matrix as inverse Hessian approximation
    H = np.eye(n_features)

    history = []

    # initial gradient and loss
    loss = loss_fn(w, X, y, **loss_kwargs)
    g = grad_fn(w, X, y, **loss_kwargs)

    for k in range(iters):
        history.append(loss)

        grad_norm = np.linalg.norm(g)
        if grad_norm < tol:
            break

        # search direction: p_k = - H_k * g_k
        p = -H @ g

        # Armijo backtracking line search
        t = 1.0
        g_dot_p = g.dot(p)

        # if not the descent direction, reset H and try steepest descent
        if g_dot_p >= 0:
            H = np.eye(n_features)
            p = -g
            g_dot_p = g.dot(p)

        while True:
            w_new = w + t * p
            loss_new = loss_fn(w_new, X, y, **loss_kwargs)

            # Armijo condition:
            # f(w + t d) <= f(w) + c1 * t * grad^T p
            if loss_new <= loss + c1 * t * g_dot_p:
                break

            t *= backtracking_beta
            if t < 1e-8:
                # if step is too small, don't update
                break

        # s_k = w_(k+1) - w_k
        s = w_new - w

        # new gradient
        g_new = grad_fn(w_new, X, y, **loss_kwargs)

        # y_k = g_(k+1) - g_k
        y_vec = g_new - g

        ys = y_vec.dot(s)

        # update H using BFGS formula if curvature condition holds
        if ys > 1e-10:
            rho = 1.0 / ys

            # I - rho * s y^T
            I = np.eye(n_features)
            V = I - rho * np.outer(s, y_vec)

            # BFGS inverse Hessian update
            # H_(k+1) = V H_k V^T + rho * s s^T
            H = V @ H @ V.T + rho * np.outer(s, s)

        else:
            # if curvature condition fails, reset H to identity
            H = np.eye(n_features)

        # move to new point
        w = w_new
        g = g_new
        loss = loss_new

    return w, history

