import numpy as np


def householder(x):
    n = len(x)
    x /= np.linalg.norm(x, ord=np.Inf)
    v = np.zeros(n)
    v[1:] = x[1:]
    sigma = np.dot(x[1:], x[1:])
    if sigma == 0:
        beta = 0
    else:
        alpha = np.sqrt(sigma + x[0] ** 2)
        if x[0] <= 0:
            v[0] = x[0] - alpha
        else:
            v[0] = - sigma / (x[0] + alpha)
        beta = 2 * (v[0] ** 2) / (sigma + v[0] ** 2)
        v = v / v[0]

    return v, beta


def givens(a, b):
    if b == 0:
        c = 1
        s = 0
    else:
        if abs(b) > abs(a):
            tau = a / b
            s = 1 / np.sqrt(1 + tau ** 2)
            c = s * tau
        else:
            tau = b / a
            c = 1 / np.sqrt(1 + tau ** 2)
            s = c * tau
    return c, s


def make_tridiagnal(X):
    n = X.shape[0]
    mask = np.ones((n, n), dtype=bool)
    # 将除了三对角线以外的位置置为False
    mask[np.abs(np.arange(n) - np.arange(n)[:, np.newaxis]) > 1] = False
    # 将满足条件的元素置为0
    X[~mask] = 0
    return X


def clean_the_errors(X, epsilon):
    """ print("Cleaning the errors...")
    print(X)
    print("epsilon: ", epsilon)
    print(np.where(np.abs(X) < epsilon, 0, X)) """
    return np.where(np.abs(X) < epsilon, 0, X)


def givens_matrix(c, s, i, k, n):
    G = np.eye(n)
    G[i, i] = c
    G[k, k] = c
    G[i, k] = s if i < k else -s
    G[k, i] = -s if i < k else s
    return G