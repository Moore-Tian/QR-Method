import numpy as np
from utils import *


class symmetric_QR():
    def __init__(self, max_iter=1000, epsilon=1e-6):
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.record = []

    def __call__(self, matrix):
        self.matrix = matrix
        T, Q = self.full_QR()
        self.T = T
        self.Q = Q
        return T, Q

    # based on householder
    def tridiagnal_decomposition(self, matrix):
        n = matrix.shape[0]
        Q = np.eye(n)

        for k in range(1, n-1):
            x = matrix[k:, k - 1].copy()
            v, beta = householder(np.squeeze(x))
            H = np.eye(n)
            H[k:, k:] = np.eye(n - k) - beta * np.outer(v, v)
            u = beta * np.dot(matrix[k:, k:], v)
            w = u - beta * np.dot(u, v) * v / 2
            matrix[k, k - 1] = np.linalg.norm(matrix[k:, k - 1])
            matrix[k - 1, k] = matrix[k, k - 1]
            matrix[k:, k:] -= (np.outer(v, w) + np.outer(w, v))
            Q = np.dot(Q, H)
        matrix = make_tridiagnal(matrix)
        return matrix, Q

    def QR_iteration(self, matrix):
        n = matrix.shape[0]
        d = (matrix[n - 2, n - 2] - matrix[n - 1, n - 1]) / 2
        u = matrix[n - 1, n - 1] - matrix[n - 1, n - 2] ** 2 / (d + np.sign(d) * np.sqrt(d ** 2 + matrix[n - 1, n - 2] ** 2))
        x = matrix[0, 0] - u
        z = matrix[1, 0]
        G_total = np.eye(n)
        for k in range(1, n):
            c, s = givens(x, z)
            G = givens_matrix(c, s, k - 1, k, n)
            matrix = np.dot(G, np.dot(matrix, G.T))
            matrix = clean_the_errors(matrix, self.epsilon)
            if k < n - 1:
                x = matrix[k, k - 1]
                z = matrix[k + 1, k - 1]
            G_total = np.dot(G, G_total)
        return matrix, G_total

    def converge_deter(self, matrix):
        n = matrix.shape[0]
        for i in range(n - 1):
            if abs(matrix[i + 1, i]) <= self.epsilon * (abs(matrix[i, i]) + abs(matrix[i + 1, i + 1])):
                matrix[i + 1, i], matrix[i, i + 1] = 0, 0
        m = n
        li = -1
        while matrix[m - 2, m - 1] == 0 and m > 0:
            m -= 1

        if m == 1:
            m -= 1

        while ~ np.all(np.diag(matrix[li + 1:m, li + 1:m], k=1) != 0):
            li += 1

        self.record.append(m - li)

        if m == (li + 1):
            return True, m, li
        else:
            return False, m, li
        
        

    def full_QR(self):
        matrix = self.matrix
        n = matrix.shape[0]
        T, Q = self.tridiagnal_decomposition(matrix)
        for _ in range(self.max_iter):
            judge, m, li = self.converge_deter(T)
            if judge:
                return T, Q
            T[li + 1:m + 1, li + 1:m + 1], G = self.QR_iteration(T[li + 1:m + 1, li + 1:m + 1])
            G_full = np.eye(n)
            G_full[li + 1:m + 1, li + 1:m + 1] = G
            Q = np.dot(Q, G_full.T)
        return T, Q
