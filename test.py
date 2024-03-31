from symmetric_QR_method import symmetric_QR
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm


np.random.seed(0)

max_iter = 100
epsilon = 1e-16
# 指定矩阵的维度
n = 10

# 生成随机的上三角矩阵
upper_triangular = np.random.rand(n, n)

# 构造对称矩阵
symmetric_matrix = upper_triangular + upper_triangular.T - np.diag(upper_triangular.diagonal())
ori_symmetric_matrix = symmetric_matrix.copy()

QR_solver = symmetric_QR(max_iter, epsilon)
eigen_values, eigen_vectors = QR_solver(symmetric_matrix)
print(eigen_values)
print(eigen_vectors)

error = ori_symmetric_matrix - np.dot(eigen_vectors, np.dot(eigen_values, eigen_vectors.T))
print(np.linalg.norm(error))



time_list = []
error_list = []
# 算法复杂度以及误差的可视化
for n in tqdm(range(1, 500, 50)):
    upper_A = np.random.rand(n, n)
    A = upper_A + upper_A.T - np.diag(upper_A.diagonal())s
    ori_A = A.copy()
    start_time = time.time()
    v, Q = QR_solver(A)
    end_time = time.time()
    time_list.append(end_time - start_time)
    print(np.linalg.norm(ori_A - np.dot(Q, np.dot(v, Q.T))))
    error_list.append(np.linalg.norm(ori_A - np.dot(Q, np.dot(v, Q.T))))


x = [i for i in range(1, 500, 50)]
plt.plot(x, time_list)
plt.title("time")
plt.xlabel("dimension")
plt.ylabel("time")
plt.show()

plt.plot(x, error_list)
plt.title("error")
plt.xlabel("dimension")
plt.ylabel("error")
plt.show()

print(error_list)
