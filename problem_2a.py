import numpy as np

A = np.loadtxt("train.txt", delimiter=",")
b = np.loadtxt("train_values.txt", delimiter=",")

print("A shape:", A.shape)
print("b shape:", b.shape)

A1 = np.hstack([A, np.ones((A.shape[0], 1))])

Q, R = np.linalg.qr(A1, mode='reduced')

Qt_b = Q.T @ b
x = np.linalg.solve(R, Qt_b)

print("\nModel coefficients (x):")
print(x)

residual = np.linalg.norm(A1 @ x - b)
print("\nResidual norm:", residual)
