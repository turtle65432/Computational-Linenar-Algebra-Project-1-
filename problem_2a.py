import numpy as np

# === Step 1: Load A (features) and b (labels) ===
A = np.loadtxt("train.txt", delimiter=",")            # shape (300, 30)
b = np.loadtxt("train values.txt", delimiter=",")     # shape (300,)

print("A shape:", A.shape)
print("b shape:", b.shape)

# === Step 2: (Optional) add an intercept column ===
# This allows the model to have a bias term
A1 = np.hstack([A, np.ones((A.shape[0], 1))])  # shape (300, 31)

# === Step 3: Compute QR decomposition ===
# mode='reduced' gives Q:(n,k), R:(k,k) with k = min(n, m)
Q, R = np.linalg.qr(A1, mode='reduced')

# === Step 4: Solve R x = Q^T b for x ===
Qt_b = Q.T @ b
x = np.linalg.solve(R, Qt_b)

print("\nModel coefficients (x):")
print(x)

# === Step 5: (Optional) check residual norm ===
residual = np.linalg.norm(A1 @ x - b)
print("\nResidual norm:", residual)
