import numpy as np

# === Step 1: Load A (features) and b (labels) ===
A = np.loadtxt("train.txt", delimiter=",")            # shape (300, 30)
b = np.loadtxt("train_values.txt", delimiter=",")     # shape (300,)

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



# === Step 6: Load validation data ===
V = np.loadtxt("validate.txt", delimiter=",")  # shape (N, 30)

# === Step 7: Add intercept column to match model ===
V1 = np.hstack([V, np.ones((V.shape[0], 1))])  # shape (N, 31)

# === Step 8: Define classifier function ===
def classify(y):
    return 1 if y >= 0 else -1

# === Step 9: Apply model and classify ===
predictions = []
for i in range(V1.shape[0]):
    y = np.dot(x, V1[i])  # model prediction
    label = classify(y)
    predictions.append(label)
    print(f"Sample {i+1}: Prediction = {label}")

# === Step 10: Load true validation labels ===
V_true = np.loadtxt("validate_values.txt", delimiter=",") # shape (N,)

# === Step 11: Convert predictions to a numpy array for easy comparison ===
predictions_array = np.array(predictions)

# === Step 12: Calculate incorrect classifications (Problem 2c) ===
# Incorrect classification occurs when prediction != true label
incorrect_count = np.sum(predictions_array != V_true)

# === Step 13: Calculate percentage of incorrect samples (Problem 2c) ===
total_samples = V_true.shape[0]
incorrect_percentage = (incorrect_count / total_samples) * 100

print(f"\nTotal validation samples: {total_samples}")
print(f"Number of incorrect classifications: {incorrect_count}")
print(f"Percentage of samples incorrectly classified: {incorrect_percentage:.2f}%")