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

V = np.loadtxt("validate.txt", delimiter=",")

V1 = np.hstack([V, np.ones((V.shape[0], 1))])

def classify(y):
    return 1 if y >= 0 else -1

predictions = []
for i in range(V1.shape[0]):
    y = np.dot(x, V1[i])
    label = classify(y)
    predictions.append(label)
    print(f"Sample {i+1}: Prediction = {label}")

V_true = np.loadtxt("validate_values.txt", delimiter=",")

predictions_array = np.array(predictions)

incorrect_count = np.sum(predictions_array != V_true)

total_samples = V_true.shape[0]
incorrect_percentage = (incorrect_count / total_samples) * 100

print(f"\nTotal validation samples: {total_samples}")
print(f"Number of incorrect classifications: {incorrect_count}")
print(f"Percentage of samples incorrectly classified: {incorrect_percentage:.2f}%")