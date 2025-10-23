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

# Indentify the percentage of incorrectly classified samples


# === Step 1: Load validation features and true labels ===
V = np.loadtxt("validate.txt", delimiter=",")              # shape (N, 30)
true_labels = np.loadtxt("validate_values.txt", delimiter=",")  # shape (N,)

# === Step 2: Add intercept column to match model ===
V1 = np.hstack([V, np.ones((V.shape[0], 1))])  # shape (N, 31)

# === Step 3: Define classifier ===
def classify(y):
    return 1 if y >= 0 else -1

# === Step 4: Predict and compare ===
incorrect = 0
for i in range(V1.shape[0]):
    y = np.dot(x, V1[i])  # model prediction
    pred = classify(y)
    if pred != true_labels[i]:
        incorrect += 1

# === Step 5: Compute error percentage ===
total = V1.shape[0]
error_rate = (incorrect / total) * 100
print(f"\nIncorrectly classified samples: {incorrect} out of {total}")
print(f"Error rate: {error_rate:.2f}%")

# --- Completion for Problem 2(c) ---

# === Step 6: Evaluate model on TRAINING data ===
# A1 and b are already loaded from the beginning of the script

training_predictions_y = A1 @ x # Vector of predictions (y) for the training data
training_incorrect = 0

# Apply classifier and compare to true training labels (b)
for i in range(A1.shape[0]):
    y = training_predictions_y[i]
    pred = classify(y) # Uses the same classify function from Step 3
    if pred != b[i]:
        training_incorrect += 1

# === Step 7: Compute training error percentage ===
training_total = A1.shape[0] # Should be 300
training_error_rate = (training_incorrect / training_total) * 100

print(f"\n--- Training Data Analysis ---")
print(f"Incorrectly classified training samples: {training_incorrect} out of {training_total}")
print(f"Training Error rate: {training_error_rate:.2f}%")
print(f"Training Success rate: {100 - training_error_rate:.2f}%")


# === Step 8: Compare validation and training success rates (Final Answer to 2c) ===
validation_success_rate = 100 - error_rate
training_success_rate_calc = 100 - training_error_rate

comparison = ""
if validation_success_rate > training_success_rate_calc:
    comparison = "greater than"
elif validation_success_rate < training_success_rate_calc:
    comparison = "smaller than"
else:
    comparison = "equal to"

print(f"\n--- Comparison ---")
print(f"The success rate on the validation data ({validation_success_rate:.2f}%) is **{comparison}** the success rate on the training data ({training_success_rate_calc:.2f}%).")