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

V = np.loadtxt("validate.txt", delimiter=",")
true_labels = np.loadtxt("validate_values.txt", delimiter=",")

V1 = np.hstack([V, np.ones((V.shape[0], 1))])

def classify(y):
    return 1 if y >= 0 else -1

incorrect = 0
for i in range(V1.shape[0]):
    y = np.dot(x, V1[i])
    pred = classify(y)
    if pred != true_labels[i]:
        incorrect += 1

total = V1.shape[0]
error_rate = (incorrect / total) * 100
print(f"\nIncorrectly classified samples: {incorrect} out of {total}")
print(f"Error rate: {error_rate:.2f}%")

training_predictions_y = A1 @ x
training_incorrect = 0

for i in range(A1.shape[0]):
    y = training_predictions_y[i]
    pred = classify(y)
    if pred != b[i]:
        training_incorrect += 1

training_total = A1.shape[0]
training_error_rate = (training_incorrect / training_total) * 100

print(f"\n--- Training Data Analysis ---")
print(f"Incorrectly classified training samples: {training_incorrect} out of {training_total}")
print(f"Training Error rate: {training_error_rate:.2f}%")
print(f"Training Success rate: {100 - training_error_rate:.2f}%")

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