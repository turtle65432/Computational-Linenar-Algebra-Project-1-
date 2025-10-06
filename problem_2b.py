import numpy as np

# Replace with your actual model weights and bias
w = np.array([...])  # shape (30,)
b = ...              # scalar bias term

# Define classifier function
def classifier(x):
    y = np.dot(w, x) + b
    return 1 if y >= 0 else -1

# Read and classify each sample from validate.txt
with open('validate.txt', 'r') as file:
    for line_num, line in enumerate(file, start=1):
        try:
            x = np.array([float(val) for val in line.strip().split(',')])
            if x.shape[0] != w.shape[0]:
                raise ValueError(f"Feature mismatch at line {line_num}")
            label = classifier(x)
            print(f"Sample {line_num}: Prediction = {label}")
        except Exception as e:
            print(f"Error at line {line_num}: {e}")
