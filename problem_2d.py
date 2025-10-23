import numpy as np
import time

DIMENSIONS = [5, 10, 20]
NUM_RUNS = 500

try:
    A_orig = np.loadtxt("train.txt", delimiter=",")
    b_orig = np.loadtxt("train_values.txt", delimiter=",")
except FileNotFoundError:
    print("Error: train.txt or train_values.txt not found. Please ensure data files are present.")
    exit()

try:
    V_orig = np.loadtxt("validate.txt", delimiter=",")
    V_true_orig = np.loadtxt("validate_values.txt", delimiter=",")
except FileNotFoundError:
    print("Error: validate.txt or validate_values.txt not found. Please ensure data files are present.")
    exit()

N_train, M_orig = A_orig.shape
N_validate = V_orig.shape[0]

def classify(y):
    """Classifier function C(y) as defined in Problem 2(b)."""
    return 1 if y >= 0 else -1

def gaussian_embed(Data_orig, d):
    """
    Implements Gaussian Random Projection for Problem 2(d).
    Embeds data (N x M) into dimension d using R (M x d).
    """
    M = Data_orig.shape[1]
    R = np.random.normal(loc=0.0, scale=1.0/np.sqrt(d), size=(M, d))
    
    Data_embedded = Data_orig @ R
    return Data_embedded

def evaluate_model(X1, y_true, x_model):
    """
    Applies the linear model (x_model) to the data (X1) and calculates 
    the success rate against the true labels (y_true).
    """
    predictions_y = X1 @ x_model
    
    predictions_labels = np.array([classify(y) for y in predictions_y])
    
    correct_count = np.sum(predictions_labels == y_true)
    success_rate = correct_count / y_true.shape[0]
    return success_rate

final_results = {}

print("--- Running Problem 2(d): Gaussian Embedding and Least-Squares ---")
print(f"Running {NUM_RUNS} independent runs for each dimension...\n")

for d in DIMENSIONS:
    total_time = 0.0
    val_accuracy_list = []
    train_accuracy_list = []
    comparison_results = []
    
    for run in range(NUM_RUNS):
        start_time = time.perf_counter()
        
        A_embed = gaussian_embed(A_orig, d)
        V_embed = gaussian_embed(V_orig, d)
        
        A1_embed = np.hstack([A_embed, np.ones((N_train, 1))])
        
        Q, R = np.linalg.qr(A1_embed, mode='reduced')
        
        Qt_b = Q.T @ b_orig
        x = np.linalg.solve(R, Qt_b) 
        
        
        V1_embed = np.hstack([V_embed, np.ones((N_validate, 1))])
        val_success_rate = evaluate_model(V1_embed, V_true_orig, x)
        val_accuracy_list.append(val_success_rate)

        train_success_rate = evaluate_model(A1_embed, b_orig, x)
        train_accuracy_list.append(train_success_rate)
        
        if val_success_rate > train_success_rate:
            comparison_results.append("greater than")
        elif val_success_rate < train_success_rate:
            comparison_results.append("smaller than")
        else:
            comparison_results.append("equal to")

        end_time = time.perf_counter()
        total_time += (end_time - start_time)

    avg_time = total_time / NUM_RUNS
    
    avg_val_accuracy = np.mean(val_accuracy_list)
    avg_train_accuracy = np.mean(train_accuracy_list)
    
    from collections import Counter
    most_frequent_comparison = Counter(comparison_results).most_common(1)[0][0]
    
    final_results[d] = {
        'avg_time': avg_time,
        'avg_val_accuracy': avg_val_accuracy,
        'avg_train_accuracy': avg_train_accuracy,
        'comparison': most_frequent_comparison
    }

print("\n--- Problem 2(d) Complete Results (Gaussian Embedding) ---")
for d, res in final_results.items():
    print(f"\nDimension d={d}:")
    print(f"  Average Computational Time ({NUM_RUNS} runs): {res['avg_time']:.6f} seconds")
    print(f"  Average Validation Success Rate: {res['avg_val_accuracy'] * 100:.2f}%")
    print(f"  Average Training Success Rate: {res['avg_train_accuracy'] * 100:.2f}%")
    print(f"  Validation Success Rate is generally **{res['comparison']}** the Training Success Rate.")
