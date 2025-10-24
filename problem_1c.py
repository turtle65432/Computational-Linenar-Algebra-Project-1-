import numpy as np
from sklearn.cluster import KMeans
import time

train_data = np.loadtxt('train.txt', delimiter=',')
train_labels = np.loadtxt('train_values.txt')
validate_data = np.loadtxt('validate.txt', delimiter=',')
validate_labels = np.loadtxt('validate_values.txt')

print(f"Training data shape: {train_data.shape}")
print(f"Validation data shape: {validate_data.shape}")

def sparse_random_projection(data, target_dim, density = 1/3):
    n_features = data.shape[1]

    random_values = np.random.rand(n_features, target_dim)
    random_matrix = np.zeros((n_features, target_dim))

    random_matrix[random_values < density/2] = 1
    random_matrix[random_values > 1 - density/2] = -1

    scale = np.sqrt(1/density) / np.sqrt(target_dim)
    random_matrix *= scale

    return data @ random_matrix

def evaluate_clustering(train_embedded, validate_embedded, train_labels, validate_labels):
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(train_embedded)

    train_clusters = kmeans.labels_
    cluster_0_labels = train_labels[train_clusters == 0]
    cluster_1_labels = train_labels[train_clusters == 1]
    
    cluster_0_mean = np.mean(cluster_0_labels)
    cluster_1_mean = np.mean(cluster_1_labels)

    if cluster_0_mean > cluster_1_mean:
        cluster_to_label = {0: 1, 1: -1}
    else:
        cluster_to_label = {0: -1, 1: 1}

    validate_clusters = kmeans.predict(validate_embedded)
    predictions = np.array([cluster_to_label[c] for c in validate_clusters])
    accuracy = np.mean(predictions == validate_labels)

    return accuracy

dimensions = [10, 20, 25]
n_runs = 500

print("\n" + "-"*60)
print("K-means with Sparse Random Projection")
print("-"*60)

results = {}

for d in dimensions:
    print(f"\nDimension d={d}:")
    print(f"Running {n_runs} independent trials...")
    
    accuracies = []
    times = []
    
    for run in range(n_runs):
        start_time = time.time()

        train_embedded = sparse_random_projection(train_data, d)
        validate_embedded = sparse_random_projection(validate_data, d)

        accuracy = evaluate_clustering(train_embedded, validate_embedded, train_labels, validate_labels)
        end_time = time.time()
        
        accuracies.append(accuracy)
        times.append(end_time - start_time)

    avg_accuracy = np.mean(accuracies)
    avg_time = np.mean(times)

    print(f"\n  Results for d={d}:")
    print(f"    Average accuracy: {avg_accuracy:.4f}")
    print(f"    Average time: {avg_time*1000:.2f} ms")