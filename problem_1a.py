import numpy as np 
from sklearn.cluster import KMeans

train_data = np.loadtxt('train.txt', delimiter=',')
train_labels = np.loadtxt('train_values.txt')
validate_data = np.loadtxt('validate.txt', delimiter=',')
validate_labels = np.loadtxt('validate_values.txt')

print(f"Training data shape: {train_data.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Validation data shape: {validate_data.shape}")
print(f"Validation labels shape: {validate_labels.shape}")

print("\n" + "-"*60)
print("Applying K-means clustering with k=2 to training data...")
print("-"*60)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(train_data)

train_clusters = kmeans.labels_

print(f"Cluster centers shape: {kmeans.cluster_centers_.shape}")
print(f"Training samples in cluster 0: {np.sum(train_clusters == 0)}")
print(f"Training samples in cluster 1: {np.sum(train_clusters == 1)}")

print("\n" + "-"*60)
print("Determining cluster labels...")
print("-"*60)

cluster_0_labels = train_labels[train_clusters == 0]
cluster_1_labels = train_labels[train_clusters == 1]
cluster_0_mean = np.mean(cluster_0_labels)
cluster_1_mean = np.mean(cluster_1_labels)

print(f"Cluster 0 mean label: {cluster_0_mean:.4f}")
print(f"Clister 1 mean label: {cluster_1_mean:.4f}")

if cluster_0_mean > cluster_1_mean:
    cluster_to_label = {0: 1, 1: -1}
    print("Cluster 0 -> Malignant (+1)")
    print("Cluster 1 -> Benign (-1)")
else:
    cluster_to_label = {0: -1, 1: 1}
    print("Cluster 0 -> Benign (-1)")
    print("Cluster 1 -> Malignant (+1)")

print("\n" + "-"*60)
print("Evaluating on validation data...")
print("-"*60)

validate_clusters = kmeans.predict(validate_data)
predictions = np.array([cluster_to_label[c] for c in validate_clusters])

correct_predictions = np.sum(predictions == validate_labels)
total_predictions = len(validate_labels)
accuracy = correct_predictions / total_predictions

print(f"Correct predictions: {correct_predictions}/{total_predictions}")
print(f"Accuracy: {accuracy:.4f}({accuracy*100:.2f}%)")