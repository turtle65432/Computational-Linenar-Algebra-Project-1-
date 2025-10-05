import numpy as np 
from sklearn.cluster import KMeans

train_data = np.loadtxt('train.txt')
train_labels = np.loadtxt('train_values.txt')
validate_data = np.loadtxt('validate.txt')
validate_labels = np.loadtxt('validate_values.txt')

print(f"Training data shape: {train_data.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Validation data shape: {validate_data.shape}")
print(f"Validation labels shape: {validate_labels.shape}")

print("\n" + "="*60)
print("Applying K-means clustering with k=2 to training data...")
print("="*60)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(train_data)