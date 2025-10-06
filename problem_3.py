import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os

Roster_File = 'MATH425_Fall2025_roster(Sheet1).csv'
Music_Feature_File = 'MATH425_songList(Sheet1).csv'
N_Clusters = 8             
Random_State = 42          


def load_data_from_csv(filename):
    """Loads data from a CSV file, assuming standard comma delimiter."""
    print(f"Loading data from {filename}...")
    try:
        df = pd.read_csv(filename) 
        return df
    except Exception as e:
        print(f"FATAL ERROR: Could not read '{filename}' as a standard CSV.")
        print(f"Details: {e}")
        return None

roster_df = load_data_from_csv(Roster_File)
if roster_df is None:
    exit()

music_df = load_data_from_csv(Music_Feature_File)
if music_df is None:
    exit()

N_Students = 29 

class_roster_names = roster_df.iloc[1 : N_Students + 1, 0].reset_index(drop=True).rename('Roster')
roster_col_index = 0

if len(class_roster_names) != N_Students:
    print(f"FATAL ERROR: Slicing failed unexpectedly. Found {len(class_roster_names)}.")
    exit()

print(f"Roster names extracted from column index {roster_col_index} of the roster file. Found {N_Students} entries.")


features_df_raw = music_df.iloc[1 : N_Students + 1, 1:].copy()

features_for_clustering_df = features_df_raw.select_dtypes(include=np.number)

Music_Features = features_for_clustering_df.astype(float).values

if Music_Features.shape[0] != N_Students:
    print(f"FATAL ERROR: Music file slicing failed to match the {N_Students} student size.")
    print(f"Roster entries: {len(class_roster_names)}, Music samples: {Music_Features.shape[0]}")
    exit()

print(f"Prepared {Music_Features.shape[0]} samples with {Music_Features.shape[1]} features for clustering.")


print(f"Applying k-means clustering (k={N_Clusters})...")

kmeans_music = KMeans(n_clusters=N_Clusters, random_state=Random_State, n_init='auto')
cluster_assignments = kmeans_music.fit_predict(Music_Features)

results_df = pd.DataFrame({
    'Roster': class_roster_names,
    'Cluster': cluster_assignments
})

print("\n--- Class Roster Grouped into 8 Music Clusters ---")

for cluster_id in sorted(results_df['Cluster'].unique()):
    cluster_group = results_df[results_df['Cluster'] == cluster_id]
    member_list = cluster_group['Roster'].tolist()
    
    print(f"\nCluster {cluster_id + 1} (Size: {len(cluster_group)} members):") 
    
    for member in member_list:
        print(f"  - {member}")

print("\nClustering analysis complete.")