import torch
import pickle

import numpy as np
import pandas as pd

from sklearn.metrics import DistanceMetric

# Script for creating a pickle file with the TOP_K middle-distance samples for each sample (train)

# Constants
TOP_K = 256

# Reading the CSV file 'train_df.csv' into a DataFrame.
df_train = pd.read_csv('data/train_df.csv')
train_ids = df_train.index

print("Length Train Ids:", len(df_train))

# Prepare GPS coordinates
gps_coords_list = []

for row in df_train.itertuples():
    coordinates = (np.radians(row.latitude), np.radians(row.longitude))
    gps_coords_list.append(coordinates)
    
print("Length of gps coords : " + str(len(gps_coords_list)))
print("Calculation...")

# Calculate distances
dist = DistanceMetric.get_metric('haversine')
dm = dist.pairwise(gps_coords_list, gps_coords_list)
print("Distance Matrix:", dm.shape)

# Convert to Torch tensor
max_distance = dm.max()
dm_torch = torch.from_numpy(dm)
dm_torch.fill_diagonal_(max_distance)

# Sort distances to find middle neighbors
sorted_distances, sorted_indices = torch.sort(dm_torch, dim=1)

# Select middle neighbors (based on TOP_K/2 from both sides of the middle point)
middle_start = len(sorted_distances[0]) // 2 - TOP_K // 2
middle_end = middle_start + TOP_K

middle_values = sorted_distances[:, middle_start:middle_end]
middle_ids = sorted_indices[:, middle_start:middle_end]

# Convert to numpy for saving or further processing
values_middle_numpy = middle_values.numpy()
ids_middle_numpy = middle_ids.numpy()

middle_neighbors = dict()
for train_id in train_ids:
    middle_neighbors[train_id] = ids_middle_numpy[train_id].tolist()

print("Saving...") 
with open("./data/gps_dict_256_medium.pkl", "wb") as f:
    pickle.dump(middle_neighbors, f)




