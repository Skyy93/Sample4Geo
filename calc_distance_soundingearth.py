import torch
import pickle

import numpy as np
import pandas as pd

from sklearn.metrics import DistanceMetric

# Script for creating an pkl with the TOP_K nearest samples for each sample (train)

# Constants
TOP_K = 128

# Reading the CSV file 'train_df.csv' into a DataFrame.
df_train = pd.read_csv('data/train_df.csv')
train_ids = df_train.index
train_id_to_short_key = df_train['short_key'].to_dict()

print("Length Train Ids:", len(df_train))

# Reading the CSV file 'final_metadata.csv' into a DataFrame.
df_gps = pd.read_csv('data/final_metadata.csv', index_col='short_key')

# Prepare GPS coordinates
gps_coords = {}
gps_coords_list = []

for train_id in train_ids:
    row = df_gps.loc[train_id_to_short_key[train_id]]
    coordinates = (np.radians(row["latitude"]), np.radians(row["longitude"]))
    gps_coords[train_id] = coordinates
    gps_coords_list.append(coordinates)
    
print("Length of gps coords : " + str(len(gps_coords_list)))
print("Calculation...")

# Calculate distances
dist = DistanceMetric.get_metric('haversine')
dm = dist.pairwise(gps_coords_list, gps_coords_list)
print("Distance Matrix:", dm.shape)

# Convert to Torch tensor
dm_torch = torch.from_numpy(dm)
dm_torch = dm_torch.fill_diagonal_(dm.max())

# Get nearest neighbors
values, ids = torch.topk(dm_torch, k=TOP_K, dim=1, largest=False)

# Convert to numpy for saving or further processing
values_near_numpy = values.numpy()
ids_near_numpy = ids.numpy()

near_neighbors = dict()
for train_id in train_ids:
    near_neighbors[train_id] = train_ids[ids_near_numpy[train_id]].tolist()

print("Saving...") 
with open("./data/gps_dict.pkl", "wb") as f:
    pickle.dump(near_neighbors, f)
