import pandas as pd
from sklearn.metrics import DistanceMetric
import torch
import pickle

# Script for creating an pkl with the TOP_K nearest samples for each sample (train)

# Constants
TOP_K = 128

# Load data
df_train = pd.read_csv('data/SoundingEarth/data/train_df.csv')
train_ids = df_train["short_key"].tolist()

print("Length Train Ids:", len(train_ids))

df_gps = pd.read_csv('data/SoundingEarth/data/final_metadata_with_captions.csv')
df_gps.set_index('short_key', inplace=True)

# Prepare GPS coordinates
gps_coords = {}
gps_coords_list = []

for idx in train_ids :
    row = df_gps.loc[idx]
    coordinates = (float(row["latitude"]), float(row["longitude"]))
    gps_coords[idx] = coordinates
    gps_coords_list.append(coordinates)
    
print("Length of gps coords : " +str(len(gps_coords_list)))
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
for i, idx in enumerate(train_ids):
    # Extract indices from the numpy array for the i-th element
    indices_for_i = ids_near_numpy[i]
    # Use a list comprehension to gather the corresponding train_ids
    near_neighbors[idx] = [train_ids[j] for j in indices_for_i]

print("Saving...") 
with open("./data/SoundingEarth/gps_dict.pkl", "wb") as f:
    pickle.dump(near_neighbors, f)
