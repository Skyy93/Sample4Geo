import numpy as np
import torch
from sklearn.metrics import DistanceMetric
import pickle
from reident.dataset.vigor import VigorDatasetTrain

SAME_AREA = True   # True: Same | False: Cross
TOP_K = 128

dataset = VigorDatasetTrain(data_folder="./data/VIGOR",
                            same_area=SAME_AREA)

df_sat = dataset.df_sat
df_ground = dataset.df_ground

idx2sat = dataset.idx2sat
train_sat_ids = np.sort(df_ground["sat"].unique())

print("Length Train Ids:", len(train_sat_ids))

gps_coords = {}
gps_coords_list = []

for idx in train_sat_ids:
    
    _, lat, long = idx2sat[idx][:-4].split('_')
    
    gps_coords[idx] = (float(lat), float(long))
    gps_coords_list.append((float(lat), float(long)))
    
print("Length of gps coords : " + str(len(gps_coords_list)))

dist = DistanceMetric.get_metric('haversine')
dm = dist.pairwise(gps_coords_list, gps_coords_list)
print(dm.shape)


dm_torch = torch.from_numpy(dm)
dm_torch = dm_torch.fill_diagonal_(dm.max())

values, ids = torch.topk(dm_torch, k=TOP_K, dim=1, largest=False)


values_near = values.numpy()
ids_near = ids.numpy()


near_neighbors = dict()

for i, idx in enumerate(train_sat_ids):
    near_neighbors[idx] = train_sat_ids[ids_near[i]].tolist()


if SAME_AREA:
    with open("./data/VIGOR/gps_dict_same.pkl", "wb") as f:
        pickle.dump(near_neighbors, f)
else:
    with open("./data/VIGOR/gps_dict_cross.pkl", "wb") as f:
        pickle.dump(near_neighbors, f)   



