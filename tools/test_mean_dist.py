import torch
from sklearn.metrics import DistanceMetric


# 128 fake features with 512 as dimension

# query
sound = torch.rand((128, 512))

# gallery
image = torch.rand((128, 512))


# distances needs to be calculated only once for all examples

# fake coordinates
gps_coords_list= (torch.rand((128, 2)) * 100).tolist()

# distance matrix 0 on diagonal
print("Length of gps coords : " + str(len(gps_coords_list)))

dist = DistanceMetric.get_metric('haversine')
distance_matrix = dist.pairwise(gps_coords_list, gps_coords_list)
print(distance_matrix.shape)

distance_matrix = torch.from_numpy(distance_matrix)




# similarity matrix from features
similarity_matrix = sound @ image.T

# sort along gallery 
value_sort, index_sort = similarity_matrix.sort(descending=True)



# calculate errors (n-gallery) until hit (hit on diagonal for 1:1 problem)
gt = torch.arange(len(image)).unsqueeze(1)
hit = index_sort == gt
n_wrong = hit.to(torch.long).argmax(-1)




# sort the distances (can be done in the next for loop as well here only for sanity check to have the sorted matrix for visualization)
d_sort = []

for i in range(len(sound)):
    
    d_sort.append(distance_matrix[i][index_sort[i]])

d_sort = torch.stack(d_sort)



mean_dist_until_hit = []

for i in range(len(sound)):
    
    # adress wrong + hit (for hit distance is 0)
        
    tmp_distance_until_hit = d_sort[i][:n_wrong[i]+1]
    
    tmp_distance_until_hit_mean = tmp_distance_until_hit.mean()
    
    print(f"Index: {i} - Wrong until hit: {len(tmp_distance_until_hit)-1} - Mean distance: {tmp_distance_until_hit_mean:.3f}")
    
    mean_dist_until_hit.append(tmp_distance_until_hit_mean.item())
    
 
mean_dist_until_hit = torch.tensor(mean_dist_until_hit)

print(f"Mean Distance: {mean_dist_until_hit.mean():.3f}")



