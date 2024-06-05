import time
import torch
import numpy as np
from tqdm import tqdm
import gc
import copy
from ..trainer import predict

from sklearn.metrics import DistanceMetric, roc_auc_score

def evaluate(config,
             model,
             reference_dataloader,
             query_dataloader, 
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    
    
    print("\nExtract Features:")
    reference_features, reference_labels, reference_chords = predict(config, model, reference_dataloader) 
    query_features, query_labels, query_chords = predict(config, model, query_dataloader)
    
    print("Compute Scores:")
    r1, median_rank =  calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=step_size, ranks=ranks) 
    mean_dist = calculate_mean_distance(query_features, reference_features, query_labels, reference_labels, reference_chords, step_size=step_size)
    roc_auc = calculate_roc_auc(query_features, reference_features, query_labels, reference_labels, step_size=step_size)

    # cleanup and free memory on GPU
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()
        
    return r1, median_rank, mean_dist, roc_auc


def calc_sim(config,
             model,
             reference_dataloader,
             query_dataloader, 
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    
    
    print("\nExtract Features:")
    reference_features, reference_labels, reference_chords = predict(config, model, reference_dataloader) 
    query_features, query_labels, query_chords = predict(config, model, query_dataloader)
    
    print("Compute Scores Train:")
    r1, median_rank = calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=step_size, ranks=ranks) 
    mean_dist = calculate_mean_distance(query_features, reference_features, query_labels, reference_labels, reference_chords, step_size=step_size)
    roc_auc = calculate_roc_auc(query_features, reference_features, query_labels, reference_labels, step_size=step_size)
    
    near_dict = calculate_nearest(query_features=query_features,
                                  reference_features=reference_features,
                                  query_labels=query_labels,
                                  reference_labels=reference_labels,
                                  neighbour_range=config.neighbour_range,
                                  step_size=step_size)
            
    # cleanup and free memory on GPU
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()
        
    return r1, median_rank, mean_dist, roc_auc, near_dict


def calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=1000, ranks=[1,5,10]):

    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(reference_features)
    
    steps = Q // step_size + 1
    
    query_labels_np = query_labels.cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()
    
    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i
    
    similarity = []
    
    for i in range(steps):
        start = step_size * i
        end = start + step_size
        sim_tmp = query_features[start:end] @ reference_features.T    
        similarity.append(sim_tmp.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)
    ranks_list = []
    topk.append(R//100)
    results = np.zeros([len(topk)])
    bar = tqdm(range(Q))
    
    for i in bar:
        # similiarity value of gt reference
        gt_sim = similarity[i, ref2index[query_labels_np[i]]]
        # number of references with higher similiarity as gt
        higher_sim = similarity[i,:] > gt_sim
        ranking = higher_sim.sum()
        ranks_list.append(ranking)
        
        ranking = higher_sim.sum()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.
                        
        
    results = results/Q * 100.
    bar.close()
    
    # wait to close pbar
    time.sleep(0.1)
    
    string = []
    for i in range(len(topk)-1):
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
        
    string.append('Recall@top1: {:.4f}'.format(results[-1]))   
        
    print(' - '.join(string)) 

    # Calculate Median Rank
    median_rank = np.median(ranks_list)
    print('Median Rank: {:.4f}'.format(median_rank))

    return results[0], median_rank
    

def calculate_nearest(query_features, reference_features, query_labels, reference_labels, neighbour_range=64, step_size=1000):

    Q = len(query_features)
    steps = Q // step_size + 1
    similarity = []
    
    for i in range(steps):
        start = step_size * i
        end = start + step_size
        sim_tmp = query_features[start:end] @ reference_features.T
        similarity.append(sim_tmp.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)
    topk_scores, topk_ids = torch.topk(similarity, k=neighbour_range+1, dim=1)
    topk_references = []
    
    for i in range(len(topk_ids)):
        topk_references.append(reference_labels[topk_ids[i,:]])
    
    topk_references = torch.stack(topk_references, dim=0)
 
    # mask for ids without gt hits
    mask = topk_references != query_labels.unsqueeze(1)
        
    topk_references = topk_references.cpu().numpy()
    mask = mask.cpu().numpy()
    
    # dict that only stores ids where similiarity higher than the lowes gt hit score
    nearest_dict = dict()
    
    for i in range(len(topk_references)):    
        nearest = topk_references[i][mask[i]][:neighbour_range]
        nearest_dict[query_labels[i].item()] = list(nearest)

    return nearest_dict


def calculate_mean_distance(query_features, reference_features, query_labels, reference_labels, coords_radians, step_size=1000) :
    
    # distance matrix 0 on diagonal
    Q = len(query_features)
    R = len(reference_features)
    steps = Q // step_size + 1
    
    # query_labels_np = query_labels.cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()
    
    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i
    
    similarity = []
    
    for i in range(steps):
        start = step_size * i
        end = start + step_size
        sim_tmp = query_features[start:end] @ reference_features.T
        similarity.append(sim_tmp.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)
    dist = DistanceMetric.get_metric('haversine')

    # Ensure coordinates are in radians and have the correct shape // reference_coords
    coords_numpy = np.stack([coord.cpu().numpy() for coord in coords_radians])
    distance_mat = torch.from_numpy(dist.pairwise(coords_numpy))

    # sort along gallery 
    value_sort, index_sort = similarity.sort(descending=True)

    # calculate errors (n-gallery) until hit (hit on diagonal for 1:1 problem)
    gt = torch.arange(len(reference_features)).unsqueeze(1)
    hit = index_sort == gt
    n_wrong = hit.to(torch.long).argmax(-1)

    # sort the distances (can be done in the next for loop as well here only for sanity check to have the sorted matrix for visualization)
    d_sort = []

    for i in range(len(query_features)):
        d_sort.append(distance_mat[i][index_sort[i]])

    d_sort = torch.stack(d_sort)
    mean_dist_until_hit = []

    for i in range(len(query_features)):    
        # adress wrong + hit (for hit distance is 0)
        tmp_distance_until_hit = d_sort[i][:n_wrong[i]+1]
        tmp_distance_until_hit_mean = tmp_distance_until_hit.mean()
        # print(f"Index: {i} - Wrong until hit: {len(tmp_distance_until_hit)-1} - Mean distance: {(6371.0 * tmp_distance_until_hit_mean):.3f} km")
        mean_dist_until_hit.append(tmp_distance_until_hit_mean.item())
        
    mean_dist_until_hit = torch.tensor(mean_dist_until_hit)
    mean_dist = mean_dist_until_hit.mean()

    # Convert radians to kilometers (multiplication with earth radius [km])
    mean_dist_km = 6371.0 * mean_dist
    print(f"Mean Distance: {mean_dist_km:.3f} km")

    return mean_dist_km 


def calculate_roc_auc(query_features, reference_features, query_labels, reference_labels, step_size=1000):
    Q = len(query_features)
    steps = Q // step_size + 1

    query_labels_np = query_labels.cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()
    
    ref2index = {idx: i for i, idx in enumerate(reference_labels_np)}
    
    all_scores = []
    all_labels = []
    
    for i in range(steps):
        start = step_size * i
        end = start + step_size
        sim_tmp = query_features[start:end] @ reference_features.T
        scores = sim_tmp.cpu().numpy()
        
        labels = np.zeros_like(scores)
        for j, label in enumerate(query_labels_np[start:end]):
            labels[j, ref2index[label]] = 1
        
        all_scores.append(scores)
        all_labels.append(labels)
    
    all_scores = np.vstack(all_scores)
    all_labels = np.vstack(all_labels)
    
    # Flatten scores and labels for roc_auc_score
    flat_scores = all_scores.flatten()
    flat_labels = all_labels.flatten()
    
    roc_auc = roc_auc_score(flat_labels, flat_scores)

    print(f"ROC-AUC: {roc_auc:.4f}")

    return roc_auc