import torch
import gc

import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path 
from sklearn.metrics import DistanceMetric

from ..trainer import predict


def evaluate(config,
             model,
             reference_dataloader,
             query_dataloader, 
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    
    split_path = Path(config.data_folder) / config.evaluate_csv
    meta_df = pd.read_csv(split_path, index_col='short_key')

    print("\nExtract Features:")
    reference_features, reference_labels = predict(config, model, reference_dataloader) 
    query_features, query_labels = predict(config, model, query_dataloader)
    
    print("Compute Scores:\n")
    label_ids_until_hit = calculate_label_ids_until_hit(query_features, reference_features, reference_labels, step_size=step_size)
    scores = calculate_scores(label_ids_until_hit, meta_df, recall_ranks=ranks, topk_recall=True) 
    r1 = scores[0][1]

    calculate_scores_continentwise(label_ids_until_hit, meta_df, recall_ranks=ranks, topk_recall=True) 
    region_wise_recalls = calculate_region_wise_recalls(label_ids_until_hit, meta_df, calc_ranks=ranks, print_ranks=ranks)
    calculate_balanced_continental_recalls(region_wise_recalls)

    # cleanup and free memory on GPU
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()
        
    return r1


def calc_sim(config,
             model,
             reference_dataloader,
             query_dataloader, 
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    
    split_path = Path(config.data_folder) / config.evaluate_csv
    meta_df = pd.read_csv(split_path, index_col='short_key')
    
    print("\nExtract Features:")
    reference_features, reference_labels = predict(config, model, reference_dataloader) 
    query_features, query_labels = predict(config, model, query_dataloader)
    
    print("Compute Scores Train:")
    label_ids_until_hit = calculate_label_ids_until_hit(query_features, reference_features, reference_labels, step_size=step_size)
    scores = calculate_scores(label_ids_until_hit, meta_df, recall_ranks=ranks, topk_recall=True) 
    r1 = scores[0][1]

    region_wise_recalls = calculate_region_wise_recalls(label_ids_until_hit, meta_df, calc_ranks=ranks, print_ranks=[])
    calculate_balanced_continental_recalls(region_wise_recalls)
    
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
        
    return r1, near_dict


def calculate_label_ids_until_hit(query_features, reference_features, labels, step_size=1000):
    '''returns an dict with query item label IDs as keys. Each key maps to a list of label IDs, 
       sorted by descending probability, up until (but not including) the hit (query item label ID).'''
    Q = len(query_features)    
    steps = Q // step_size + 1
    labels_np = labels.cpu().numpy()
    ref2index = {idx: i for i, idx in enumerate(labels_np)}
    similarity = []
    
    for i in range(steps):
        start = step_size * i
        end = start + step_size
        sim_tmp = query_features[start:end] @ reference_features.T    
        similarity.append(sim_tmp.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)
    label_ids_until_hit = {}
    bar = tqdm(range(Q), desc='Generate lists of label_ids until Hit')

    for i in bar:
        # similiarity value of gt reference
        gt_sim = similarity[i, ref2index[labels_np[i]]]
        # number of references with higher similiarity as gt
        higher_sim = (similarity[i, :] > gt_sim).numpy()
        # creating list of label_ids until hit
        hit_indices = np.where(higher_sim)[0]
        # sorting in descending order
        sorted_hit_indices = hit_indices[np.argsort(-similarity[i, hit_indices].numpy())]
        # label_ids_until_hit[label_id] = [], [label_id1], [label_id1, label_id2, ...]
        label_ids_until_hit[labels_np[i]] = labels_np[sorted_hit_indices].tolist()

    print()

    return label_ids_until_hit


def calculate_scores(label_ids_until_hit, metadata_df, recall_ranks, topk_recall=True, verbose=False):
    '''returns an tuple containing an dict of recall results with the ranks as keys, the median rank, the mean error distance
       and top_str, which is the string used as key for the Recall@~1%.'''
    count_until_hit = [len(value) for value in label_ids_until_hit.values()]
    id_count = len(label_ids_until_hit)
    topk = id_count//100
    topk_str = f'{topk}/{topk/id_count*100:0.2f}%'

    #### Set up headers for display
    if not verbose:
        header_format = ' | '.join(['{:<13}' for _ in recall_ranks]) 
        headers = [f'Recall@{rank}' for rank in recall_ranks]
        if topk_recall:
            header_format += ' | {:<16}'
            headers += [f'Recall@{topk_str}']
        header_format += ' | {:<13}' + ' | {:<24}'
        headers += ['Median Rank'] + ['Mean Error Distance [km]']
        header_formated = (header_format).format(*headers)

        print('Calculate Recalls, Median Rank and Mean Error Distance!')
        print(header_formated)
        print('-' * len(header_formated))

    #### Calculating Recalls
    recall_results = {rank: np.mean([int(count < rank) for count in count_until_hit]) * 100 for rank in recall_ranks}
    if topk_recall:
        recall_results[topk_str] = np.mean([int(count < topk) for count in count_until_hit]) * 100

    #### Calculating Median Rank
    median_rank = np.median(count_until_hit)

    #### Calculating Mean Error Distance
    coordinates = metadata_df.loc[:, ['latitude', 'longitude']]
    error_distances = []
    dist = DistanceMetric.get_metric('haversine')
    
    for true_label_id, wrong_label_ids in label_ids_until_hit.items():
        true_coords = coordinates.loc[true_label_id].to_numpy()
        if len(wrong_label_ids) > 0:
            wrong_coords = coordinates.loc[wrong_label_ids].to_numpy()
            # Calculate Haversine distances
            distances = dist.pairwise(np.radians([true_coords]), np.radians(wrong_coords)).flatten()
            error_distances.append(np.mean(distances))
        else:
            error_distances.append(0)
    
    mean_distance_error = np.mean(error_distances) * 6371  # Convert to kilometer

    #### Output the calculated metrics
    if not verbose:
        result_format = ' | '.join(['{:<13.4f}' for _ in recall_ranks]) 
        result_values = [recall_results[rank] for rank in recall_ranks] 
        if topk_recall:
            result_format += ' | {:<16.4f}'
            result_values += [recall_results[topk_str]]
        result_format += ' | {:<13.0f}' + ' | {:<24.4f}'
        result_values += [median_rank, mean_distance_error]
        print(result_format.format(*result_values))
        print()
    
    return recall_results, median_rank, mean_distance_error, topk_str


def calculate_scores_continentwise(label_ids_until_hit, metadata_df, recall_ranks=[1,5,10,50,100], topk_recall=True):
    '''returns an dict with continent as keys and tuples as values. 
       These tuples contain an dict of recall results with the ranks as keys, the median rank, the mean error distance
       and top_str, which is the string used as key for the Recall@~1%.'''
    continents = sorted(set(metadata_df['continent']))
    header_format = '{:<13} | ' + '{:<13} | ' + ' | '.join(['{:<13}' for _ in recall_ranks]) 
    headers = ['Continent', 'used Samples'] + [f'Recall@{rank}' for rank in recall_ranks]
    if topk_recall:
        header_format += ' | {:<13}' + ' | {:<8}' 
        headers += [f'Recall@~1% ->'] + [f'~1%']
    header_format += ' | {:<13}' + ' | {:<24}'
    headers += ['Median Rank'] + ['Mean Error Distance [km]']
    header_formated = header_format.format(*headers)

    print('Calculate Recalls, Median Rank and Mean Error Distance within Continents: ' + ', '.join(continents) + '!')
    print(header_formated)
    print('-' * len(header_formated))

    continent_scores = {}
    for continent in continents:
        allowed_label_ids = set(metadata_df[metadata_df['continent'] == continent].index)
        continent_label_ids_until_hit = {
            key: value for key, value in label_ids_until_hit.items() if key in allowed_label_ids
        }

        for key in continent_label_ids_until_hit:
            continent_label_ids_until_hit[key] = [
                label_id for label_id in continent_label_ids_until_hit[key] if label_id in allowed_label_ids
            ]
        
        # tuple containing an dict of recall results with the ranks as keys, the median rank, the mean error distanceand top_str
        continent_scores[continent] = calculate_scores(continent_label_ids_until_hit, metadata_df, recall_ranks, topk_recall, verbose=True)
        recall_results, median_rank, mean_distance_error, topk_str = continent_scores[continent]

        result_format = '{:<13} | ' + '{:<13.0f} | ' + ' | '.join(['{:<13.4f}' for _ in recall_ranks]) 
        result_values = [continent, len(allowed_label_ids)] + [recall_results[rank] for rank in recall_ranks] 
        if topk_recall:
            result_format += ' | {:<13.4f}' + ' | {:<8}' 
            result_values += [recall_results[topk_str], topk_str]
        result_format += ' | {:<13.0f}' + ' | {:<24.4f}'
        result_values += [median_rank, mean_distance_error]

        print(result_format.format(*result_values))

    print()

    return continent_scores


def calculate_region_wise_recalls(label_ids_until_hit, metadata_df, calc_ranks=[1,5,10], print_ranks=[1,5,10]):
    '''returns an dict where each key is a rank (from calc_ranks) and the value is another dict.
       The nested dict has continents as keys and region-wise recall scores as values.'''
    continents = sorted(set(metadata_df['continent']))

    print('Calculate RegionWiseRecalls!')
    if print_ranks and len(print_ranks)>1:
        header_format = '{:<15} | {:<14} | ' + ' | '.join(['{:<19}' for _ in print_ranks])
        headers = ['Continent', 'valid Samples'] + [f'RegionWiseRecall@{rank}' for rank in print_ranks]
        header_formated = (header_format).format(*headers)
        print(header_formated)
        print('-' * (len(header_formated)))

    region_wise_recalls = {}
    for continent in continents:
        allowed_label_ids = set(metadata_df[metadata_df['continent'] == continent].index)

        label_ids_until_continent_hit = {}
        for key, wrong_label_ids in label_ids_until_hit.items():
            if key in allowed_label_ids:
                continental_wrong_label_ids = next((wrong_label_ids[:i] for i, id in enumerate(wrong_label_ids) if id in allowed_label_ids), wrong_label_ids)
                label_ids_until_continent_hit[key] = continental_wrong_label_ids

        count_until_continent_hit = [len(value) for value in label_ids_until_continent_hit.values()]

        recall_results = {}
        for rank in calc_ranks:
            region_wise_recall = np.mean([int(count < rank) for count in count_until_continent_hit])*100
            recall_results[rank] = region_wise_recall

        region_wise_recalls[continent] = recall_results

        if print_ranks and len(print_ranks)>1:
            result_format = '{:<15} | {:<14} | ' + ' | '.join(['{:<19.4f}' for _ in print_ranks])
            result_values = [continent, len(allowed_label_ids)] + [region_wise_recalls[continent].get(rank, 0.0) for rank in print_ranks]
            print(result_format.format(*result_values))
    print()
    
    return region_wise_recalls


def calculate_balanced_continental_recalls(region_wise_recalls):
    '''returns an dict with continents as keys and balanced recall scores as values.'''
    print('Calculate BalancedContinentalRecalls!')

    # inverts region_wise_recalls dictionary from {continent: {rank: value}} to {rank: {continent: value}}
    ranks = sorted(set(rank for subdict in region_wise_recalls.values() for rank in subdict.keys()))
    region_wise_recalls = {rank: {continent: region_wise_recalls[continent].get(rank, None) for continent in region_wise_recalls} for rank in ranks}

    header_format = ' | '.join(['{:<29}' for _ in ranks])
    headers = [f'BalancedContinentalRecall@{rank}' for rank in ranks]
    header_formated = (header_format).format(*headers)
    print(header_formated)
    print('-' * (len(header_formated)))

    balanced_continental_recalls = {}
    for rank in ranks:
        recall_values = [value for value in region_wise_recalls[rank].values()]
        balanced_continental_recalls[rank] = np.mean(recall_values)

    result_format = ' | '.join(['{:<29.4f}' for _ in ranks])
    result_values = [balanced_continental_recalls[rank] for rank in ranks]
    print(result_format.format(*result_values))
    print()

    return balanced_continental_recalls


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