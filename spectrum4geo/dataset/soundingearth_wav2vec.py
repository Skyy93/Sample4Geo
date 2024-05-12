import cv2
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import random
import copy
import torch
from tqdm import tqdm
import time

from pathlib import Path 
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# -> shuffle erstmal ignorieren
class Wav2VecSoundingEarthDatasetTrain(Dataset):

    def __init__(self,
                 data_folder='data',
                 split_csv = 'train_df.csv',
                 transforms_sat_image=None,
                 prob_flip=0.0,
                 prob_rotate=0.0,
                 shuffle_batch_size=128,
                 ):
        
        super().__init__()

        self.data_folder = Path(data_folder)
        self.meta = pd.read_csv(str(self.data_folder / split_csv))

        # Satellite Image Transformations
        self.transforms_sat_image = transforms_sat_image  

        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size
                                   
    def __getitem__(self, index):
        sample = self.meta.iloc[index]
        key = sample['short_key']

        img = cv2.imread(str(self.data_folder / 'images' / f'{key}.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Flip satellite
        if np.random.random() < self.prob_flip:
            img = cv2.flip(img, 1)
                       
        # satellite image transforms
        if self.transforms_sat_image is not None:
            img = self.transforms_sat_image(image=img)['image']

        # rotate sat img 90 or 180 or 270
        if np.random.random() < self.prob_rotate:
            r = np.random.choice([1,2,3])
            img = torch.rot90(img, k=r, dims=(1, 2)) 


        audio = torch.load(str(self.data_folder / 'raw_audio_tensorized' / f'{key}.pt'))

        lon = np.radians(sample.longitude)
        lat = np.radians(sample.latitude)
        coords = torch.from_numpy(np.stack([lat, lon])).float()
        label = torch.tensor(int(key), dtype=torch.long)  

        return img, audio, label #, coords
    
    def __len__(self):
        return len(self.meta)
        

    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):
        '''
        Custom shuffle function for unique class_id sampling in batch
        '''
        print("\nShuffle Dataset:")

        # Prepare a list of indices based on the metadata dataframe
        idx_pool = list(range(len(self.meta)))
        
        neighbour_split = neighbour_select // 2
        
        if sim_dict is not None:
            similarity_pool = copy.deepcopy(sim_dict)
            
        # Shuffle the order of indices
        random.shuffle(idx_pool)
       
        # Lookup if already used in epoch
        idx_epoch = set()
        idx_batch = set()
 
        # buckets
        batches = []
        current_batch = []
        
        # counter
        break_counter = 0
        
        # progressbar
        pbar = tqdm(total=len(idx_pool))
    
        while idx_pool:
            pbar.update(1)
            
            idx = idx_pool.pop(0)
            if idx not in idx_batch and idx not in idx_epoch:
                idx_batch.add(idx)
                current_batch.append(idx)
                idx_epoch.add(idx)
                break_counter = 0
              
                if sim_dict and len(current_batch) < self.shuffle_batch_size:
                    # Access similar and dissimilar indices based on similarity dictionary
                    near_similarity = similarity_pool[idx][:neighbour_range]
                    
                    near_neighbours = copy.deepcopy(near_similarity[:neighbour_split])
                    far_neighbours = copy.deepcopy(near_similarity[neighbour_split:])
                    
                    random.shuffle(far_neighbours)
                    far_neighbours = far_neighbours[:neighbour_split]
                    
                    near_similarity_select = near_neighbours + far_neighbours
                    
                    for idx_near in near_similarity_select:
                        if len(current_batch) >= self.shuffle_batch_size:
                            break
                        if idx_near not in idx_batch and idx_near not in idx_epoch:
                            idx_batch.add(idx_near)
                            current_batch.append(idx_near)
                            idx_epoch.add(idx_near)
                            similarity_pool[idx].remove(idx_near)
                            break_counter = 0
            else:
                # if idx does not fit in batch and is not already used in epoch -> back to pool
                if idx not in idx_batch and idx not in idx_epoch:
                    idx_pool.append(idx)
                    
                break_counter += 1
                
            if break_counter >= 1024:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                # empty current batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()
        
        # wait before closing progress bar
        time.sleep(0.3)
        
        self.samples = batches
        print("idx_pool:", len(idx_pool))
        print("Original Length: {} - Length after Shuffle: {}".format(len(self.meta), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.meta) - len(self.samples))
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0], self.samples[-1]))


class Wav2VecSoundingEarthDatasetEval(Dataset):
    
    def __init__(self,
                 data_folder='data',
                 split_csv = 'valid_df.csv',
                 query_type = "sat",
                 transforms = None,
                 ):
        
        super().__init__()
 
        self.data_folder = Path(data_folder)
        self.meta = pd.read_csv(str(self.data_folder / split_csv))
        self.query_type = query_type

        self.transforms = transforms     
           
        if not ( self.query_type == "sat" or self.query_type == "audio" ):
            raise ValueError("Invalid 'query_type' parameter. 'query_type' must be 'sat' or 'audio'")
                
    def __getitem__(self, index):

        sample = self.meta.iloc[index]
        key = sample['short_key']
        
        if self.query_type == "sat":
            item = cv2.imread(str(self.data_folder / 'images' / f'{key}.jpg'))
            item = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)
            
            # image transforms
            if self.transforms is not None:
                item = self.transforms(image=item)['image']
                        
        else: # if self.query_type == "audio"

            item = torch.load(str(self.data_folder / 'raw_audio_tensorized' / f'{key}.pt'))

        lon = np.radians(sample.longitude)
        lat = np.radians(sample.latitude)
        coords = torch.from_numpy(np.stack([lat, lon])).float()
        label = torch.tensor(int(key), dtype=torch.long)  

        return item, label #, coords
    
    def __len__(self):
        return len(self.meta)
