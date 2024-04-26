import cv2
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import random
import copy
import torch
from tqdm import tqdm
import time

# My Changes:
import librosa

 
class GeoSoundDatasetTrain(Dataset):
    
    def __init__(self,
                 data_folder,
                 transforms_query=None,
                 transforms_reference=None,
                 audio_duration=120,
                 n_mels=128,
                 prob_flip=0.0,
                 prob_rotate=0.0,
                 shuffle_batch_size=128,
                 ):
        
        super().__init__()

        self.data_folder = data_folder
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size
        
        self.transforms_query = transforms_query           # ground
        self.transforms_reference = transforms_reference   # satellite

        self.audio_duration = audio_duration  # sample time limit (cut)
        self.n_mels = n_mels  # Specifies the number of Mel bands (frequency bands) to use in the Mel spectrogram
        
        self.df = pd.read_csv(f'{data_folder}/splits/train-19zl.csv', header=None)
        self.df = self.df.rename(columns={0: "sat", 1: "sound", 2: "sound_anno"})
        self.df["idx"] = self.df.sat.map(lambda x : int(x.split("/")[-1].split(".")[0]))
        
        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2sound = dict(zip(self.df.idx, self.df.sound))
   
        self.pairs = list(zip(self.df.idx, self.df.sat, self.df.sound))
        
        self.idx2pair = dict()
        train_ids_list = list()
        
        # for shuffle pool
        for pair in self.pairs:
            idx = pair[0]
            self.idx2pair[idx] = pair
            train_ids_list.append(idx)
            
        self.train_ids = train_ids_list
        self.samples = copy.deepcopy(self.train_ids)
            

    def __getitem__(self, index):
        
        idx, sat, sound = self.idx2pair[self.samples[index]]
        


        # load query -> mel_specto: sound_file
        audio_path = f'{self.data_folder}/{self.idx2audio[index]}'
        y, sr = librosa.load(audio_path, duration=self.audio_duration)
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=self.n_mels)
        S_DB = librosa.power_to_db(S, ref=np.max)

        # Normalizing the spectrogram and converting to an image
        S_norm = cv2.normalize(S_DB, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        query_img = cv2.applyColorMap(np.uint8(S_norm), cv2.COLORMAP_VIRIDIS)

        # load reference -> satellite image
        reference_img = cv2.imread(f'{self.data_folder}/{sat}')
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
            
        # NO FLIP # # Flip simultaneously query and reference
        # NO FLIP # if np.random.random() < self.prob_flip:
        # NO FLIP #     query_img = cv2.flip(query_img, 1)
        # NO FLIP #     reference_img = cv2.flip(reference_img, 1) 
        
        # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']
            
        if self.transforms_reference is not None:
            reference_img = self.transforms_reference(image=reference_img)['image']
                
        # NO ROTATE # # Rotate simultaneously query and reference
        # NO ROTATE # if np.random.random() < self.prob_rotate:
        # NO ROTATE # 
        # NO ROTATE #     r = np.random.choice([1,2,3])
        # NO ROTATE #     
        # NO ROTATE #     # rotate sat img 90 or 180 or 270
        # NO ROTATE #     reference_img = torch.rot90(reference_img, k=r, dims=(1, 2)) 
        # NO ROTATE #     
        # NO ROTATE #     # use roll for ground view if rotate sat view
        # NO ROTATE #     c, h, w = query_img.shape
        # NO ROTATE #     shifts = - w//4 * r
        # NO ROTATE #     query_img = torch.roll(query_img, shifts=shifts, dims=2)  
                   
            
        label = torch.tensor(idx, dtype=torch.long)  
        
        return query_img, reference_img, label
    
    def __len__(self):
        return len(self.samples)
        
        
            
    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):

            '''
            custom shuffle function for unique class_id sampling in batch
            '''
            
            print("\nShuffle Dataset:")
            
            idx_pool = copy.deepcopy(self.train_ids)
        
            neighbour_split = neighbour_select // 2
            
            if sim_dict is not None:
                similarity_pool = copy.deepcopy(sim_dict)
                
            # Shuffle pairs order
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
            pbar = tqdm()
    
            while True:
                
                pbar.update()
                
                if len(idx_pool) > 0:
                    idx = idx_pool.pop(0)

                    
                    if idx not in idx_batch and idx not in idx_epoch and len(current_batch) < self.shuffle_batch_size:
                    
                        idx_batch.add(idx)
                        current_batch.append(idx)
                        idx_epoch.add(idx)
                        break_counter = 0
                      
                        if sim_dict is not None and len(current_batch) < self.shuffle_batch_size:
                            
                            near_similarity = similarity_pool[idx][:neighbour_range]
                            
                            near_neighbours = copy.deepcopy(near_similarity[:neighbour_split])
                            
                            far_neighbours = copy.deepcopy(near_similarity[neighbour_split:])
                            
                            random.shuffle(far_neighbours)
                            
                            far_neighbours = far_neighbours[:neighbour_split]
                            
                            near_similarity_select = near_neighbours + far_neighbours
                            
                            for idx_near in near_similarity_select:
                           
                                # check for space in batch
                                if len(current_batch) >= self.shuffle_batch_size:
                                    break
                                
                                # check if idx not already in batch or epoch
                                if idx_near not in idx_batch and idx_near not in idx_epoch and idx_near:
                            
                                    idx_batch.add(idx_near)
                                    current_batch.append(idx_near)
                                    idx_epoch.add(idx_near)
                                    similarity_pool[idx].remove(idx_near)
                                    break_counter = 0
                                    
                    else:
                        # if idx fits not in batch and is not already used in epoch -> back to pool
                        if idx not in idx_batch and idx not in idx_epoch:
                            idx_pool.append(idx)
                            
                        break_counter += 1
                        
                    if break_counter >= 1024:
                        break
                   
                else:
                    break

                if len(current_batch) >= self.shuffle_batch_size:
                    # empty current_batch bucket to batches
                    batches.extend(current_batch)
                    idx_batch = set()
                    current_batch = []

            pbar.close()
            
            # wait before closing progress bar
            time.sleep(0.3)
            
            self.samples = batches
            print("idx_pool:", len(idx_pool))
            print("Original Length: {} - Length after Shuffle: {}".format(len(self.train_ids), len(self.samples))) 
            print("Break Counter:", break_counter)
            print("Pairs left out of last batch to avoid creating noise:", len(self.train_ids) - len(self.samples))
            print("First Element ID: {} - Last Element ID: {}".format(self.samples[0], self.samples[-1]))  


class GeoSoundDatasetEval(Dataset):

    def __init__(self, data_folder, split, img_type, transforms=None, audio_duration=120, n_mels=128):
        super().__init__()
        self.data_folder = data_folder
        self.split = split
        self.img_type = img_type
        self.transforms = transforms
        self.audio_duration = audio_duration  # sample time limit (cut)
        self.n_mels = n_mels  # Specifies the number of Mel bands (frequency bands) to use in the Mel spectrogram





        # TODO: csv files!! create!
        if split == 'train':
            self.df = pd.read_csv(f'{data_folder}/splits/train-19zl.csv', header=None)
        else:
            self.df = pd.read_csv(f'{data_folder}/splits/val-19zl.csv', header=None)
        




        self.df = self.df.rename(columns={0: "sat", 1: "audio", 2: "audio_anno"})
        
        self.df["idx"] = self.df.sat.map(lambda x: int(x.split("/")[-1].split(".")[0]))

        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2audio = dict(zip(self.df.idx, self.df.audio))

        if self.img_type == "reference":
            self.images = self.df.sat.values
            self.label = self.df.idx.values
            
        elif self.img_type == "query":
            self.images = self.df.audio.values
            self.label = self.df.idx.values 
        else:
            raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'")
             
    def __getitem__(self, index):
        if self.img_type == "query":
            audio_path = f'{self.data_folder}/{self.idx2audio[index]}'
            y, sr = librosa.load(audio_path, duration=self.audio_duration)
            S = librosa.feature.melspectrogram(y, sr=sr, n_mels=self.n_mels)
            S_DB = librosa.power_to_db(S, ref=np.max)

            # Normalizing the spectrogram and converting to an image
            S_norm = cv2.normalize(S_DB, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            img = cv2.applyColorMap(np.uint8(S_norm), cv2.COLORMAP_VIRIDIS)

        else:
            img_path = f'{self.data_folder}/{self.idx2sat[index]}'
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            img = self.transforms(img)

        label = torch.tensor(self.df.loc[index, 'idx'], dtype=torch.long)
        return img, label

    def __len__(self):
        return len(self.df)       



