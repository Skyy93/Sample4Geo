import cv2
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import random
import copy
import torch
from tqdm import tqdm
import time
 
class SoundingEarthDatasetEval(Dataset):
    
    def __init__(self,
                 data_folder='data/SoundingEarth/data',
                 transforms_image=None,
                 transforms_spectrogram=None,
                 patch_time_steps=120,
                 sr_kHz=48,
                 n_mels=128,
                 prob_flip=0.0,
                 prob_rotate=0.0,
                 shuffle_batch_size=128,
                 ):
        
        super().__init__()

        self.data_folder = data_folder
        self.meta = pd.read_csv(self.data_folder / 'metadata.csv')

        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size
        
        self.transforms_image = transforms_image               # satellite
        self.transforms_spectrogram = transforms_spectrogram   # audio

        self.sr_kHz = sr_kHz
        self.hop_length = 512
        # hop length: librosa(default) = 512
        # if default: hop_length = win_length//4
        #                          win_length = n_fft
        #                                       n_fft = 2048
        self.patch_time_steps = patch_time_steps # Count of datapoints (X-Direction)
        self.time_step == self.hop_length / sr_kHz * 1e3 #  timestep between two datapoints (X-Direction)
        self.n_mels = n_mels  # Specifies the number of Mel bands (frequency bands) to use in the Mel spectrogram
                                                
    def __getitem__(self, index):
        sample = self.meta.iloc[index]
        key = sample['short_key']

        img = np.array(Image.open(self.data_folder / 'images' / f'{key}.jpg'))
        img = torch.from_numpy(img).permute(2, 0, 1)

        audio = np.load(self.root / 'spectrograms' / f'{key}.npy')

        # width of the patch = Count of patch_time_steps
        patch_width = self.patch_time_steps

        # Check if the spectrogram is wide enough, and add padding if necessary
        if audio.shape[1] < patch_width:
            padding_width = patch_width - audio.shape[1]
            # Padding on the right side (along X-axis) to match the required width
            audio = np.pad(audio, ((0, 0), (0, padding_width)), mode='constant', constant_values=0)

        # Cut a random patch if the spectrogram is wide enough
        if audio.shape[1] > patch_width:
            start = torch.randint(0, audio.shape[1] - patch_width + 1, (1,)).item()
            # Selecting a random start point and cutting out the patch
            audio = audio[:, start:start + patch_width]

        # Add a new axis
        audio = audio[np.newaxis]
                       
        # image transforms
        if self.transforms_image is not None:
            img = self.transforms_image(image=img)['image']
            
        if self.transforms_spectrogram is not None:
            audio = self.transforms_spectrogram(image=audio)['image']
                
        lon = np.radians(sample.longitude)
        lat = np.radians(sample.latitude)
        coords = torch.from_numpy(np.stack([lat, lon])).float()

        return key, img, audio, coords
    
    def __len__(self):
        return len(self.meta)
        


class SoundingEarthDatasetTrain(SoundingEarthDatasetEval):

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