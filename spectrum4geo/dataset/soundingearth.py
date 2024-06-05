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

# To Convert Spectrogram to "True" 3 Channel Image!
def apply_viridis_colormap(array):

    #array = (array - array.min()) / (array.max() - array.min())
    norm = plt.Normalize(vmin=array.min(), vmax=array.max())
    cmap = cm.get_cmap('viridis')

    rgba_array = cmap(norm(array)).astype(np.float32)
    rgb_array = rgba_array[:, :, :3]

    return rgb_array

# -> shuffle erstmal ignorieren
class SoundingEarthDatasetTrain(Dataset):

    def __init__(self,
                 data_folder='data',
                 split_csv = 'train_df.csv',
                 transforms_sat_image=None,
                 transforms_spectrogram=None,
                 patch_time_steps=120,
                 sr_kHz=48,
                 n_mels=128,
                 prob_flip=0.0,
                 prob_rotate=0.0,
                 shuffle_batch_size=128,
                 ):
        
        super().__init__()

        self.data_folder = Path(data_folder)
        self.meta = pd.read_csv(str(self.data_folder / split_csv))

        # Satellite Image Transformations
        self.transforms_sat_image = transforms_sat_image  

        # Spectrogram Image Transformations     
        self.transforms_spectrogram = transforms_spectrogram   
        
        self.patch_time_steps = patch_time_steps # Count of datapoints (X-Direction)
        self.sr_kHz = sr_kHz
        self.hop_length = 512
        # hop length: librosa(default) = 512
        # if default: hop_length = win_length//4
        #                          win_length = n_fft
        #                                       n_fft = 2048
        self.time_step = self.hop_length / sr_kHz * 1e3 #  timestep between two datapoints (X-Direction)
        self.n_mels = n_mels  # Specifies the number of Mel bands (frequency bands) to use in the Mel spectrogram

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
        else:
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1)

        # rotate sat img 90 or 180 or 270
        if np.random.random() < self.prob_rotate:
            r = np.random.choice([1,2,3])
            img = torch.rot90(img, k=r, dims=(1, 2)) 

        spectrogram = np.load(self.data_folder / 'spectrograms' / f'{self.n_mels}mel_{self.sr_kHz}kHz' / f'{key}.npy')

        # Check if the spectrogram is wide enough, and add padding if necessary
        if spectrogram.shape[1] < self.patch_time_steps:
            padding_width = self.patch_time_steps - spectrogram.shape[1]
            left_padding = np.random.randint(0, padding_width + 1)
            right_padding = padding_width - left_padding
            spectrogram = np.pad(spectrogram, ((0, 0), (left_padding, right_padding)), mode='constant', constant_values=-60) # Assumption: -60 dB is considered silence

        # Cut a random patch if the spectrogram is wide enough
        elif spectrogram.shape[1] > self.patch_time_steps:
            start = torch.randint(0, spectrogram.shape[1] - self.patch_time_steps + 1, (1,)).item()
            spectrogram = spectrogram[:, start:start + self.patch_time_steps]
    
        # spectrogram = librosa.power_to_db(spectrogram, ref=np.max) # only if data isnt in dB, important: change padded values above!
        
        spectrogram = apply_viridis_colormap(spectrogram)#_back)

        if self.transforms_spectrogram is not None:
            spectrogram = self.transforms_spectrogram(image=spectrogram)['image']
        else:
            spectrogram = torch.from_numpy(spectrogram)
            spectrogram = spectrogram.permute(2, 0, 1)

        lon = np.radians(sample['longitude'])
        lat = np.radians(sample['latitude'])
        coords_radians = torch.from_numpy(np.stack([lat, lon])).float()
        label = torch.tensor(int(key), dtype=torch.long)  

        return img, spectrogram, label #, coords_radians (only in Eval)
    
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


class SoundingEarthDatasetEval(Dataset):
    
    def __init__(self,
                 data_folder='data',
                 split_csv = 'valid_df.csv',
                 query_type = "sat",
                 transforms = None,
                 patch_time_steps=120,
                 sr_kHz=48,
                 n_mels=128,
                 ):
        
        super().__init__()
 
        self.data_folder = Path(data_folder)
        self.meta = pd.read_csv(str(self.data_folder / split_csv))
        self.query_type = query_type

        self.transforms = transforms     

        # look at SoundingEarthDatasetTrain init for comments
        self.patch_time_steps = patch_time_steps 
        self.sr_kHz = sr_kHz
        self.hop_length = 512
        self.time_step = self.hop_length / sr_kHz * 1e3 
        self.n_mels = n_mels  
           
        if not ( self.query_type == "sat" or self.query_type == "spectro" ):
            raise ValueError("Invalid 'query_type' parameter. 'query_type' must be 'sat' or 'spectro'")
                
    def __getitem__(self, index):

        sample = self.meta.iloc[index]
        key = sample['short_key']
        
        if self.query_type == "sat":
            img = cv2.imread(str(self.data_folder / 'images' / f'{key}.jpg'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # satellite image transforms
            if self.transforms is not None:
                img = self.transforms(image=img)['image']
            else:
                img = torch.from_numpy(img)
                img = img.permute(2, 0, 1)
                            
        else: # if self.query_type == "spectro"
            img = np.load(self.data_folder / 'spectrograms' / f'{self.n_mels}mel_{self.sr_kHz}kHz' / f'{key}.npy')

            # Check if the spectrogram is wide enough, and add equal padding to the left and right if necessary
            if img.shape[1] < self.patch_time_steps:
                padding_width = self.patch_time_steps - img.shape[1]
                left_padding = padding_width // 2
                right_padding = padding_width - left_padding
                img = np.pad(img, ((0, 0), (left_padding, right_padding)), mode='constant', constant_values=-60) # Assumption: -60 dB is considered silence
            
            # # Check if the spectrogram is too long and cut an exact segment from the middle
            # elif img.shape[1] > self.patch_time_steps:
            #     start = (img.shape[1] - self.patch_time_steps) // 2
            #     img = img[:, start:start + self.patch_time_steps]       

            # Check if the spectrogram is too long and cut an exact segment from start
            elif img.shape[1] > self.patch_time_steps:
                img = img[:, :self.patch_time_steps]        
            
            # img = librosa.power_to_db(img, ref=np.max) # only if data isnt in dB, important: change padded values above!

            img = apply_viridis_colormap(img)

            if self.transforms is not None:
                img = self.transforms(image=img)['image']
            else:
                img = torch.from_numpy(img)
                img = img.permute(2, 0, 1)

        lon = sample['longitude']
        lat = sample['latitude']
        coords_radians = torch.tensor([np.radians(lat), np.radians(lon)], dtype=torch.float)
        label = torch.tensor(int(key), dtype=torch.long)  

        return img, label, coords_radians
    
    def __len__(self):
        return len(self.meta)
