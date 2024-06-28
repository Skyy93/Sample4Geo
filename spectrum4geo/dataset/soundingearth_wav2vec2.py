import cv2
import random
import copy
import torch
import time
import torchaudio

import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path 
from transformers import Wav2Vec2Processor
from torch.utils.data import Dataset


class Wav2Vec2SoundingEarthDatasetTrain(Dataset):

    def __init__(self,
                 data_folder='data',
                 split_csv = 'train_df.csv',
                 transforms_sat_image=None,
                 transforms_wave=None,
                 audio_length_s=20,
                 sr_kHz=16,
                 processor_wav2vec2='facebook/wav2vec2-base-960h',
                 prob_flip=0.0,
                 prob_rotate=0.0,
                 shuffle_batch_size=128,
                 ):
        
        super().__init__()

        self.data_folder = Path(data_folder)
        self.meta = pd.read_csv(str(self.data_folder / split_csv))

        # Transformations
        self.transforms_sat_image = transforms_sat_image  
        self.transforms_wave = transforms_wave

        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size

        self.sr_kHz = sr_kHz
        self.sample_rate = sr_kHz * 1e3
        self.sample_length = int(audio_length_s * self.sample_rate)
        self.processor = Wav2Vec2Processor.from_pretrained(processor_wav2vec2)


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

        # Load the audio file
        wav_path = str(self.data_folder / f'mono_audio_wav_{self.sr_kHz}kHz' / f'{key}.wav')
        waveform, file_sr = torchaudio.load(wav_path, normalize = True)
        waveform = waveform.numpy()

        # Check if the sample rate matches the expected sample rate
        if file_sr != self.sample_rate:
            raise ValueError(f"Sample rate of {wav_path} is {file_sr*1e-3} kHz, but expected {self.sr_kHz} kHz")
        
        # Choose a random Audiosegment inside the sample
        if waveform.shape[1] > self.sample_length:
            max_start = waveform.shape[1] - self.sample_length
            start = random.randint(0, max_start)
            waveform = waveform[:, start:start + self.sample_length]
        
        # Apply audio transforms
        if self.transforms_wave is not None:
            # Reshape the waveform to (1, samples) if it is mono
            if waveform.ndim == 1:
                waveform = np.expand_dims(waveform, axis=0)
            elif waveform.shape[1] == 1:
                waveform = waveform.T
        
            waveform = self.transforms_wave(samples=waveform, sample_rate=self.sample_rate)

        # Remove the channel dimension to get a 1D array
        waveform = waveform.squeeze(axis=0)

        label = torch.tensor(int(key), dtype=torch.long)  

        return img, waveform, label
    

    def collate_fn(self, batch):
        img_tensor_data, waveform_data, labels_tensor_data = zip(*batch)
        
        waveform_data_padded = self.processor(waveform_data, 
                                              sampling_rate=self.sample_rate, 
                                              return_tensors="pt", 
                                              padding=True, 
                                              truncation=True, 
                                              return_attention_mask=True,
                                              max_length=self.sample_length)

        img_tensor_data = torch.stack(img_tensor_data)
        labels_tensor_data = torch.stack(labels_tensor_data)
        waveform_data_packed = (waveform_data_padded['input_values'], waveform_data_padded['attention_mask'])

        return img_tensor_data, waveform_data_packed, labels_tensor_data


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


class Wav2Vec2SoundingEarthDatasetEval(Dataset):
    
    def __init__(self,
                 data_folder='data',
                 split_csv = 'valid_df.csv',
                 query_type = "sat",
                 transforms = None,
                 audio_length_s=20,
                 sr_kHz=16,
                 processor_wav2vec2='facebook/wav2vec2-base-960h',
                 ):
        
        super().__init__()
 
        self.data_folder = Path(data_folder)
        self.meta = pd.read_csv(str(self.data_folder / split_csv))
        self.query_type = query_type

        self.transforms = transforms     

        self.sr_kHz = sr_kHz
        self.sample_rate = sr_kHz * 1e3
        self.sample_length = int(audio_length_s * self.sample_rate)
        self.processor = Wav2Vec2Processor.from_pretrained(processor_wav2vec2)

        if not ( self.query_type == "sat" or self.query_type == "audio" ):
            raise ValueError("Invalid 'query_type' parameter. 'query_type' must be 'sat' or 'audio'")


    def __getitem__(self, index):

        sample = self.meta.iloc[index]
        key = sample['short_key']
        
        if self.query_type == "sat":
            item = cv2.imread(str(self.data_folder / 'images' / f'{key}.jpg'))
            item = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)
            
            # satellite image transforms
            if self.transforms is not None:
                item = self.transforms(image=item)['image']
            else:
                item = torch.from_numpy(item)
                item = item.permute(2, 0, 1)
                        
        else: # if self.query_type == "audio"

            # Load the audio file
            wav_path = str(self.data_folder / f'mono_audio_wav_{self.sr_kHz}kHz' / f'{key}.wav')
            item, file_sr = torchaudio.load(wav_path, normalize = True)
            item = item.numpy()

            # Check if the sample rate matches the expected sample rate
            if file_sr != self.sample_rate:
                raise ValueError(f"Sample rate of {wav_path} is {file_sr*1e-3} kHz, but expected {self.sr_kHz} kHz")
            
            # Apply audio transforms
            if self.transforms is not None:
                # Reshape the waveform to (1, samples) if it is mono
                if item.ndim == 1:
                    item = np.expand_dims(item, axis=0)
                elif item.shape[1] == 1:
                    item = item.T
            
                item = self.transforms(samples=item, sample_rate=self.sample_rate)

            # Remove the channel dimension to get a 1D array 
            item = item.squeeze(axis=0)

        label = torch.tensor(int(key), dtype=torch.long)  

        return item, label


    def collate_fn(self, batch):
        item_data, labels_tensor_data = zip(*batch)

        if self.query_type == "sat":
            item_stack = (torch.stack(item_data), None)
        else:
            waveform_data_padded = self.processor(item_data, 
                                                sampling_rate=self.sample_rate, 
                                                return_tensors="pt", 
                                                padding=True, 
                                                truncation=True, 
                                                return_attention_mask=True,
                                                max_length=self.sample_length)
            
            item_stack = (waveform_data_padded['input_values'], waveform_data_padded['attention_mask'])

        labels_stack = torch.stack(labels_tensor_data)

        return item_stack, labels_stack
    
    def __len__(self):
        return len(self.meta)
