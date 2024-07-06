import cv2
import random
import copy
import time
import torch
import torchaudio

import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path 
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor
from spectrum4geo.utils import apply_viridis_colormap


class SatSpectroTrainDataset(Dataset):
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
        self.idx2key = self.meta['short_key'].to_dict()
        self.transforms_sat_image = transforms_sat_image  
        self.transforms_spectrogram = transforms_spectrogram   
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        
        self.patch_time_steps = patch_time_steps # Count of datapoints (X-Direction)
        self.sr_kHz = sr_kHz
        self.hop_length = 512
        # hop length: librosa(default) = 512
        # if default: hop_length = win_length//4
        #                          win_length = n_fft
        #                                       n_fft = 2048
        self.time_step = self.hop_length / sr_kHz * 1e3 #  timestep between two datapoints (X-Direction)
        self.n_mels = n_mels  # Specifies the number of Mel bands (frequency bands) to use in the Mel spectrogram
        
        self.shuffle_batch_size = shuffle_batch_size
        self.train_ids = self.meta.index.tolist()
        self.samples = copy.deepcopy(self.train_ids)
                                   
    def __getitem__(self, index):
        idx = self.samples[index]
        key = self.idx2key[idx]

        sat_img = cv2.imread(str(self.data_folder / 'images' / f'{key}.jpg'))
        sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)

        # Flip satellite
        if np.random.random() < self.prob_flip:
            sat_img = cv2.flip(sat_img, 1)
                       
        # satellite image transforms
        if self.transforms_sat_image is not None:
            sat_img_tensor = self.transforms_sat_image(image=sat_img)['image']
        else:
            sat_img_tensor = torch.from_numpy(sat_img)
            sat_img_tensor = sat_img_tensor.permute(2, 0, 1)

        # rotate sat img 90 or 180 or 270
        if np.random.random() < self.prob_rotate:
            r = np.random.choice([1,2,3])
            sat_img_tensor = torch.rot90(sat_img_tensor, k=r, dims=(1, 2)) 

        spectrogram = np.load(self.data_folder / 'spectrograms' / f'{self.n_mels}mel_{self.sr_kHz}kHz' / f'{key}.npy')

        # Check if the spectrogram is too short, and if so, add the needed padding on the left and right randomly
        if spectrogram.shape[1] < self.patch_time_steps:
            padding_width = self.patch_time_steps - spectrogram.shape[1]
            left_padding = np.random.randint(0, padding_width + 1) 
            right_padding = padding_width - left_padding
            spectrogram = np.pad(spectrogram, ((0, 0), (left_padding, right_padding)), mode='constant', constant_values=-80) # Assumption: -80 dB is considered silence

        # Cut a random patch if the spectrogram is wide enough
        elif spectrogram.shape[1] > self.patch_time_steps:
            start = torch.randint(0, spectrogram.shape[1] - self.patch_time_steps + 1, (1,)).item()
            spectrogram = spectrogram[:, start:start + self.patch_time_steps]
            
        spectrogram = apply_viridis_colormap(spectrogram)

        if self.transforms_spectrogram is not None:
            spectrogram_tensor = self.transforms_spectrogram(image=spectrogram)['image']
        else:
            spectrogram_tensor = torch.from_numpy(spectrogram)
            spectrogram_tensor = spectrogram.permute(2, 0, 1)

        id_label_tensor = torch.tensor(idx, dtype=torch.long)  

        return sat_img_tensor, spectrogram_tensor, id_label_tensor 
    
    def __len__(self):
        return len(self.samples)
        
    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):
        '''
        Custom shuffle function for unique class_id sampling in batch
        '''
        print("\nShuffle Dataset:")

        # Prepare a list of indices based on the metadata dataframe
        idx_pool = copy.deepcopy(self.train_ids)
        
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
        print("Original Length: {} - Length after Shuffle: {}".format(len(self.meta), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.meta) - len(self.samples))
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0], self.samples[-1]))


class SatWavTrainDataset(Dataset):
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
        self.idx2key = self.meta['short_key'].to_dict()

        self.transforms_sat_image = transforms_sat_image  
        self.transforms_wave = transforms_wave
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate

        self.sr_kHz = sr_kHz
        self.sample_rate = sr_kHz * 1e3
        self.sample_length = int(audio_length_s * self.sample_rate)
        # deactivate normalization since samples are already normalized during torchaudio.load
        self.processor = Wav2Vec2Processor.from_pretrained(processor_wav2vec2, do_normalize=False)

        self.shuffle_batch_size = shuffle_batch_size
        self.train_ids = self.meta.index.tolist()
        self.samples = copy.deepcopy(self.train_ids)

    def __getitem__(self, index):
        idx = self.samples[index]
        key = self.idx2key[idx]

        img = cv2.imread(str(self.data_folder / 'images' / f'{key}.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Flip satellite
        if np.random.random() < self.prob_flip:
            img = cv2.flip(img, 1)
                       
        # satellite image transforms
        if self.transforms_sat_image is not None:
            img_tensor = self.transforms_sat_image(image=img)['image']
        else:
            img_tensor = torch.from_numpy(img)
            img_tensor = img_tensor.permute(2, 0, 1)

        # rotate sat img 90 or 180 or 270
        if np.random.random() < self.prob_rotate:
            r = np.random.choice([1,2,3])
            img_tensor = torch.rot90(img_tensor, k=r, dims=(1, 2)) 

        # Load the audio file
        wav_path = str(self.data_folder / f'mono_audio_wav_{self.sr_kHz}kHz' / f'{key}.wav')
        waveform, file_sr = torchaudio.load(wav_path, normalize = True)
        waveform = torchaudio.functional.vad(waveform, file_sr)
        waveform_np = waveform.numpy()

        # Check if the sample rate matches the expected sample rate
        if file_sr != self.sample_rate:
            raise ValueError(f"Sample rate of {wav_path} is {file_sr*1e-3} kHz, but expected {self.sr_kHz} kHz")
        
        # Choose a random Audiosegment inside the sample
        if waveform_np.shape[1] > self.sample_length:
            max_start = waveform_np.shape[1] - self.sample_length
            start = random.randint(0, max_start)
            waveform_np = waveform_np[:, start:start + self.sample_length]
        
        # Apply audio transforms
        if self.transforms_wave is not None:
            # Reshape the waveform to (1, samples) if it is mono
            if waveform_np.ndim == 1:
                waveform_np = np.expand_dims(waveform_np, axis=0)
            elif waveform.shape[1] == 1:
                waveform_np = waveform_np.T
        
            waveform_np = self.transforms_wave(samples=waveform_np, sample_rate=self.sample_rate)

        # Remove the channel dimension to get a 1D array
        waveform_np = waveform_np.squeeze(axis=0)

        label_id_tensor = torch.tensor(idx, dtype=torch.long)  

        return img_tensor, waveform_np, label_id_tensor

    def collate_fn(self, batch):
        """This function needs to be provided to the DataLoader during initialization."""
        img_tensor_data, waveform_np_data, label_id_tensor_data = zip(*batch)
        
        waveform_data_padded = self.processor(waveform_np_data, 
                                              sampling_rate=self.sample_rate, 
                                              return_tensors="pt", 
                                              padding=True, 
                                              truncation=True, 
                                              return_attention_mask=True,
                                              max_length=self.sample_length
                                              )

        img_tensor_data = torch.stack(img_tensor_data)
        label_id_tensor_data = torch.stack(label_id_tensor_data)
        waveform_tensor_data = waveform_data_padded['input_values'] 
        attention_mask_tensor_data = waveform_data_padded['attention_mask']

        return img_tensor_data, waveform_tensor_data, attention_mask_tensor_data, label_id_tensor_data

    def __len__(self):
        return len(self.samples)
        
    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):
        '''
        Custom shuffle function for unique class_id sampling in batch
        '''
        print("\nShuffle Dataset:")

        # Prepare a list of indices based on the metadata dataframe
        idx_pool = copy.deepcopy(self.train_ids)
        
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
        print("Original Length: {} - Length after Shuffle: {}".format(len(self.meta), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.meta) - len(self.samples))
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0], self.samples[-1]))

