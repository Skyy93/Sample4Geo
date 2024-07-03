import cv2
import random
import copy
import time
import torch
import torchaudio

import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from math import ceil
from tqdm import tqdm
from pathlib import Path 
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor


def apply_viridis_colormap(array):
    norm = plt.Normalize(vmin=array.min(), vmax=array.max())
    cmap = cm.get_cmap('viridis')
    rgba_array = cmap(norm(array)).astype(np.float32)
    rgb_array = rgba_array[:, :, :3]
    return rgb_array


class SatEvalDataset(Dataset):
    def __init__(self,
                 data_folder='data',
                 split_csv = 'valid_df.csv',
                 transforms = None,
                 ):
        
        super().__init__()
 
        self.data_folder = Path(data_folder)
        self.meta = pd.read_csv(str(self.data_folder / split_csv))
        self.transforms = transforms     


    def __getitem__(self, index):
        sample = self.meta.iloc[index]
        key = sample['short_key']

        img = cv2.imread(str(self.data_folder / 'images' / f'{key}.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # satellite image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        else:
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1)

        label = torch.tensor(int(key), dtype=torch.long)  

        return img, label

    def __len__(self):
        return len(self.meta)


class SpectroEvalDataset(Dataset):
    def __init__(self,
                 data_folder='data',
                 split_csv = 'valid_df.csv',
                 transforms = None,
                 patch_time_steps=120,
                 sr_kHz=48,
                 n_mels=128,
                 ):
        
        super().__init__()
 
        self.data_folder = Path(data_folder)
        self.meta = pd.read_csv(str(self.data_folder / split_csv))
        self.transforms = transforms     

        # look at SoundingEarthDatasetTrain init for comments
        self.patch_time_steps = patch_time_steps 
        self.sr_kHz = sr_kHz
        self.hop_length = 512
        self.time_step = self.hop_length / sr_kHz * 1e3 
        self.n_mels = n_mels  


    def __getitem__(self, index):
        sample = self.meta.iloc[index]
        key = sample['short_key']
        label = torch.tensor(int(key), dtype=torch.long)  
        img = np.load(self.data_folder / 'spectrograms' / f'{self.n_mels}mel_{self.sr_kHz}kHz' / f'{int(key)}.npy')
 
        # Check if the spectrogram is wide enough, and add equal padding to the left and right if necessary
        if img.shape[1] < self.patch_time_steps:
            padding_width = self.patch_time_steps - img.shape[1]
            left_padding = padding_width // 2
            right_padding = padding_width - left_padding
            img = np.pad(img, ((0, 0), (left_padding, right_padding)), mode='constant', constant_values=-60) # Assumption: -60 dB is considered silence

        # Check if the spectrogram is too long and cut an exact segment from start
        elif img.shape[1] > self.patch_time_steps:
            img = img[:, :self.patch_time_steps]   
        
        img = apply_viridis_colormap(img)

        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        else:
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1)

        return img, label


    def __len__(self):
        return len(self.meta)



# -> include chunking loading directly in function which is used to create batches!!! (not collate_fm)
class SpectroTestDataset(SpectroEvalDataset):
    def __init__(self,
                 data_folder='data',
                 split_csv='valid_df.csv',
                 transforms=None,
                 patch_time_steps=120,
                 sr_kHz=48,
                 n_mels=128,
                 stride=None,
                 min_frame=None,
                 ):
        
        super().__init__(data_folder, split_csv, transforms, patch_time_steps, sr_kHz, n_mels)

        self.last_indices = None
        self.next_chunk_counter = 0
        # Default stride to patch_time_steps if not provided
        self.stride = stride if stride is not None else patch_time_steps  
        # min_frame is min_length of cutted spectrogram inside an chunk
        self.min_frame = min_frame if min_frame is not None else 0  

        self.chunk_count = {}
        self.chunk_weight = {}
        # Generate chunk_count and chunk_weight dicts
        for key in tqdm(self.meta["short_key"].tolist(), desc="Precalculating Chunk Counts and Weights"):
            img = np.load(self.data_folder / 'spectrograms' / f'{self.n_mels}mel_{self.sr_kHz}kHz' / f'{key}.npy')
            total_frames = img.shape[1] - self.min_frame
            chunk = ceil(total_frames / self.stride)

            self.chunk_count[key] = chunk
            self.chunk_weight[key] = 1 / chunk 


    def __get_chunk(self, key, chunk, min_frame):
            img = np.load(self.data_folder / 'spectrograms' / f'{self.n_mels}mel_{self.sr_kHz}kHz' / f'{int(key)}.npy')
 
            # Select the specified chunk
            start_sample = chunk * self.stride
            end_sample = start_sample + self.patch_time_steps

            # Cut the chunk from the spectrogram
            img = img[:, start_sample:end_sample]

            # Check if the spectrogram is wide enough, and add equal padding to the left and right if necessary
            if img.shape[1] < self.patch_time_steps:
                padding_width = self.patch_time_steps - img.shape[1]
                left_padding = padding_width // 2
                right_padding = padding_width - left_padding
                img = np.pad(img, ((0, 0), (left_padding, right_padding)), mode='constant', constant_values=-60) # Assumption: -60 dB is considered silence
            
            img = apply_viridis_colormap(img)

            if self.transforms is not None:
                img = self.transforms(image=img)['image']
            else:
                img = torch.from_numpy(img)
                img = img.permute(2, 0, 1)
            return img

    def get_next_chunk(self, key):
        img = self.__get_chunk(key, chunk=self.next_chunk_counter, min_frame=self.min_frame)
        label = torch.tensor(int(key), dtype=torch.long)  
        self.next_chunk_counter +=1
        return img, label

    def get_chunk_weights(self, key_list):
        return [self.chunk_weight[int(key)] for key in key_list]
    
    def get_chunk_count(self, key):
        return self.chunk_count[int(key)]


class WavEvalDataset(Dataset):
    
    def __init__(self,
                 data_folder='data',
                 split_csv = 'valid_df.csv',
                 transforms = None,
                 audio_length_s=20,
                 sr_kHz=16,
                 processor_wav2vec2='facebook/wav2vec2-base-960h',
                 ):
        
        super().__init__()
 
        self.data_folder = Path(data_folder)
        self.meta = pd.read_csv(str(self.data_folder / split_csv))

        self.transforms = transforms     

        self.sr_kHz = sr_kHz
        self.sample_rate = sr_kHz * 1e3
        self.sample_length = int(audio_length_s * self.sample_rate)
        self.processor = Wav2Vec2Processor.from_pretrained(processor_wav2vec2)


    def __getitem__(self, index):

        sample = self.meta.iloc[index]
        key = sample['short_key']
        
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
        waveform_data, labels_tensor_data = zip(*batch)

        waveform_data_padded = self.processor(waveform_data, 
                                              sampling_rate=self.sample_rate, 
                                              return_tensors="pt", 
                                              padding=True, 
                                              truncation=True, 
                                              return_attention_mask=True,
                                              max_length=self.sample_length
                                              )
        
        waveform_stack = (waveform_data_padded['input_values'], waveform_data_padded['attention_mask'])
        labels_stack = torch.stack(labels_tensor_data)

        return item_stack, labels_stack
    
    def __len__(self):
        return len(self.meta)


    def get_next_chunk(self, key, counter):
        wav_path = str(self.data_folder / f'mono_audio_wav_{self.sr_kHz}kHz' / f'{key}.wav')
        item, file_sr = torchaudio.load(wav_path, normalize=True)
        item = item.numpy()

        start_sample = counter * self.sample_length
        end_sample = start_sample + self.sample_length

        if end_sample <= item.shape[1]:
            item = item[:, start_sample:end_sample]
        else:
            return None

        if self.transforms is not None:
            if item.ndim == 1:
                item = np.expand_dims(item, axis=0)
            elif item.shape[1] == 1:
                item = item.T
            item = self.transforms(samples=item, sample_rate=self.sample_rate)

        waveform_data = self.processor(item,
                                       sampling_rate=self.sample_rate,
                                       return_tensors="pt",
                                       padding=True,
                                       truncation=True,
                                       return_attention_mask=True,
                                       max_length=self.sample_length
                                       )

        item = waveform_data['input_values'] 
        attention_mask = waveform_data['attention_mask']

        return item, attention_mask







class SpectroTestDataloader(DataLoader):
    def __init__(self, 
    dataset, 
    batch_size=1, 
    shuffle=False, 
    collate_fn=None, 
    chunking=False, 
    **kwargs):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, **kwargs)
        self.chunking = chunking
        self.extant_chunks = []

    def set_label_ids(self, label_ids):
        self.label_ids = label_ids.tolist()

    def get_next_chunks_batched(self, _):
        if not self.chunking:
            return None

        next_chunks = self.extant_chunks
        self.extant_chunks = []

        while self.label_ids:  
            key = self.label_ids.pop()  
            self.set_next_chunk_counter(1)
            for _ in range(self.dataset.get_chunk_count(key)):
                next_chunks.append(self.dataset.get_next_chunk(key))
            if len(next_chunks) >= self.batch_size:
                break

        # Split valid_chunks into chunk_batch (self.batch_size or smaller)
        chunk_batch = next_chunks[:self.batch_size]
        self.extant_chunks = next_chunks[self.batch_size:]

        if not chunk_batch:
            return None

        item_data, labels_tensor_data = zip(*chunk_batch)
        return torch.stack(item_data), torch.stack(labels_tensor_data)

    def reset_next_chunk_counter(self):
        self.dataset.next_chunk_counter = 0

    def increment_next_chunk_counter(self):
        self.dataset.next_chunk_counter += 1

    def set_next_chunk_counter(self,num):
        self.dataset.next_chunk_counter = num

    def get_chunks_weights(self, _):
        if not self.chunking:
            return torch.ones((len(label_ids), 1))
        weights = torch.tensor(self.dataset.get_chunk_weights(self.label_ids))
        return weights.view(-1, 1)



# 1.) zuerst chunking logic direkt im dataset integrieren 
# 2.) Gewichte boolean einführen
# ((vllt.)) Skript schreiben um Gewichte für Spektren sowie Counts in csv.Dateien speichern (nur wenn es zu Lange dauert)
# 3.) Selbe Logik in Wav2Vec2 implementieren