import cv2
import random
import copy
import time
import torch
import torchaudio

import numpy as np
import pandas as pd

from math import ceil
from tqdm import tqdm
from pathlib import Path 
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor
from spectrum4geo.utils import Customizable, apply_viridis_colormap


class SatEvalDataset(Dataset):
    def __init__(self,
                 data_folder='data',
                 split_csv = 'valid_df.csv',
                 transforms = None,
                 ):
        
        super().__init__()
 
        self.data_folder = Path(data_folder)
        self.meta = pd.read_csv(str(self.data_folder / split_csv))
        self.idx2key = self.meta['short_key'].to_dict()
        self.transforms = transforms     

    def __getitem__(self, idx):
        key = self.idx2key[idx]

        sat_img = cv2.imread(str(self.data_folder / 'images' / f'{key}.jpg'))
        sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)
        
        # satellite image transforms
        if self.transforms is not None:
            sat_img_tensor = self.transforms(image=sat_img)['image']
        else:
            sat_img_tensor = torch.from_numpy(sat_img)
            sat_img_tensor = sat_img.permute(2, 0, 1)

        label_id_tensor = torch.tensor(idx, dtype=torch.long)  

        return sat_img_tensor, label_id_tensor

    def __len__(self):
        return len(self.meta)


class SpectroEvalDataset(Dataset):
    def __init__(self,
                 data_folder='data',
                 split_csv = 'valid_df.csv',
                 transforms = None,
                 patch_time_steps=4096,
                 sr_kHz=48,
                 n_mels=128,
                 stride=None,
                 min_frame=None,
                 use_power_weights=False,
                 dB_power_weights=False,
                 chunking=False
                 ):
        
        super().__init__()
 
        self.data_folder = Path(data_folder)
        self.meta = pd.read_csv(str(self.data_folder / split_csv))
        self.idx2key = self.meta['short_key'].to_dict()
        self.transforms = transforms     

        # look at SoundingEarthDatasetTrain init for comments
        self.patch_time_steps = patch_time_steps 
        self.sr_kHz = sr_kHz
        self.hop_length = 512
        self.time_step = self.hop_length / sr_kHz * 1e3 
        self.n_mels = n_mels  

        # enables the creation of power-weights for chunks 
        self.use_power_weights = use_power_weights
        # default stride to patch_time_steps if not provided
        self.stride = patch_time_steps - stride if stride is not None else patch_time_steps  
        # min_frame is min_length of cutted spectrogram inside an chunk
        self.min_frame = min_frame if min_frame is not None else 0  
        # Set to True after first initialization
        self.chunking_is_initialized = False
        # Set to True after chunking is switched on 
        self.chunking = False
        # for power weight calculation (if activated)
        self.dB_power_weights = dB_power_weights

        self.switch_chunking(chunking)

    def __init_chunking(self):
        self.cidx2idx = {}
        self.idx2ccount = {}
        self.cidx2chunk = {}
        # generate idx2ccount, cidx2idx and cidx2chunk dicts
        last_idx = 0
        for idx, key in tqdm(self.idx2key.items(), desc="Precalculating amount of chunks", total=len(self.idx2key)):
            spectro_full = np.load(self.data_folder / 'spectrograms' / f'{self.n_mels}mel_{self.sr_kHz}kHz' / f'{key}.npy')
            total_frames = spectro_full.shape[1] - self.min_frame
            chunk = ceil(total_frames / self.stride)

            self.idx2ccount[idx] = chunk        
            for chunk_nr in range(0, chunk):
                self.cidx2idx[last_idx + chunk_nr] = idx 
                self.cidx2chunk[last_idx + chunk_nr] = chunk_nr

            last_idx += chunk 


        # generate power sized weights for chunks
        if self.use_power_weights:
            if self.dB_power_weights:
                self.cidx2mean = {}
                last_idx = None
                for cidx, idx in tqdm(self.cidx2idx.items(), desc="Precalculating (dB) power sized weights for chunks", total=len(self.cidx2idx)):
                    if idx != last_idx:
                        key = self.idx2key[idx]
                        spectro_full = np.load(self.data_folder / 'spectrograms' / f'{self.n_mels}mel_{self.sr_kHz}kHz' / f'{key}.npy')
                        last_idx = idx
                    # Cut the chunk from the spectrogram
                    chunk_num = self.cidx2chunk[cidx]
                    start_sample = chunk_num * self.stride
                    end_sample = start_sample + self.patch_time_steps
                    # Select the specified chunk
                    spectro_chunk = spectro_full[:, start_sample:end_sample]
                    # Check if the spectrogram is wide enough, and add padding to the right side if necessary
                    if spectro_chunk.shape[1] < self.patch_time_steps:
                        padding_width = self.patch_time_steps - spectro_chunk.shape[1]
                        spectro_chunk = np.pad(spectro_chunk, ((0, 0), (0, padding_width)), mode='constant', constant_values=-90)  # to give empty spectrograms not much weight
                    self.cidx2mean[cidx] = np.mean(spectro_chunk)
   
                # use the means of the dB values
                if False:
                    # Convert to positive dB and calculate the sum of the means for each idx
                    self.idx2mean = {idx: 0 for idx in self.idx2ccount.keys()}
                    # Ensure positive dB values >= +1, ignoring reference power
                    min_mean = min(self.cidx2mean.values()) - 1 
                    if min_mean < 0:
                        self.cidx2mean = {cidx: mean - min_mean for cidx, mean in self.cidx2mean.items()}
                    for cidx, idx in self.cidx2idx.items():
                            self.idx2mean[idx] += self.cidx2mean[cidx]

                # Use the linearized means of the dB values. Note that these are not the standard linear means. 
                # Additionally, the mel spectrogram is normalized such that the highest value is 1, corresponding to 0 dB = 0.0.                
                else: 
                    self.idx2mean = {idx: 0 for idx in self.idx2ccount.keys()}
                    self.cidx2mean = {cidx: 1 / (10**(mean / 10)) for cidx, mean in self.cidx2mean.items()}
                    for cidx, idx in self.cidx2idx.items():
                            self.idx2mean[idx] += self.cidx2mean[cidx]

            else:
                self.cidx2mean = {}
                last_idx = None
                for cidx, idx in tqdm(self.cidx2idx.items(), desc="Precalculating (linear) power sized weights for chunks", total=len(self.cidx2chunk)):
                    if idx != last_idx:
                        key = self.idx2key[idx]
                        spectro_full = np.load(self.data_folder / 'spectrograms' / f'{self.n_mels}mel_{self.sr_kHz}kHz' / f'{key}.npy')
                        last_idx = idx
                    # Cut the chunk from the spectrogram
                    chunk_num = self.cidx2chunk[cidx]
                    start_sample = chunk_num * self.stride
                    end_sample = start_sample + self.patch_time_steps
                    # Select the specified chunk
                    spectro_chunk = spectro_full[:, start_sample:end_sample]
                    # Check if the spectrogram is wide enough, and add padding to the right side if necessary
                    if spectro_chunk.shape[1] < self.patch_time_steps:
                        padding_width = self.patch_time_steps - spectro_chunk.shape[1]
                        spectro_chunk = np.pad(spectro_chunk, ((0, 0), (0, padding_width)), mode='constant', constant_values=-90)  # to give empty spectrogram not much weight
                    self.cidx2mean[cidx] = np.log(np.mean(spectro_chunk))
   
                self.idx2mean = {idx: 0 for idx in self.idx2ccount.keys()}
                for cidx, idx in self.cidx2idx.items():
                    self.idx2mean[idx] += self.cidx2mean[cidx]


            self.cidx2weight = {cidx: self.cidx2mean[cidx] / self.idx2mean[self.cidx2idx[cidx]] for cidx in self.cidx2idx.keys()}

        # generate equal sized weights for chunks
        else:
            self.cidx2weight = {cidx: 1 / self.idx2ccount[self.cidx2idx[cidx]] for cidx in self.cidx2idx.keys()}

        self.chunking_is_initialized = True
         
    def __getitem__no_chunking(self, idx):
        key = self.idx2key[idx]

        spectro_tensor = self.__get_spectro(key, 0)
        label_id_tensor = torch.tensor(idx, dtype=torch.long)

        return spectro_tensor, label_id_tensor

    def __getitem__chunking(self, cidx):
        idx = self.cidx2idx[cidx]
        key = self.idx2key[idx]

        chunk_num = self.cidx2chunk[cidx]
        weight = self.cidx2weight[cidx]

        spectro_tensor = self.__get_spectro(key, chunk_num * self.stride)
        label_id_tensor = torch.tensor(idx, dtype=torch.long)
        weight_tensor = torch.tensor(weight, dtype=torch.float)

        return spectro_tensor, label_id_tensor, weight_tensor

    def __get_spectro(self, key, start_sample):
        """for both __getitem__ methods"""
        spectro = np.load(self.data_folder / 'spectrograms' / f'{self.n_mels}mel_{self.sr_kHz}kHz' / f'{int(key)}.npy')
    
        # Cut the chunk from the spectrogram
        end_sample = start_sample + self.patch_time_steps
        spectro = spectro[:, start_sample:end_sample]

        # Check if the spectrogram is wide enough, and add equal padding to the left and right if necessary
        if spectro.shape[1] < self.patch_time_steps:
            padding_width = self.patch_time_steps - spectro.shape[1]
            left_padding = padding_width // 2
            right_padding = padding_width - left_padding
            spectro = np.pad(spectro, ((0, 0), (left_padding, right_padding)), mode='constant', constant_values=-80) # Assumption: -80 dB is considered silence
            
        spectro = apply_viridis_colormap(spectro)

        if self.transforms is not None:
            spectro_tensor = self.transforms(image=spectro)['image']
        else:
            spectro_tensor = torch.from_numpy(spectro)
            spectro_tensor = spectro_tensor.permute(2, 0, 1)

        return spectro_tensor

    @Customizable
    def __getitem__(self, idx):
        ...  # Set during self.switch_chunking(True/False)

    @Customizable
    def __len__(self):
        ...  # Set during self.switch_chunking(True/False)

    def switch_chunking(self, activate):
        if activate:
            self.chunking = True
            if self.chunking_is_initialized is False:
                self.__init_chunking()
            self.__getitem__ = self.__getitem__chunking
            self.__len__ = self.len_of_chunk_ids
        else:
            self.chunking = False
            self.__getitem__ = self.__getitem__no_chunking
            self.__len__ = self.len_of_label_ids

    def len_of_chunk_ids(self):
        return len(self.cidx2idx)

    def len_of_label_ids(self):
        return len(self.idx2key)


class WavEvalDataset(Dataset):
    def __init__(self,
                 data_folder='data',
                 split_csv = 'valid_df.csv',
                 transforms = None,
                 audio_length_s=20,
                 sr_kHz=16,
                 processor_wav2vec2='facebook/wav2vec2-base-960h',
                 stride_s=None,
                 min_frame_s=None,
                 use_power_weights=False,
                 dB_power_weights=False,
                 chunking=False
                 ):
        
        super().__init__()
 
        self.data_folder = Path(data_folder)
        self.meta = pd.read_csv(str(self.data_folder / split_csv))
        self.idx2key = self.meta['short_key'].to_dict()
        self.transforms = transforms     

        self.sr_kHz = sr_kHz
        self.sample_rate = sr_kHz * 1e3
        self.sample_length = int(audio_length_s * self.sample_rate)
        # deactivate normalization since samples are already normalized during torchaudio.load
        self.processor = Wav2Vec2Processor.from_pretrained(processor_wav2vec2, do_normalize=False)

        # enables the creation of power-weights for chunks 
        self.use_power_weights = use_power_weights
        # default stride to sample_length if not provided
        self.stride = self.sample_length - int(stride_s * self.sample_rate) if stride_s is not None else self.sample_length
        # min_frame is min_length of cutted waveform inside an chunk
        self.min_frame = int(min_frame_s * self.sample_rate) if min_frame_s is not None else 0  
        # Set to True after first initialization
        self.chunking_is_initialized = False
        # Set to True after chunking is switched on 
        self.chunking = False
        # for power weight calculation (if activated)
        self.dB_power_weights = dB_power_weights

        self.switch_chunking(chunking)

    def __init_chunking(self):
        self.cidx2idx = {}
        self.idx2ccount = {}
        self.cidx2chunk = {}
        # generate idx2ccount, cidx2idx and cidx2chunk dicts
        last_idx = 0
        for idx, key in tqdm(self.idx2key.items(), desc="Precalculating amount of chunks", total=len(self.idx2key)):
            wav_path = str(self.data_folder / f'mono_audio_wav_{self.sr_kHz}kHz' / f'{key}.wav')
            wav, file_sr = torchaudio.load(wav_path, normalize = True)
            wav = wav.numpy()    
            total_frames = wav.shape[1] - self.min_frame
            chunk = ceil(total_frames / self.stride)

            self.idx2ccount[idx] = chunk        
            for chunk_nr in range(0, chunk):
                self.cidx2idx[last_idx + chunk_nr] = idx 
                self.cidx2chunk[last_idx + chunk_nr] = chunk_nr

            last_idx += chunk 

        # generate power sized weights for chunks
        if self.use_power_weights:            
            self.cidx2mean = {}
            last_idx = None
            for cidx, idx in tqdm(self.cidx2idx.items(), desc="Precalculating (linear) power sized weights for chunks", total=len(self.cidx2chunk)):
                if idx != last_idx:
                    wav_path = str(self.data_folder / f'mono_audio_wav_{self.sr_kHz}kHz' / f'{key}.wav')
                    wav_full, file_sr = torchaudio.load(wav_path, normalize = True)
                    wav_full = torchaudio.functional.vad(wav_full, file_sr)
                    wav_full = wav_full.numpy()    
                    last_idx = idx
                # Cut the chunk from the spectrogram
                chunk_num = self.cidx2chunk[cidx]
                start_sample = chunk_num * self.stride
                end_sample = start_sample + self.sample_length
                # Select the specified chunk
                wav_chunk = wav_full[:, start_sample:end_sample]
                self.cidx2mean[cidx] = np.mean(wav_chunk)

            self.idx2mean = {idx: 0 for idx in self.idx2ccount.keys()}
            for cidx, idx in self.cidx2idx.items():
                self.idx2mean[idx] += self.cidx2mean[cidx]


            self.cidx2weight = {cidx: self.cidx2mean[cidx] / self.idx2mean[self.cidx2idx[cidx]] for cidx in self.cidx2idx.keys()}

        # generate equal sized weights for chunks
        else:
            self.cidx2weight = {cidx: 1 / self.idx2ccount[self.cidx2idx[cidx]] for cidx in self.cidx2idx.keys()}

        self.chunking_is_initialized = True

    def __getitem__no_chunking(self, idx):
        key = self.idx2key[idx]

        wav_np = self.__get_wav(key, 0)
        label_id_tensor = torch.tensor(idx, dtype=torch.long)

        return wav_np, label_id_tensor

    def __getitem__chunking(self, cidx):
        idx = self.cidx2idx[cidx]
        key = self.idx2key[idx]

        chunk_num = self.cidx2chunk[cidx]
        weight = self.cidx2weight[cidx]

        wav_np = self.__get_wav(key, chunk_num * self.stride)
        label_id_tensor = torch.tensor(idx, dtype=torch.long)
        weight_tensor = torch.tensor(weight, dtype=torch.float)

        return wav_np, label_id_tensor, weight_tensor

    def __get_wav(self, key, start_sample):
        """for both __getitem__ methods"""
        wav_path = str(self.data_folder / f'mono_audio_wav_{self.sr_kHz}kHz' / f'{key}.wav')
        wav, file_sr = torchaudio.load(wav_path, normalize = True)
        wav_np = wav.numpy()    

        # Cut the chunk from the waveform
        end_sample = start_sample + self.sample_length
        wav_np = wav_np[:, start_sample:end_sample]

        # Apply audio transforms
        if self.transforms is not None:
            # Reshape the waveform to (1, samples) if it is mono
            if wav_np.ndim == 1:
                wav_np = np.expand_dims(wav_np, axis=0)
            elif wav_np.shape[1] == 1:
                wav_np = wav_np.T
        
            wav_np = self.transforms(samples=wav_np, sample_rate=self.sample_rate)

        # Remove the channel dimension to get a 1D array 
        wav_np = wav_np.squeeze(axis=0)

        return wav_np
    
    @Customizable
    def __getitem__(self, idx):
        ...  # Set during self.switch_chunking(True/False)

    @Customizable
    def __len__(self):
        ...  # Set during self.switch_chunking(True/False)

    def switch_chunking(self, activate):
        if activate:
            self.chunking = True
            if self.chunking_is_initialized is False:
                self.__init_chunking()
            self.__getitem__ = self.__getitem__chunking
            self.__len__ = self.len_of_chunk_ids
        else:
            self.chunking = False
            self.__getitem__ = self.__getitem__no_chunking
            self.__len__ = self.len_of_label_ids

    def len_of_chunk_ids(self):
        return len(self.cidx2idx)

    def len_of_label_ids(self):
        return len(self.idx2key)


class WavEvalDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=64, shuffle=False, chunking=False, collate_fn=None, **kwargs):
        if collate_fn is not None:
            raise ValueError("It is not implemented to provide a 'collate_fn' argument.")
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=None, **kwargs)
        self.switch_chunking(chunking)

    def switch_chunking(self, activate):
        if activate:
            self.chunking = True
            self.dataset.switch_chunking(True)
            self.collate_fn = self.chunked_wav_collate_fn
        else:
            self.chunking = False
            self.collate_fn = self.wav_collate_fn
    
    @Customizable
    def collate_fn(self, batch):
        ...  # Set during self.switch_chunking(True/False)

    def wav_collate_fn(self, batch):
        waveform_data, label_tensor_data = zip(*batch)

        waveform_data_padded = self.dataset.processor(waveform_data, 
                                                      sampling_rate=self.dataset.sample_rate, 
                                                      return_tensors="pt", 
                                                      padding=True, 
                                                      truncation=True, 
                                                      return_attention_mask=True,
                                                      max_length=self.dataset.sample_length
                                                      )

        waveform_tensor_stack = waveform_data_padded['input_values']
        attention_mask_tensor_stack = waveform_data_padded['attention_mask']
        label_tensor_stack = torch.stack(label_tensor_data)

        return waveform_tensor_stack, attention_mask_tensor_stack, label_tensor_stack

    def chunked_wav_collate_fn(self, batch):
        waveform_data, label_tensor_data, weigth_tensor_data = zip(*batch)

        waveform_data_padded = self.dataset.processor(waveform_data, 
                                                      sampling_rate=self.dataset.sample_rate, 
                                                      return_tensors="pt", 
                                                      padding=True, 
                                                      truncation=True, 
                                                      return_attention_mask=True,
                                                      max_length=self.dataset.sample_length
                                                      )
        
        waveform_tensor_stack = waveform_data_padded['input_values'] 
        attention_mask_tensor_stack = waveform_data_padded['attention_mask']
        label_tensor_stack = torch.stack(label_tensor_data)
        weigth_tensor_stack = torch.stack(weigth_tensor_data)

        return waveform_tensor_stack, attention_mask_tensor_stack, label_tensor_stack, weigth_tensor_stack

    def __len__(self):
        return ceil(len(self.dataset)/self.batch_size)