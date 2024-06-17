import os
import pandas as pd
import torchaudio
import torch
import warnings
from tqdm import tqdm
import datetime
import numpy as np

warnings.filterwarnings('ignore')

# Define the data path
raw_audio_path = "data/raw_audio"
metadata_path = "data/final_metadata.csv"
output_dir = "data/raw_audio_tensorized"

logfile = 'data/create_tensors.log'

def simple_logprint(msg):
    print(msg)
    with open(logfile, 'a') as file:
        file.write(msg + "\n")

def stereo_to_mono_intelligent(audio, key):
    # Early return for mono audio to avoid unnecessary processing
    if audio.ndim == 1:
        return audio
    
    # Handle stereo audio with specific correlation check
    if audio.ndim == 2 and audio.shape[0] == 2:
        # Convert torch tensor to numpy array for correlation computation
        audio_np = audio.numpy()
        correlation = np.corrcoef(audio_np[0, :], audio_np[1, :])[0, 1]
        if correlation < -0.9:
            mono_audio = audio[0, :]  # Could also choose audio[1, :]
            simple_logprint(f'Negative correlation detected! => Only using first channel for mono: {key}')
            return mono_audio
    
    # For all other cases, convert to mono by averaging the channels
    return torch.mean(audio, dim=0)

def get_audio_paths(metadata_path):
    """ Reads the CSV file and generates paths and short_keys for audio files. """
    df = pd.read_csv(metadata_path)
    # Creating a list of tuples (audio_path, short_key)
    audio_info = [(os.path.join(raw_audio_path, df.iloc[i]['key'], df.iloc[i]['mp3name']), df.iloc[i]['short_key']) for i in range(len(df))]
    return audio_info

def process_audio_file(audio_info):
    """ Processes each audio file, converting it to a tensor and saving with short_key as filename. 
        Throws an error and crashes if the file cannot be processed.
    """
    audio_path, short_key = audio_info
    wav, _ = torchaudio.load(audio_path)  # This will throw an error if the file cannot be processed
    wav = stereo_to_mono_intelligent(wav, short_key)

    temp_file_path = os.path.join(output_dir, f"{short_key}.pt")
    torch.save(wav, temp_file_path)


current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
simple_logprint(f"Initiating the process to generate tensors in the directory \"{output_dir}\"")
simple_logprint(f"Process started at {current_datetime}")

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Get paths and short_keys to audio files from metadata
audio_info = get_audio_paths(metadata_path)
audio_info.sort(key=lambda x: x[1])  # Sort by short_key if needed

# Processing audio files
for info in tqdm(audio_info):
    process_audio_file(info)  # Any error here will crash the program

# Success message
simple_logprint("\nSuccess! All samples have been successfully converted to tensors and saved.")
simple_logprint("The processed files are located in the directory: '{}'".format(output_dir))


