import os
import pandas as pd
import warnings
from tqdm import tqdm
import datetime
import numpy as np
import librosa
import soundfile as sf

warnings.filterwarnings('ignore')

# Define the data path
raw_audio_path = "data/raw_audio"
metadata_path = "data/final_metadata.csv"
output_dir = "data/mono_audio_wav_16kHz"

logfile = 'data/create_mono_audio_wav_16kHz.log'

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
        correlation = np.corrcoef(audio[0, :], audio[1, :])[0, 1]
        if correlation < -0.9:
            mono_audio = audio[0, :]  # Could also choose audio[1, :]
            simple_logprint(f'Negative correlation detected! => Only using first channel for mono: {key}')
            return mono_audio
    
    # For all other cases, convert to mono using librosa
    return librosa.to_mono(audio)

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
    audio, sr = librosa.load(audio_path, sr=16000, mono=False)  # Load the audio file and resample to 16kHz
    audio = stereo_to_mono_intelligent(audio, short_key)

    output_file_path = os.path.join(output_dir, f"{short_key}.wav")
    sf.write(output_file_path, audio, sr)  # Save the processed audio as WAV

current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
simple_logprint(f"Starting conversion to 16kHz mono WAV files in directory: \"{output_dir}\"")
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
simple_logprint("\nSuccess! All samples have been successfully converted to .wav mono audio and saved.")
simple_logprint("The processed files are located in the directory: '{}'".format(output_dir))
