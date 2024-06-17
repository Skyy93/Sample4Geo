import os
import librosa
import numpy as np
import pandas as pd
from config import cfg
import datetime


########################################################################################################################
########## Define parameters ######################
mel_bins = 160
# 16 kHz for speech, 22.05 kHz as a compromise, 44.1/48 kHz for ambient sounds, None to keep the original sampling rate
sr_kHz = 48 #22.05 # 48kHz (to compare results with geoclap) 
sr = sr_kHz * 1e3

logfile = 'data/create_spectrogramm.log'
metadata_file = 'final_metadata.csv' 
###################################################
########################################################################################################################

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


# Creating data folder and write first lines into logfile
output_folder = os.path.join(cfg.sat_audio_spectrograms_path, f"{mel_bins}mel_{sr_kHz}kHz")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
simple_logprint(f"Initiating the process to generate spectrograms in the directory \"{mel_bins}mel_{sr_kHz}kHz\"")
simple_logprint(f"Process started at {current_datetime}")

# Read metadata
metadata_path = os.path.join(cfg.data_path, metadata_file)
df = pd.read_csv(metadata_path)

# Initialize calculation of total size of Mel spectrograms and total processed size
total_mel_size = 0
processed_files_size = 0
processed_files_count = 0
files_5GB_counter = 0
total_files_size = df['mp3mb'].sum() / 1024  # Convert from MB to GB

# Initialize list to store rows with non-NULL data
valid_rows = []

# Iterate through each row of metadata
for idx, row in df.iterrows():
    mp3_path = os.path.join(cfg.sat_audio_path, row['key'], row['mp3name'])
    output_path = os.path.join(output_folder, f"{row['short_key']}.npy")
    
    # Load MP3 and calculate Mel spectrogram
    audio, sr_audio = librosa.load(mp3_path, sr=sr, mono=False)  # sr=None to keep the original sampling rate
    mono_audio = stereo_to_mono_intelligent(audio, key=row['key'], logfile=logfile)
    S = librosa.feature.melspectrogram(y=mono_audio, sr=sr_audio, n_mels=mel_bins)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # Save the Mel spectrogram as a NumPy array
    np.save(output_path, S_dB)

    # Update statistics
    mel_size = os.path.getsize(output_path) / (1024 ** 3)  # Size in GB
    file_size_gb = row['mp3mb'] / 1024  # Size in GB
    total_mel_size += mel_size
    processed_files_size += file_size_gb
    files_5GB_counter += file_size_gb
    processed_files_count += 1
    
    # Print status every 5GB of processed audio data
    if files_5GB_counter >= 5:
        simple_logprint(f"Processed {processed_files_size:.2f} GB of {total_files_size:.2f} GB audio files ({processed_files_count} files)")
        simple_logprint(f"Total data size of saved arrays so far: {total_mel_size:.2f} GB")
        files_5GB_counter = 0  # Reset after each output

# Final output
if processed_files_count != len(df):
    missing_files_count = len(df) - processed_files_count
    simple_logprint(f'Caution: Only {processed_files_count} of {len(df)} spectrograms created. {missing_files_count} files are missing.')
else:
    simple_logprint(f'All {processed_files_count} spectrograms have been successfully created.')
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
simple_logprint(f"Total data size of saved arrays: {total_mel_size:.2f} GB")
simple_logprint(f"Process finished at {current_datetime}")

#### TODO: Maybe wrong calculation of total size (audio files) => total_files_size