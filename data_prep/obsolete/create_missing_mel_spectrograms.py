import os, glob
import librosa
import numpy as np
import pandas as pd
from config import cfg
import datetime

########################################################################################################################
########## Define parameters ######################
mel_bins = 128
sr_kHz = 48 # 48kHz (to compare results with geoclap) 
sr = sr_kHz * 1e3

logfile = 'data/create_spectrogramm.log'
metadata_file = 'final_metadata.csv'

########################################################################################################################

def simple_logprint(msg):
    print(msg)
    with open(logfile, 'a') as file:
        file.write(msg + "\n")

def stereo_to_mono_intelligent(audio, key, logfile):
    if audio.ndim == 1:
        return audio
    
    if audio.ndim == 2 and audio.shape[0] == 2:
        correlation = np.corrcoef(audio[0, :], audio[1, :])[0, 1]
        if correlation < -0.9:
            mono_audio = audio[0, :]  # Could also choose audio[1, :]
            simple_logprint(f'Negative correlation detected! => Only using first channel for mono: {key}')
            return mono_audio
    
    return librosa.to_mono(audio)

output_folder = os.path.join(cfg.sat_audio_spectrograms_path, f"{mel_bins}mel_{sr_kHz}kHz")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
simple_logprint(f"Initiating the process to generate missing spectrograms in the directory \"{mel_bins}mel_{sr_kHz}kHz\"")
simple_logprint(f"Process started at {current_datetime}")

metadata_path = os.path.join(cfg.data_path, metadata_file)
df = pd.read_csv(metadata_path)

for idx, row in df.iterrows():
    mp3_path = os.path.join(cfg.sat_audio_path, row['key'], row['mp3name'])
    output_path = os.path.join(output_folder, f"{row['short_key']}.npy")
    
    # Check if the NPY file already exists and skip processing if it does
    if not os.path.exists(output_path):
        try:
            audio, sr_audio = librosa.load(mp3_path, sr=sr, mono=False)
            mono_audio = stereo_to_mono_intelligent(audio, key=row['key'], logfile=logfile)
            S = librosa.feature.melspectrogram(y=mono_audio, sr=sr_audio, n_mels=mel_bins)
            S_dB = librosa.power_to_db(S, ref=np.max)
            np.save(output_path, S_dB)
            simple_logprint(f"Missing .npy file created: {output_path}")
        except Exception as e:
            simple_logprint(f"Error processing {mp3_path}: {str(e)}")

npy_files = glob.glob(os.path.join(output_folder, '*.npy'))
spectrogram_count = len(npy_files)

# Final output
if spectrogram_count != len(df):
    missing_files_count = len(df) - spectrogram_count
    simple_logprint(f'Caution: Only {spectrogram_count} of {len(df)} spectrograms created. {missing_files_count} files are missing.')
else:
    simple_logprint(f'All {spectrogram_count} spectrograms have been successfully created.')
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
simple_logprint(f"Process finished at {current_datetime}")