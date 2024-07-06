import os
import datetime
import numpy as np
import pandas as pd
import soundfile as sf

from tqdm import tqdm
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS

# Define paths
mono_audio_path = "data/mono_audio_wav_16kHz"
metadata_path = "data/final_metadata.csv"
output_dir = "data/mono_audio_wav_16kHz_trimmed"
logfile = 'data/create_mono_audio_wav_16kHz_trimmed.log'

def simple_logprint(msg):
    print(msg)
    with open(logfile, 'a') as file:
        file.write(msg + "\n")

def get_audio_paths(metadata_path):
    """ Reads the CSV file and generates paths and short_keys for audio files. """
    df = pd.read_csv(metadata_path)
    # Creating a list of tuples (audio_path, short_key)
    audio_info = [(os.path.join(mono_audio_path, f'{df.iloc[i]["short_key"]}.wav'), df.iloc[i]['short_key']) for i in range(len(df))]
    return audio_info

def read_and_remove_silence(audio_path):
    """Reads audio file, removes silence, and returns the sample rate and trimmed signal."""
    [Fs, x] = aIO.read_audio_file(audio_path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    segments = aS.silence_removal(x, Fs, 0.020, 0.020, smooth_window=1.0, weight=0.3, plot=False)
    trimmed_signal = np.concatenate([x[int(Fs * start):int(Fs * stop)] for start, stop in segments], axis=0)
    return trimmed_signal, Fs

def process_audio_file(audio_info):
    """ Processes each audio file by removing silence and saving the trimmed version. """
    audio_path, short_key = audio_info
    try:
        trimmed_audio, Fs = read_and_remove_silence(audio_path)
        output_file_path = os.path.join(output_dir, f"{short_key}.wav")
        sf.write(output_file_path, trimmed_audio, Fs)  # Save the trimmed audio
    except Exception as e:
        simple_logprint(f"Error processing {audio_path}: {str(e)}")


# Main execution logic
if __name__ == "__main__":
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    simple_logprint(f"Starting Trimming WAV files, to be stored in the directory: \"{output_dir}\"")
    simple_logprint(f"Process started at {current_datetime}")

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Get paths and short_keys to audio files from metadata
    audio_info = get_audio_paths(metadata_path)

    # Processing audio files
    for info in tqdm(audio_info):
        process_audio_file(info)  # Any error here will be logged

    # Success message
    simple_logprint("\nSuccess! All samples have been successfully ______ as .wav saved.")
    simple_logprint("The processed files are located in the directory: '{}'".format(output_dir))
