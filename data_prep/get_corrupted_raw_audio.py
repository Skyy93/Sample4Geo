import pandas as pd
import os

# Paths to the CSV files
corrupt_ids_path = 'data/corrupt_ids_final.csv'
metadata_path = 'data/metadata.csv'

# Reading CSVs
corrupt_ids = pd.read_csv(corrupt_ids_path)
metadata = pd.read_csv(metadata_path)

# Keep only the entries from 'metadata' that are present in 'corrupt_ids'
corrupted_metadata = metadata[metadata['key'].isin(corrupt_ids['key'])]

# Create a directory for the downloads
os.makedirs('data/raw_audio_cor_reload', exist_ok=True)

# Change to the directory
os.chdir('data/raw_audio_cor_reload')

# Prepare the download command for each corrupted audio
for key in corrupted_metadata['key']:
    # Here the Internet Archive Utility (ia) is used for downloading
    # This script assumes that `ia` is installed and configured
    os.system(f'/usr/bin/bash -c "ia download {key} --glob=\'*.mp3\'"')