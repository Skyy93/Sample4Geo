import os
import pandas as pd
import torchaudio
import warnings
import torch
import tempfile
warnings.filterwarnings('ignore')

from config import cfg
from tqdm import tqdm

data_path = cfg.data_path
logfile = 'sanity.log'

def simple_logprint(msg):
    print(msg)
    with open(os.path.join(data_path,logfile), 'a') as file:
        file.write(msg + "\n")

def get_audio_paths(metadata_path): 
    df = pd.read_csv(metadata_path)
    audio_paths =[os.path.join(df.iloc[i]['key'],df.iloc[i]['mp3name']) for i in range(len(df))]
    return audio_paths

def check_file(audio_path, failed_paths):
    try:
        audio_path = os.path.join(data_path,"raw_audio", audio_path)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"File not found: {audio_path}") 
        wav, _ = torchaudio.load(audio_path)
        aporee_id = str(audio_path.split('/')[-2])
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, aporee_id + '.pt')
            torch.save(wav, temp_file_path)    
    except FileNotFoundError:
        failed_paths.append(audio_path)
        simple_logprint(f"File not found: {audio_path}. This suggests the audio might no longer be available in the Internet Archive (ia).")
    except Exception as e:
        failed_paths.append(audio_path)
        simple_logprint(f"Unexpected error for {audio_path}: {type(e).__name__}")
        simple_logprint(f"Full exception details: {e}\n")


if __name__ == "__main__":
    audio_paths = get_audio_paths(os.path.join(data_path,"metadata.csv"))
    audio_paths.sort()
    failed_paths = []

    for i in tqdm(range(len(audio_paths))):
        check_file(audio_paths[i], failed_paths)
    print(failed_paths)

    #Save the id corresponding to corrupt mp3s 
    ignore_ids = []
    for f in failed_paths:
        fileid = str(f).split('/')[-2]
        ignore_ids.append(fileid)
    df = pd.DataFrame(ignore_ids,columns=['key'])
    df.to_csv(os.path.join(data_path,"corrupt_ids_final.csv"), index=False)