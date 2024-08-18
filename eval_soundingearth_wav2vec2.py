import os
import re
import sys
import time
import torch

from dataclasses import dataclass
from torch.utils.data import DataLoader

from spectrum4geo.utils import Logger
from spectrum4geo.model import TimmModelWav2Vec2
from spectrum4geo.evaluate.metrics import evaluate
from spectrum4geo.transforms import get_transforms_val_sat
from spectrum4geo.dataset.evaluation import SatEvalDataset, WavEvalDataset, WavEvalDataLoader


def extract_checkpoint_info(checkpoint_path):
    # Split the path into parts by '/'
    checkpoint_parts = checkpoint_path.split('/')

    # Find the parts containing relevant information
    kHz_part = next(part for part in checkpoint_parts if 'kHz' in part).split('_')
    patch_batch_part = next(part for part in checkpoint_parts if 'batch_size' in part).split('_')
    shuffle_part = next(part for part in checkpoint_parts if 'Shuffle' in part)

    # Extract values
    sr_kHz = int(kHz_part[0].replace('kHz', ''))
    audio_length_s = int(patch_batch_part[0])
    batch_size = int(patch_batch_part[4]) 
    shuffle = shuffle_part == 'Shuffle_On'

    return sr_kHz, audio_length_s, batch_size, shuffle


@dataclass
class Configuration:
    
    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    model_wav2vec2: str = 'facebook/wav2vec2-large-960h'  

    # facebook/wav2vec2-base-960h
    # facebook/wav2vec2-large-960h
    # facebook/wav2vec2-large-960h-lv60-self

    checkpoint_start = 'soundingearth_wav2vec2/training/16_kHz/15_audio_length_s_112_batch_size/Shuffle_Off/convnext_base.fb_in22k_ft_in1k_384/210037_0.000573_lr_3.818813079129867e-05_lr_wav2vec2/weights_end.pth'   
    sr_kHz, audio_length_s, batch_size, shuffle = extract_checkpoint_info(checkpoint_start)

    # Override model image size
    img_size: int = 384                                        # for satallite images

    # Evaluation
    batch_size_eval: int = 64*4
    verbose: bool = True
    gpu_ids: tuple =  (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)  # GPU ids for evaluating
    normalize_features: bool = True
    
    # Savepath for model eval logs
    if shuffle == True:
        log_path =  f'./soundingearth_wav2vec2/testing/{sr_kHz}_kHz/{audio_length_s}_audio_length_s_{batch_size}_batch_size/Shuffle_On'
    else:
        log_path =  f'./soundingearth_wav2vec2/testing/{sr_kHz}_kHz/{audio_length_s}_audio_length_s_{batch_size}_batch_size/Shuffle_Off'

    # Dataset
    data_folder = 'data'        
    evaluate_csv = 'test_df.csv' 
  
    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

config = Configuration() 

if __name__ == '__main__':

    # Directory path
    directory = "soundingearth/training/"

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join('soundingearth/testing/', 'init_log.txt'))
    logger = sys.stdout


    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    model_path = f'{config.log_path}/{time.strftime("%H%M%S")}'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    logger.set_log_file(os.path.join(model_path, 'log.txt'))

    print(f'\nModel: {config.model}')

    print(f'Used .csv file for evaluating: {config.evaluate_csv}')

    model = TimmModelWav2Vec2(config.model,
                                config.model_wav2vec2,
                                pretrained=True,
                                img_size=config.img_size
                                )
                        
    data_config = model.get_config()
    print(data_config)
    mean = data_config['mean']
    std = data_config['std']
    img_size = config.img_size
    
    img_size_sat = (img_size, img_size)
    
    # load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print('Start from:', config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=False)     

    # Data parallel
    print('\nGPUs available:', torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            
    # Print information about audio related settings
    print(f'Model sampling rate: {config.sr_kHz} kHz')
    print(f'Audio segment length: {config.audio_length_s} s')

    # Model to device   
    model = model.to(config.device)

    print('\nImage Size Sat:', img_size_sat)
    print(f'Mean: {mean}')
    print(f'Std:  {std}\n') 

    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#
    
    # Eval
    sat_transforms_val = get_transforms_val_sat(img_size_sat,
                                                mean=mean,
                                                std=std
                                                )
    
    # Satellite Images (Reference)
    sat_dataset_test = SatEvalDataset(data_folder=config.data_folder ,
                                        split_csv=config.evaluate_csv, 
                                        transforms=sat_transforms_val,
                                        )

    sat_dataloader_test = DataLoader(sat_dataset_test,
                                        batch_size=config.batch_size_eval,
                                        num_workers=config.num_workers,
                                        shuffle=False,
                                        pin_memory=True,
                                        )
    
    # Wav Data (Query)
    wav_dataset_test = WavEvalDataset(data_folder=config.data_folder,
                                        split_csv=config.evaluate_csv,
                                        transforms=None,
                                        audio_length_s=config.audio_length_s,
                                        sr_kHz=config.sr_kHz,
                                        processor_wav2vec2=config.model_wav2vec2
                                        )
    
    wav_dataloader_test = WavEvalDataLoader(wav_dataset_test,
                                            batch_size=config.batch_size_eval,
                                            num_workers=config.num_workers,
                                            shuffle=False,
                                            pin_memory=True,
                                            chunking=False
                                            )
    
    print('Reference (Sat) Images Test:', len(sat_dataset_test))
    print('Query (Wav) Wav2Vec Test:', len(wav_dataset_test))

    #-----------------------------------------------------------------------------#
    # Evaluate                                                                    #
    #-----------------------------------------------------------------------------#
    
    print('\n{}[{}]{}'.format(30*'-', 'SoundingEarth', 30*'-'))  

    r1_test = evaluate(config=config,
                        model=model,
                        reference_dataloader=sat_dataloader_test,
                        query_dataloader=wav_dataloader_test, 
                        ranks=[1, 5, 10, 50, 100],
                        step_size=1000,
                        cleanup=True
                        )  

    print("\nNow starting Evaluation with [CHUNKING] enabled:\n")

    wav_dataloader_test.switch_chunking(True)
    
    r1_test_chunked = evaluate(config=config,
                            model=model,
                            reference_dataloader=sat_dataloader_test,
                            query_dataloader=wav_dataloader_test, 
                            ranks=[1, 5, 10, 50, 100],
                            step_size=1000,
                            cleanup=True
                            )  