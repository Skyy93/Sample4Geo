import os
import re
import sys
import time
import torch

from dataclasses import dataclass
from torch.utils.data import DataLoader

from spectrum4geo.utils import Logger
from spectrum4geo.model import TimmModel
from spectrum4geo.evaluate.metrics import evaluate
from spectrum4geo.dataset.evaluation import SatEvalDataset, SpectroEvalDataset
from spectrum4geo.transforms import get_transforms_val_sat, get_transforms_val_spectro 


def extract_checkpoint_info(checkpoint_path):
    # Split the path into parts by '/'
    checkpoint_parts = checkpoint_path.split('/')

    # Find the parts containing relevant information
    mel_kHz_part = next(part for part in checkpoint_parts if part.endswith('kHz')).split('_')
    patch_batch_part = next(part for part in checkpoint_parts if part.endswith('batch_size')).split('_')
    shuffle_part = next(part for part in checkpoint_parts if part.startswith('Shuffle'))

    # Extract values
    n_mels = int(mel_kHz_part[0])
    sr_kHz = int(mel_kHz_part[2].replace('kHz', ''))
    patch_time_steps = int(patch_batch_part[0])
    batch_size = int(patch_batch_part[3].replace('batch_size', ''))
    shuffle = shuffle_part == 'Shuffle_On'

    return n_mels, sr_kHz, patch_time_steps, batch_size, shuffle


@dataclass
class Configuration:
    
    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'

    checkpoint_start = "backup/runs before new shuffeling/soundingearth/training/convnext_base.fb_in22k_ft_in1k_384/old/before_eval_middle/48kHz_128mel/1024_patch_width/205032/weights_e40_9.0373.pth"
    #n_mels, sr_kHz, patch_time_steps, batch_size, shuffle = extract_checkpoint_info(checkpoint_start)
    n_mels, sr_kHz, patch_time_steps, batch_size, shuffle = 128, 48, 1024, 64, "BEFORE_SPLIT!!!"

    # Override model image size
    img_size: int = 384                                             # for satallite images
    
    # Evaluation
    batch_size_eval: int = 64*8 
    verbose: bool = True
    gpu_ids: tuple =  (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)       # GPU ids for evaluating
    normalize_features: bool = True
    
    # Savepath for model eval logs
    log_path: str = f'./soundingearth/testing/{n_mels}_mel_{sr_kHz}_kHz/{patch_time_steps}_patch_width_{batch_size}_batch_size/{shuffle}' 

    # Dataset
    data_folder = 'data'        
    evaluate_csv = 'test_df.csv' 

    # Checkpoint to start from
    # checkpoint_start = 'backup/runs before new shuffeling/soundingearth/training/convnext_base.fb_in22k_ft_in1k_384/old/before_eval_middle/48kHz_128mel/24576_patch_width/103639/weights_e36_10.3929.pth'   
    # 1024 checkpoint_start = 'backup/runs before new shuffeling/soundingearth/training/convnext_base.fb_in22k_ft_in1k_384/old/before_eval_middle/48kHz_128mel/1024_patch_width/205032/weights_end.pth'   

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

config = Configuration() 

if __name__ == '__main__':

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    model_path = f'{config.log_path}/{time.strftime("%H%M%S")}'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    print(f'\nModel: {config.model}')

    print(f'Used .csv file for evaluating: {config.evaluate_csv}')

    model = TimmModel(config.model,
                      pretrained=True,
                      img_size=config.img_size
                      )
                          
    data_config = model.get_config()
    print(data_config)
    mean = data_config['mean']
    std = data_config['std']
    img_size = config.img_size
    
    img_size_sat = (img_size, img_size)
    img_size_spectro = (config.patch_time_steps, config.n_mels)
     
    # load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print('Start from:', config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=False)     

    # Data parallel
    print('\nGPUs available:', torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    # Print information about Spectrogram settings
    print(f'Spectrogram details:\n'
          f'\tSample rate: {config.sr_kHz} kHz\n'
          f'\tn_mels: {config.n_mels}\n'
          f'\tPatch width (time steps): {config.patch_time_steps}'
          )     
           
    # Model to device   
    model = model.to(config.device)

    print('\nImage Size Sat:', img_size_sat)
    print('Image Size Spectro:', img_size_spectro)
    print(f'Mean: {mean}')
    print(f'Std:  {std}\n') 

    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#
    
    # Eval
    sat_transforms_val = get_transforms_val_sat(img_size_sat,
                                                mean=mean,
                                                std=std,
                                                )
    
    spectro_transforms_val = get_transforms_val_spectro(mean=mean,       
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
                                     pin_memory=True
                                     )
    
    # Spectrogram Images (Query)
    spectro_dataset_test = SpectroEvalDataset(data_folder=config.data_folder ,
                                              split_csv=config.evaluate_csv, 
                                              transforms=spectro_transforms_val,
                                              patch_time_steps=config.patch_time_steps,
                                              sr_kHz=config.sr_kHz,
                                              n_mels=config.n_mels,
                                              #stride=config.patch_time_steps//2,
                                              min_frame=None,
                                              chunking=False,
                                              dB_power_weights=False,
                                              use_power_weights=True,
                                              )
    
    spectro_dataloader_test = DataLoader(spectro_dataset_test,
                                         batch_size=config.batch_size_eval,
                                         num_workers=config.num_workers,
                                         shuffle=False,
                                         pin_memory=True,
                                         )
    
    print('Reference (Sat) Images Test:', len(sat_dataset_test))
    print('Reference (Spectro) Wav2Vec Test:', len(spectro_dataset_test))

    #-----------------------------------------------------------------------------#
    # Evaluate                                                                    #
    #-----------------------------------------------------------------------------#
    
    print('\n{}[{}]{}'.format(30*'-', 'SoundingEarth', 30*'-'))  

    r1_test = evaluate(config=config,
                       model=model,
                       reference_dataloader=sat_dataloader_test,
                       query_dataloader=spectro_dataloader_test, 
                       ranks=[1, 5, 10, 50, 100],
                       step_size=1000,
                       cleanup=True
                       )  


    print("\nNow starting Evaluation with [CHUNKING] enabled:\n")

    query_dataloader.dataset.switch_chunking(True)
    
    r1_test_chunked = evaluate(config=config,
                               model=model,
                               reference_dataloader=sat_dataloader_test,
                               query_dataloader=spectro_dataloader_test, 
                               ranks=[1, 5, 10, 50, 100],
                               step_size=1000,
                               cleanup=True
                               )  
