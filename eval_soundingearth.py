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
    print(checkpoint_path)
    # Split the path into parts by '/'
    checkpoint_parts = checkpoint_path.split('/')

    # Find the parts containing relevant information
    mel_kHz_part = next(part for part in checkpoint_parts if part.endswith('kHz')).split('_')
    patch_batch_part = next(part for part in checkpoint_parts if part.endswith('batch_size')).split('_')
    shuffle_part = next(part for part in checkpoint_parts if part.startswith('Shuffle'))

    # Extract values
    n_mels = int(mel_kHz_part[0])
    sr_kHz = int(mel_kHz_part[2].replace('kHz', ''))  # Changed to float to handle values like 22.05
    patch_time_steps = int(patch_batch_part[0])
    batch_size = int(patch_batch_part[3].replace('batch_size', ''))
    shuffle = shuffle_part == 'Shuffle_On'

    return n_mels, sr_kHz, patch_time_steps, batch_size, shuffle


@dataclass
class Configuration:
    
    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
            # soundingearth/training/128_mel_48_kHz/4096_patch_width_256_batch_size/Shuffle_On/convnext_base.fb_in22k_ft_in1k_384/b-similarity/175030_0.001_neighbour_select32_b_similarity_start_epoch_25/weights_end.pth
    checkpoint_start = 'soundingearth/training/128_mel_48_kHz/4096_patch_width_288_batch_size/Shuffle_Off/convnext_base.fb_in22k_ft_in1k_384/082509_0.00106_lr_256_sat/weights_end.pth'   
    n_mels, sr_kHz, patch_time_steps, batch_size, shuffle = extract_checkpoint_info(checkpoint_start)
    #n_mels, sr_kHz, patch_time_steps, batch_size, shuffle = 128, 48, 1024, 64, "BEFORE_SPLIT!!!"

    # Override model image size
    img_size: int = 256                                             # for satallite images
    
    # Evaluation
    batch_size_eval: int = 64*8 
    verbose: bool = True
    gpu_ids: tuple =  (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)       # GPU ids for evaluating
    normalize_features: bool = True
    
    # Savepath for model eval logs
    if shuffle == True:
        log_path: str = f'./soundingearth/testing/{n_mels}_mel_{sr_kHz}_kHz/{patch_time_steps}_patch_width_{batch_size}_batch_size/Shuffle_On'
    else:
        log_path: str = f'./soundingearth/testing/{n_mels}_mel_{sr_kHz}_kHz/{patch_time_steps}_patch_width_{batch_size}_batch_size/Shuffle_Off'

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

    # directory_list = [
    #     "soundingearth/training/128_mel_48_kHz/4096_patch_width_256_batch_size/Shuffle_On/convnext_base.fb_in22k_ft_in1k_384/121755_0.001_lr/weights_end.pth",
    # ]

#    striding_list = [None,1024,2048,3072]
#
#    # Iterate through files in the directory list
#    #for checkpoint_path in directory_list:
#    for stride in striding_list:
#        # also test "soundingearth/training/128_mel_48_kHz/4096_patch_width_256_batch_size/Shuffle_Off/convnext_base.fb_in22k_ft_in1k_384/020318_0.001_lr_best/weights_e36_16.4472.pth"
#        config.checkpoint_start = "soundingearth/training/128_mel_48_kHz/4096_patch_width_256_batch_size/Shuffle_Off/convnext_base.fb_in22k_ft_in1k_384/020318_0.001_lr_best/weights_end.pth"
#        config.n_mels, config.sr_kHz, config.patch_time_steps, config.batch_size, config.shuffle = extract_checkpoint_info(config.checkpoint_start)
#        config.batch_size_eval = config.batch_size
#
#        if config.shuffle == True:
#            config.log_path = f'./soundingearth/testing/{config.n_mels}_mel_{config.sr_kHz}_kHz/{config.patch_time_steps}_patch_width_{config.batch_size}_batch_size/Shuffle_On'
#        else:
#            config.log_path = f'./soundingearth/testing/{config.n_mels}_mel_{config.sr_kHz}_kHz/{config.patch_time_steps}_patch_width_{config.batch_size}_batch_size/Shuffle_Off'
#
    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    model_path = f'{config.log_path}/{time.strftime("%H%M%S")}_512_image'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    logger.set_log_file(os.path.join(model_path, 'log.txt'))

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
                                              #stride=stride,
                                              min_frame=None,
                                              chunking=False,
                                              dB_power_weights=False,
                                              use_power_weights=False,
                                              )
    
    spectro_dataloader_test = DataLoader(spectro_dataset_test,
                                         batch_size=config.batch_size_eval,
                                         num_workers=config.num_workers,
                                         shuffle=False,
                                         pin_memory=True,
                                         )
    
    print('Reference (Sat) Images Test:', len(sat_dataset_test))
    print('Query (Spectro) Images Test:', len(spectro_dataset_test))

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

    spectro_dataloader_test.dataset.switch_chunking(True)
    
    r1_test_chunked = evaluate(config=config,
                               model=model,
                               reference_dataloader=sat_dataloader_test,
                               query_dataloader=spectro_dataloader_test, 
                               ranks=[1, 5, 10, 50, 100],
                               step_size=1000,
                               cleanup=True
                               )  
    