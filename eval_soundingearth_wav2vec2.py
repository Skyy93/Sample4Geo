import os
import time
import torch
import sys

from dataclasses import dataclass

from torch.utils.data import DataLoader
from spectrum4geo.dataset.soundingearth_wav2vec2 import Wav2Vec2SoundingEarthDatasetEval
from spectrum4geo.transforms import get_transforms_val_sat
from spectrum4geo.utils import Logger
from spectrum4geo.evaluate.soundingearth import evaluate
from spectrum4geo.model import TimmModelWav2Vec2


@dataclass
class Configuration:
    
    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    model_wav2vec2: str = 'facebook/wav2vec2-large-960h'  

    # facebook/wav2vec2-base-960h
    # facebook/wav2vec2-large-960h
    # facebook/wav2vec2-large-960h-lv60-self

    # Override model image size
    img_size: int = 384                 # for satallite images
    sr_kHz = 16
    audio_length_s = 15 #old:15
    
    # Evaluation
    batch_size_eval: int = 128
    verbose: bool = True
    gpu_ids: tuple =  (0,1,2,3,4,5,6,7) # GPU ids for evaluating
    normalize_features: bool = True
    
    # Savepath for model eval logs
    custom_sampling: bool = False        # use custom sampling instead of random (not used during eval.)    
    gps_sample: bool = False             # use gps sampling                      (not used during eval.)   
    sim_sample: bool = False             # use similarity sampling               (not used during eval.)  

    if custom_sampling and gps_sample and sim_sample:
        model_path: str = f'./soundingearth_wav2vec2/testing/Shuffle_On/{audio_length_s}_s' 
    else:
        model_path: str = f'./soundingearth_wav2vec2/testing/Shuffle_Off/{audio_length_s}_s' 

    # Dataset
    data_folder = 'data'        
    evaluate_csv = 'test_df.csv' 

    # Checkpoint to start from
    checkpoint_start = 'soundingearth_wav2vec2/training/audio_length_15_s/Shuffle_Off/convnext_base.fb_in22k_ft_in1k_384/facebook/wav2vec2-large-960h/071505/weights_e40_7.0937.pth'   
  
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
        
    model_path = f'{config.model_path}/{config.model}/{config.model_wav2vec2}/{time.strftime('%H%M%S')}'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

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
    
    # Satalite Satellite Images
    sat_dataset_test = Wav2Vec2SoundingEarthDatasetEval(data_folder=config.data_folder ,
                                      split_csv=config.evaluate_csv, 
                                      query_type = 'sat',
                                      transforms=sat_transforms_val,
                                      audio_length_s=config.audio_length_s,
                                      sr_kHz=config.sr_kHz,
                                      processor_wav2vec2=config.model_wav2vec2
                                      )

    sat_dataloader_test = DataLoader(sat_dataset_test,
                                     batch_size=config.batch_size_eval,
                                     num_workers=config.num_workers,
                                     shuffle=False,
                                     pin_memory=True
                                     )
    
    # wave Data Test
    wave_dataset_test = Wav2Vec2SoundingEarthDatasetEval(data_folder=config.data_folder,
                                                         split_csv=config.evaluate_csv,
                                                         query_type = 'audio',
                                                         transforms=None,
                                                         audio_length_s=config.audio_length_s,
                                                         sr_kHz=config.sr_kHz,
                                                         processor_wav2vec2=config.model_wav2vec2
                                                         )
    
    wave_dataloader_test = DataLoader(wave_dataset_test,
                                      batch_size=config.batch_size_eval,
                                      num_workers=config.num_workers,
                                      shuffle=False,
                                      pin_memory=True,
                                      collate_fn=wave_dataset_test.collate_fn
                                      )
    
    print('Reference (Sat) Images Test:', len(sat_dataset_test))
    print('Reference (Wave) Wav2Vec Test:', len(wave_dataset_test))

    #-----------------------------------------------------------------------------#
    # Evaluate                                                                    #
    #-----------------------------------------------------------------------------#
    
    print('\n{}[{}]{}'.format(30*'-', 'SoundingEarth', 30*'-'))  

    r1_test = evaluate(config=config,
                       model=model,
                       reference_dataloader=sat_dataloader_test,
                       query_dataloader=wave_dataloader_test, 
                       ranks=[1, 5, 10, 50, 100],
                       step_size=1000,
                       cleanup=True)  