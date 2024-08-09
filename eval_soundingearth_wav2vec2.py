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

    checkpoint_start = 'soundingearth_wav2vec2/training/16_kHz/10_audio_length_s_192_batch_size/Shuffle_Off/convnext_base.fb_in22k_ft_in1k_384/173702_0.00075_lr_5e-05_lr_wav2vec2_best/weights_end.pth'   
    sr_kHz, audio_length_s, batch_size, shuffle = extract_checkpoint_info(checkpoint_start)

    # Override model image size
    img_size: int = 384                                        # for satallite images

    # Evaluation
    batch_size_eval: int = "lower set"
    verbose: bool = True
    gpu_ids: tuple =  (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)  # GPU ids for evaluating
    normalize_features: bool = True
    
    # Savepath for model eval logs
    log_path: str = f'./soundingearth_wav2vec2/testing/{sr_kHz}_kHz/{audio_length_s}_audio_length_s_{batch_size}_batch_size/{shuffle}' 

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

    # striding_list = [None,5,7.5,10]
    striding_list = [2.5,7.5,12.5]

    # Iterate through files in the directory list
    #for checkpoint_path in directory_list:
    for stride_s in striding_list:
        config.checkpoint_start = "soundingearth_wav2vec2/training/16_kHz/20_audio_length_s_80_batch_size/Shuffle_Off/convnext_base.fb_in22k_ft_in1k_384/070217_0.000484_lr_3.2274861218395145e-05_lr_wav2vec2/weights_end.pth"
        config.sr_kHz, config.audio_length_s, config.batch_size, config.shuffle = extract_checkpoint_info(config.checkpoint_start)
        config.batch_size_eval = 128*3

        if config.shuffle == True:
            config.log_path =  f'./soundingearth_wav2vec2/testing/{config.sr_kHz}_kHz/{config.audio_length_s}_audio_length_s_{config.batch_size}_batch_size/Shuffle_On'
        else:
            config.log_path =  f'./soundingearth_wav2vec2/testing/{config.sr_kHz}_kHz/{config.audio_length_s}_audio_length_s_{config.batch_size}_batch_size/Shuffle_Off'


        #-----------------------------------------------------------------------------#
        # Model                                                                       #
        #-----------------------------------------------------------------------------#
            
        model_path = f'{config.log_path}/{time.strftime("%H%M%S")}_striding_{stride_s}s-epoch50'

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
                                          stride_s=stride_s,
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

        total_count = sum(wav_dataloader_test.dataset.idx2ccount.values())
        
        if stride_s:
            stride_points = int(stride_s * wav_dataloader_test.dataset.sample_rate)
            rounded_stride_s = stride_points / wav_dataloader_test.dataset.sample_rate
            print(f"Sum of all chunks for stride of {stride_points} = {rounded_stride_s} seconds: ", total_count)
        else:
            print(f"Sum of all chunks: ", total_count)

        
        r1_test_chunked = evaluate(config=config,
                                model=model,
                                reference_dataloader=sat_dataloader_test,
                                query_dataloader=wav_dataloader_test, 
                                ranks=[1, 5, 10, 50, 100],
                                step_size=1000,
                                cleanup=True
                                )  