import os
import time
import torch
import shutil
import sys

from dataclasses import dataclass

from torch.utils.data import DataLoader
from spectrum4geo.dataset.soundingearth import SoundingEarthDatasetEval
from spectrum4geo.transforms import get_transforms_val_sat, get_transforms_val_spectro 
from spectrum4geo.utils import Logger
from spectrum4geo.evaluate.soundingearth import evaluate
from spectrum4geo.model import TimmModel


@dataclass
class Configuration:
    
    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    
    # Override model image size
    img_size: int = 384                  # for satallite images
    patch_time_steps: int = 1024*4       # Image size for spectrograms (Width)
    n_mels: int = 128                    # image size for spectrograms (Height)
    sr_kHz: float = 48
    
    # Evaluation
    batch_size_eval: int = 128
    verbose: bool = True
    gpu_ids: tuple =  (0,1,2,3,4,5,6,7)          # GPU ids for evaluating
    normalize_features: bool = True
    
    # Savepath for model eval logs
    model_path: str = "./soundingearth/testing"

    # Dataset
    data_folder = "data"        
    split_csv = 'test_df.csv' #TODO: change back to test.csv

    # Checkpoint to start from
    checkpoint_start = 'soundingearth/training/convnext_base.fb_in22k_ft_in1k_384/145835/weights_end.pth'   
  
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
        
    model_path = "{}/{}/{}".format(config.model_path,
                                   config.model,
                                   time.strftime("%H%M%S"))

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    print("\nModel: {}".format(config.model))

    print(f"Used .csv file for evaluating: {config.split_csv}")

    model = TimmModel(config.model,
                      pretrained=True,
                      img_size=config.img_size)
                          
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = config.img_size
    
    img_size_sat = (img_size, img_size)
    img_size_spectro = (config.patch_time_steps, config.n_mels)
     
    # load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=False)     

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            
    # Model to device   
    model = model.to(config.device)

    print(f"\nSpectrogram details:\n"
          f"\tSample rate: {config.sr_kHz} kHz\n"
          f"\tn_mels: {config.n_mels}\n"
          f"\tPatch width (time steps): {config.patch_time_steps}")     

    print("\nImage Size Sat:", img_size_sat)
    print("Image Size Spectro:", img_size_spectro)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std)) 

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

    # Satalite Satellite Images
    sat_dataset_test = SoundingEarthDatasetEval(data_folder=config.data_folder ,
                                      split_csv=config.split_csv, 
                                      query_type = "sat",
                                      transforms=sat_transforms_val,
                                      patch_time_steps=config.patch_time_steps,
                                      sr_kHz=config.sr_kHz,
                                      n_mels=config.n_mels,
                                      )

    sat_dataloader_test = DataLoader(sat_dataset_test,
                                     batch_size=config.batch_size_eval,
                                     num_workers=config.num_workers,
                                     shuffle=False,
                                     pin_memory=True)
    
    # Spectrogram Ground Images Test
    spectro_dataset_test = SoundingEarthDatasetEval(data_folder=config.data_folder ,
                                      split_csv=config.split_csv, 
                                      query_type="spectro",
                                      transforms=spectro_transforms_val,
                                      patch_time_steps=config.patch_time_steps,
                                      sr_kHz=config.sr_kHz,
                                      n_mels=config.n_mels,
                                      )
    
    spectro_dataloader_test = DataLoader(spectro_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    print("Satalite Images Test:", len(sat_dataset_test))
    print("Spectrogram Images Test:", len(spectro_dataset_test))

    #-----------------------------------------------------------------------------#
    # Evaluate                                                                    #
    #-----------------------------------------------------------------------------#
    
    print("\n{}[{}]{}".format(30*"-", "SoundingEarth", 30*"-"))  

    r1_test, median_rank_test, mean_dist_test, roc_auc_test = evaluate(config=config,
                       model=model,
                       reference_dataloader=sat_dataloader_test,
                       query_dataloader=spectro_dataloader_test, 
                       ranks=[1, 5, 10, 50, 100],
                       step_size=1000,
                       cleanup=True)  
