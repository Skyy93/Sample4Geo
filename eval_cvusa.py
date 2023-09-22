import os
import time
import math
import shutil
import sys
import torch
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from reident.dataset.cvusa import CVUSADatasetEval, CVUSADatasetTrain
from reident.transforms import get_transforms_train, get_transforms_val
from reident.utils import setup_system, Logger
from reident.trainer import train
from reident.evaluate.cvusa_and_cvact import evaluate, calc_sim
from reident.loss import ClipLoss
from reident.model import TimmModel
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup
import pickle


@dataclass
class TrainingConfiguration:
    '''
    Describes configuration of the training process
    '''
    #--------------------------------------------------------------------------
    # Timm Models:
    #--------------------------------------------------------------------------    
    # 'convnext_base.fb_in22k_ft_in1k_384'   
    # 'convnextv2_base.fcmae_ft_in22k_in1k_384'
    # 'vit_base_patch16_384.augreg_in21k_ft_in1k'
    # 'vit_base_patch16_clip_224.openai'    
    #--------------------------------------------------------------------------
    
    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    
    # Override model image size
    img_size: int = 384
    
    # Training 
    mixed_precision: bool = True
    seed = 1
    epochs: int = 40
    batch_size: int = 128        # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = True
    gpu_ids: tuple = (0,)   # GPU ids for training
    
    
    # Similarity Sampling
    custom_sampling: bool = True   # use custom sampling instead of random
    gps_sample: bool = True        # use gps sampling
    sim_sample: bool = True        # use similarity sampling
    neighbour_select: int = 64     # max selection size from pool
    neighbour_range: int = 128     # pool size for selection
    gps_dict_path: str = "./data/CVUSA/gps_dict.pkl"   # path to pre-computed distances
 
    # Eval
    batch_size_eval: int = 128
    eval_every_n_epoch: int = 4      # eval every n Epoch
    normalize_features: bool = True

    # Optimizer 
    clip_grad = 100.                 # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False # Gradient Checkpointing
    
    # Loss
    label_smoothing: float = 0.1
    
    # Learning Rate
    lr: float = 0.001                  # 1 * 10^-4 for ViT | 1 * 10^-1 for CNN
    scheduler: str = "cosine"          # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 1
    lr_end: float = 0.0001             #  only for "polynomial"
    gradient_accumulation: int = 1
    
    # Dataset
    data_folder = "./data/CVUSA"     
    
    # Augment Images
    prob_rotate: float = 0.75           # rotates the sat image and ground images simultaneously
    prob_flip: float = 0.5              # flipping the sat image and ground images simultaneously
    
    # Savepath for model checkpoints
    model_path: str = "./cvusa_e40_eval4_sat_384_ground_140x768_aug_final"
    
    # Eval before training
    zero_shot: bool = True 
    
    # Checkpoint to start from
    checkpoint_start = 'pretrained/cvusa_e40-4_sat_384_ground_140x768_aug_final/convnext_base.fb_in22k_ft_in1k_384/203234/weights_e40_98.5142.pth'   
  
    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    # for better performance
    cudnn_benchmark: bool = True
    
    # make cudnn deterministic
    cudnn_deterministic: bool = False

#-----------------------------------------------------------------------------#
# Train Config                                                                #
#-----------------------------------------------------------------------------#

config = TrainingConfiguration() 

#%%

if __name__ == '__main__':

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    print("\nModel: {}".format(config.model))


    model = TimmModel(config.model,
                      pretrained=True,
                      img_size=config.img_size)
                          
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = config.img_size
    
    image_size_sat = (img_size, img_size)
    
    new_width = config.img_size * 2    
    new_hight = round((224 / 1232) * new_width)
    img_size_ground = (new_hight, new_width)
     
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

    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std)) 


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # Transforms
    sat_transforms_train, ground_transforms_train = get_transforms_train(image_size_sat,
                                                                   img_size_ground,
                                                                   mean=mean,
                                                                   std=std,
                                                                   )
                                                                   
                                                                   
    # Train
    train_dataset = CVUSADatasetTrain(data_folder=config.data_folder ,
                                      transforms_query=ground_transforms_train,
                                      transforms_reference=sat_transforms_train,
                                      prob_flip=config.prob_flip,
                                      prob_rotate=config.prob_rotate,
                                      shuffle_batch_size=config.batch_size
                                      )
    
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True)
    
    
    # Eval
    sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                                               img_size_ground,
                                                               mean=mean,
                                                               std=std,
                                                               )


    # Reference Satellite Images
    reference_dataset_test = CVUSADatasetEval(data_folder=config.data_folder ,
                                              split="test",
                                              img_type="reference",
                                              transforms=sat_transforms_val,
                                              )
    
    reference_dataloader_test = DataLoader(reference_dataset_test,
                                           batch_size=config.batch_size_eval,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=True)
    
    
    
    # Query Ground Images Test
    query_dataset_test = CVUSADatasetEval(data_folder=config.data_folder ,
                                          split="test",
                                          img_type="query",    
                                          transforms=ground_transforms_val,
                                          )
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    
    print("Reference Images Test:", len(reference_dataset_test))
    print("Query Images Test:", len(query_dataset_test))
    
    
        
    #-----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    #-----------------------------------------------------------------------------#
    if config.zero_shot:
        print("\n{}[{}]{}".format(30*"-", "CVUSA", 30*"-"))  

      
        r1_test = evaluate(config=config,
                           model=model,
                           reference_dataloader=reference_dataloader_test,
                           query_dataloader=query_dataloader_test, 
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)
