import os
import time
import math
import shutil
import sys
import torch
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from reident.dataset.university import U1652DatasetEval, U1652DatasetTrain, get_transforms
from reident.utils import setup_system, Logger
from reident.trainer import train
from reident.evaluate.university import evaluate
from reident.loss import ClipLoss
from reident.model import TimmModel
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

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
    img_size: int = 512
    
    # Training 
    mixed_precision: bool = True
    custom_sampling: bool = True         # use custom sampling instead of random
    seed = 1
    epochs: int = 1
    batch_size: int = 128                # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = True
    gpu_ids: tuple = (0,)           # GPU ids for training
    
    # Eval
    batch_size_eval: int = 128
    eval_every_n_epoch: int = 1          # eval every n Epoch
    normalize_features: bool = True
    eval_gallery_n: int = -1             # -1 for all or int

    # Optimizer 
    clip_grad = 100.                     # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False     # Gradient Checkpointing
    
    # Loss
    label_smoothing: float = 0.1
    
    # Learning Rate
    lr: float = 0.001                    # 1 * 10^-4 for ViT | 1 * 10^-1 for CNN
    scheduler: str = "cosine"           # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 0.1
    lr_end: float = 0.0001               #  only for "polynomial"
    gradient_accumulation: int = 1
    
    # Dataset
    dataset: str = 'U1652-D2S'           # 'U1652-D2S' | 'U1652-S2D'
    data_folder: str = "./data/U1652"
    single_sample: bool = False
    
    # Augment Images
    prob_flip: float = 0.5              # flipping the sat image and drone image simultaneously
    
    # Savepath for model checkpoints
    model_path: str = "./university_e40_eval4_384_aug_final"
    
    # Eval before training
    zero_shot: bool = True
    
    # Checkpoint to start from
    checkpoint_start = 'pretrained/university_e40_eval4_384_aug_final/convnext_base.fb_in22k_ft_in1k_384/085046/weights_e1_0.9515.pth'
  
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

if config.dataset == 'U1652-D2S':
    config.query_folder_train = './data/U1652/train/satellite'
    config.gallery_folder_train = './data/U1652/train/drone'   
    config.query_folder_test = './data/U1652/test/query_drone' 
    config.gallery_folder_test = './data/U1652/test/gallery_satellite'    
elif config.dataset == 'U1652-S2D':
    config.query_folder_train = './data/U1652/train/satellite'
    config.gallery_folder_train = './data/U1652/train/drone'    
    config.query_folder_test = './data/U1652/test/query_satellite'
    config.gallery_folder_test = './data/U1652/test/gallery_drone'

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
    img_size = (config.img_size, config.img_size)
    

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

    print("\nImage Size Query:", img_size)
    print("Image Size Ground:", img_size)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std)) 


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # Transforms
    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std)
                                                                                                                                 
    # Train
    train_dataset = U1652DatasetTrain(query_folder=config.query_folder_train,
                                      gallery_folder=config.gallery_folder_train,
                                      transforms_query=train_sat_transforms,
                                      transforms_gallery=train_drone_transforms,
                                      prob_flip=config.prob_flip,
                                      shuffle_batch_size=config.batch_size,
                                      single_sample=config.single_sample
                                      )
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True)
    
    # Reference Satellite Images
    query_dataset_test = U1652DatasetEval(data_folder=config.query_folder_test,
                                               mode="query",
                                               transforms=val_transforms,
                                               )
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    # Query Ground Images Test
    gallery_dataset_test = U1652DatasetEval(data_folder=config.gallery_folder_test,
                                               mode="gallery",
                                               transforms=val_transforms,
                                               sample_ids=query_dataset_test.get_sample_ids(),
                                               gallery_n=config.eval_gallery_n,
                                               )
    
    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    print("Query Images Test:", len(query_dataset_test))
    print("Gallery Images Test:", len(gallery_dataset_test))
   
    #-----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    #-----------------------------------------------------------------------------#
    if config.zero_shot:
        print("\n{}[{}]{}".format(30*"-", "University-1652", 30*"-"))  

        r1_test = evaluate(config=config,
                           model=model,
                           query_loader=query_dataloader_test,
                           gallery_loader=gallery_dataloader_test, 
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)
 
