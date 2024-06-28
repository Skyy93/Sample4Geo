import os
import time
import shutil
import sys
import torch
import pickle

from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

from spectrum4geo.dataset.soundingearth import SoundingEarthDatasetEval, SoundingEarthDatasetTrain
from spectrum4geo.transforms import get_transforms_train_sat, get_transforms_train_spectro 
from spectrum4geo.transforms import get_transforms_val_sat, get_transforms_val_spectro 
from spectrum4geo.utils import setup_system, Logger
from spectrum4geo.trainer import train
from spectrum4geo.evaluate.soundingearth import evaluate, calc_sim
from spectrum4geo.loss import InfoNCE
from spectrum4geo.model import TimmModel


@dataclass
class Configuration:
    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k_384' 
    
    # Override model image size
    img_size: int = 384                 # for satallite images
    patch_time_steps: int = 256         # Image size for spectrograms (Width)
    n_mels: int = 128                   # image size for spectrograms (Height)
    sr_kHz: float = 48
    
    # Training 
    mixed_precision: bool = True
    seed = 42
    epochs: int = 40
    batch_size: int = 48                # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = True
    gpu_ids: tuple =  (0,1,2,3,4,5,6,7)         # GPU ids for training
    #gpu_ids: tuple = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)   # GPU ids for training
    
    # Similarity Sampling
    custom_sampling: bool = True                # use custom sampling instead of random     -> To False for not using shuffle function of dataloader!
    gps_sample: bool = True                     # use gps sampling                          -> To False for not using shuffle function of dataloader!
    sim_sample: bool = True                     # use similarity sampling                   -> To False for not using shuffle function of dataloader!
    neighbour_select: int = 64                  # max selection size from pool
    neighbour_range: int = 128                  # pool size for selection
    gps_dict_path: str = 'data/gps_dict.pkl'    # path to pre-computed distances
 
    # Eval
    batch_size_eval: int = 64
    eval_every_n_epoch: int = 4         # eval every n Epoch
    normalize_features: bool = True

    # Optimizer 
    clip_grad = 100.                    # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False    # Gradient Checkpointing
    
    # Loss
    label_smoothing: float = 0.1
    
    # Learning Rate
    lr: float = 0.001                   # 1 * 10^-4 for ViT | 1 * 10^-1 for CNN
    scheduler: str = 'cosine'           # 'polynomial' | 'cosine' | 'constant' | None
    warmup_epochs: int = 1
    lr_end: float = 0.0001              #  only for 'polynomial'
    
    # Dataset
    data_folder = 'data'  
    train_csv = 'train_df.csv'
    evaluate_csv = 'validate_df.csv'
    
    # Augment Images
    prob_rotate: float = 0.75          # rotates the sat image 
    prob_flip: float = 0.5             # flipping the sat image 
    
    # Savepath for model checkpoints
    if custom_sampling and gps_sample and sim_sample:
        model_path: str = f'./soundingearth/Shuffle_On/{n_mels}_mel_{sr_kHz}_kHz' 
    else:
        model_path: str = f'./soundingearth/Shuffle_Off/{n_mels}_mel_{sr_kHz}_kHz' 

    # Eval before training
    zero_shot: bool = False 
    
    # Checkpoint to start from
    checkpoint_start = None   
  
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

config = Configuration() 

if __name__ == '__main__':

    model_path = f'{config.model_path}/{config.model}/{time.strftime('%H%M%S')}'

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(__file__, os.path.join(model_path, 'train.py'))

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic
                 )
    
    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    print(f'\nModel: {config.model}\n')

    model = TimmModel(config.model,
                      pretrained=True,
                      img_size=config.img_size
                      )     # no image size needed, this wis more important for an implementation of an ViT (base_model)
                          
    data_config = model.get_config()
    print(data_config)
    mean = data_config['mean']
    std = data_config['std']
    img_size = config.img_size
    
    img_size_sat = (img_size, img_size)
    img_size_spectro = (config.patch_time_steps, config.n_mels)
    
    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)
     
    # Load pretrained Checkpoint    
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
          f'\tPatch width (time steps): {config.patch_time_steps}')     
           
    # Model to device   
    model = model.to(config.device)

    print('\nImage Size Sat:', img_size_sat)
    print('Image Size Spectro:', img_size_spectro)
    print(f'Mean: {mean}')
    print(f'Std:  {std}\n') 

    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # Transforms
    sat_transforms_train = get_transforms_train_sat(img_size_sat,
                                                    mean=mean,
                                                    std=std,
                                                    )
    
    spectro_transforms_train = get_transforms_train_spectro(img_size_spectro,
                                                            mean=mean,       
                                                            std=std,       
                                                            )                                                               
                                                                  
    # Train
    train_dataset = SoundingEarthDatasetTrain(data_folder=config.data_folder,
                                              split_csv=config.train_csv,
                                              transforms_sat_image=sat_transforms_train,
                                              transforms_spectrogram=spectro_transforms_train,
                                              patch_time_steps=config.patch_time_steps,
                                              sr_kHz=config.sr_kHz,
                                              n_mels=config.n_mels,
                                              prob_flip=config.prob_flip,
                                              prob_rotate=config.prob_rotate,
                                              shuffle_batch_size=config.batch_size
                                              )
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True
                                  )
    
    # Eval
    sat_transforms_val = get_transforms_val_sat(img_size_sat,
                                                mean=mean,
                                                std=std
                                                )
    
    spectro_transforms_val = get_transforms_val_spectro(mean=mean,       
                                                        std=std
                                                        )        

    # Reference Satellite Images
    sat_dataset_test = SoundingEarthDatasetEval(data_folder=config.data_folder,
                                                split_csv=config.evaluate_csv,
                                                query_type = 'sat',
                                                transforms=sat_transforms_val,
                                                patch_time_steps=config.patch_time_steps,
                                                sr_kHz=config.sr_kHz,
                                                n_mels=config.n_mels
                                                )
    
    sat_dataloader_test = DataLoader(sat_dataset_test,
                                     batch_size=config.batch_size_eval,
                                     num_workers=config.num_workers,
                                     shuffle=False,
                                     pin_memory=True
                                     )
    
    # Reference Spectrogram Images
    spectro_dataset_test = SoundingEarthDatasetEval(data_folder=config.data_folder ,
                                                    split_csv=config.evaluate_csv,
                                                    query_type = 'spectro',
                                                    transforms=spectro_transforms_val,
                                                    patch_time_steps=config.patch_time_steps,
                                                    sr_kHz=config.sr_kHz,
                                                    n_mels=config.n_mels,
                                                    )
    
    spectro_dataloader_test = DataLoader(spectro_dataset_test,
                                         batch_size=config.batch_size_eval,
                                         num_workers=config.num_workers,
                                         shuffle=False,
                                         pin_memory=True
                                         )
    
    print('\nReference (Sat) Images Test:', len(sat_dataset_test))
    print('Reference (Spectro) Images Test:', len(spectro_dataset_test))
    
    #-----------------------------------------------------------------------------#
    # GPS Sample                                                                  #
    #-----------------------------------------------------------------------------#

    if config.gps_sample:
        with open(config.gps_dict_path, 'rb') as f:
            sim_dict = pickle.load(f)
    else:
        sim_dict = None

    #-----------------------------------------------------------------------------#
    # Sim Sample                                                                  #
    #-----------------------------------------------------------------------------#
    
    if config.sim_sample:
    
        # Query Spectrogram Images Train for simsampling
        spectro_dataset_train = SoundingEarthDatasetEval(data_folder=config.data_folder,
                                                         split_csv=config.train_csv,
                                                         query_type = 'spectro',
                                                         transforms=spectro_transforms_val,
                                                         patch_time_steps=config.patch_time_steps,
                                                         sr_kHz=config.sr_kHz,
                                                         n_mels=config.n_mels
                                                         )
            
        spectro_dataloader_train = DataLoader(spectro_dataset_train,
                                              batch_size=config.batch_size_eval,
                                              num_workers=config.num_workers,
                                              shuffle=False,
                                              pin_memory=True
                                              )
        
        sat_dataset_train = SoundingEarthDatasetEval(data_folder=config.data_folder,
                                                     split_csv=config.train_csv,
                                                     query_type = 'sat',
                                                     transforms=sat_transforms_val,
                                                     patch_time_steps=config.patch_time_steps,
                                                     sr_kHz=config.sr_kHz,
                                                     n_mels=config.n_mels,
                                                     )
        
        sat_dataloader_train = DataLoader(sat_dataset_train,
                                          batch_size=config.batch_size_eval,
                                          num_workers=config.num_workers,
                                          shuffle=False,
                                          pin_memory=True
                                          )

        print('\nReference Images Train:', len(sat_dataset_train))
        print('Query Images Train:', len(spectro_dataset_train))        

    #-----------------------------------------------------------------------------#
    # Loss                                                                        #
    #-----------------------------------------------------------------------------#

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loss_function = InfoNCE(loss_function=loss_fn,
                            device=config.device
                            )

    if config.mixed_precision:
        scaler = GradScaler(init_scale=2.**10)
    else:
        scaler = None
        
    #-----------------------------------------------------------------------------#
    # optimizer                                                                   #
    #-----------------------------------------------------------------------------#

    if config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias']
        optimizer_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01,
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    #-----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    #-----------------------------------------------------------------------------#

    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs
       
    if config.scheduler == 'polynomial':
        print(f'\nScheduler: polynomial - max LR: {config.lr} - end LR: {config.lr_end}')  
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end = config.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps
                                                              )
        
    elif config.scheduler == 'cosine':
        print(f'\nScheduler: cosine - max LR: {config.lr}')   
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps
                                                    )
        
    elif config.scheduler == 'constant':
        print(f'\nScheduler: constant - max LR: {config.lr}')   
        scheduler =  get_constant_schedule_with_warmup(optimizer,
                                                       num_warmup_steps=warmup_steps
                                                       )
           
    else:
        scheduler = None
        
    print(f'Warmup Epochs: {str(config.warmup_epochs).ljust(2)} - Warmup Steps: {warmup_steps}')
    print(f'Train Epochs:  {config.epochs} - Train Steps:  {train_steps}')

    #-----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    #-----------------------------------------------------------------------------#

    if config.zero_shot:
        print('\n{}[{}]{}'.format(30*'-', 'Zero Shot', 30*'-'))  

        r1_test = evaluate(config=config,
                           model=model,
                           reference_dataloader=sat_dataloader_test,
                           query_dataloader=spectro_dataloader_test, 
                           ranks=[1, 5, 10, 50, 100],
                           step_size=1000,
                           cleanup=True
                           )
        
        if config.sim_sample:
            r1_train, sim_dict = calc_sim(config=config,
                                          model=model,
                                          reference_dataloader=sat_dataloader_train,
                                          query_dataloader=spectro_dataloader_train, 
                                          ranks=[1, 5, 10, 50, 100],
                                          step_size=1000,
                                          cleanup=True
                                          )
                
    #-----------------------------------------------------------------------------#
    # Shuffle                                                                     #
    #-----------------------------------------------------------------------------#     
           
    if config.custom_sampling:
        train_dataloader.dataset.shuffle(sim_dict,
                                         neighbour_select=config.neighbour_select,
                                         neighbour_range=config.neighbour_range
                                         )
            
    #-----------------------------------------------------------------------------#
    # Train                                                                       #
    #-----------------------------------------------------------------------------#

    start_epoch = 0   
    best_score = 0
    
    for epoch in range(1, config.epochs+1):
        
        print('\n{}[Epoch: {}]{}'.format(30*'-', epoch, 30*'-'))

        train_loss = train(config,
                           model,
                           dataloader=train_dataloader,
                           loss_function=loss_function,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           scaler=scaler
                           )
        
        print(f'Epoch: {epoch}, Train Loss = {train_loss:.3f}, Lr = {optimizer.param_groups[0]['lr']:.6f}')
        
        # evaluate
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:
        
            print('\n{}[{}]{}'.format(30*'-', 'Evaluate', 30*'-'))
        
            r1_test = evaluate(config=config,
                               model=model,
                               reference_dataloader=sat_dataloader_test,
                               query_dataloader=spectro_dataloader_test, 
                               ranks=[1, 5, 10, 50, 100],
                               step_size=1000,
                               cleanup=True
                               )
            
            if config.sim_sample:
                r1_train, sim_dict = calc_sim(config=config,
                                               model=model,
                                               reference_dataloader=sat_dataloader_train,
                                               query_dataloader=spectro_dataloader_train, 
                                               ranks=[1, 5, 10, 50, 100],
                                               step_size=1000,
                                               cleanup=True
                                               )
                
            if r1_test > best_score:

                best_score = r1_test

                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(), f'{model_path}/weights_e{epoch}_{r1_test:.4f}.pth')
                else:
                    torch.save(model.state_dict(), f'{model_path}/weights_e{epoch}_{r1_test:.4f}.pth')
                
        if config.custom_sampling:
            train_dataloader.dataset.shuffle(sim_dict,
                                             neighbour_select=config.neighbour_select,
                                             neighbour_range=config.neighbour_range
                                             )
                
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), f'{model_path}/weights_end.pth')
    else:
        torch.save(model.state_dict(), f'{model_path}/weights_end.pth')            