import os
import sys
import time
import torch
import pickle
import shutil

from math import sqrt
from copy import deepcopy
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

from spectrum4geo.loss import InfoNCE
from spectrum4geo.trainer import train
from spectrum4geo.model import TimmModel
from spectrum4geo.utils import setup_system, Logger
from spectrum4geo.evaluate.metrics import evaluate, calc_sim
from spectrum4geo.dataset.evaluation import SatEvalDataset, SpectroEvalDataset
from spectrum4geo.dataset.training import SatSpectroTrainDataset, SpectroSimDataset
from spectrum4geo.transforms import get_transforms_val_sat, get_transforms_val_spectro 
from spectrum4geo.transforms import get_transforms_train_sat, get_transforms_train_spectro 

# -> 1024 fac_2 = 544/384

fac_2 = 544/384

@dataclass
class Configuration:
    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k_384' 
    
    # Override model image size
    img_size: int = 256                          # for satallite images
    patch_time_steps: int = 4096                 # Image size for spectrograms (Width)
    n_mels: int = 128                            # image size for spectrograms (Height)
    sr_kHz: float = 48

    # Training 
    batch_size: int = 288                        # keep in mind real_batch_size = batch_size * (1 + spectrogramm.shape/img.shape)
    mixed_precision: bool = True
    seed = 42
    epochs: int = 40
    verbose: bool = True
    gpu_ids: tuple =  (0,1,2,3,4,5,6,7,
                       8,9,10,11,12,13,14,15)   # GPU ids for training
    
    # Eval
    batch_size_eval: int = 16
    eval_every_n_epoch: int = 4                  # eval every n Epoch
    normalize_features: bool = True
    
    # Similarity Sampling
    custom_sampling: bool = False                 # use custom sampling instead of random     -> To False for not using shuffle function of dataloader!
    gps_sample: bool = False                       # use gps sampling                          -> To False for not using shuffle function of dataloader!
    min_bound_km: int = 0
    sim_sample: bool = False                      # use similarity sampling                   -> To False for not using shuffle function of dataloader!
    neighbour_select: int = 32  #32 #128          # max selection size from pool
    neighbour_range: int = 256                   # pool size for selection
    gps_dict_path: str = 'data/gps_dict_256.pkl' # path to pre-computed distances
 
    # Optimizer 
    clip_grad = 100.                            # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False            # Gradient Checkpointing
    
    # Loss
    label_smoothing: float = 0.1
    
    # Learning Rate
    lr: float = 0.001 * sqrt(288/256)           # 1 * 10^-4 for ViT | 1 * 10^-1 for CNN
    scheduler: str = 'cosine'                   # 'polynomial' | 'cosine' | 'constant' | None
    warmup_epochs: int = 1
    lr_end: float = 0.00001                     #  only for 'polynomial'
    
    # Dataset
    data_folder = 'data'  
    train_csv = 'train_df.csv'
    evaluate_csv = 'validate_df.csv'
    
    # Augment Images
    prob_rotate: float = 0.75                   # rotates the sat image 
    prob_flip: float = 0.5                      # flipping the sat image 
    
    # Savepath for model checkpoints
    if custom_sampling:
        model_path: str = f'./soundingearth/training/{n_mels}_mel_{sr_kHz}_kHz/{patch_time_steps}_patch_width_{batch_size}_batch_size/Shuffle_On' 
    else:
        model_path: str = f'./soundingearth/training/{n_mels}_mel_{sr_kHz}_kHz/{patch_time_steps}_patch_width_{batch_size}_batch_size/Shuffle_Off' 

    # Eval before training
    zero_shot: bool = False 
    
    # Checkpoint to start from
    start_epoch = 0 #13                             # 5 if you choose an checkpoint of epoch 4   / 8 if 7 / 13 if 12
    checkpoint_start = None #'soundingearth/training/128_mel_48_kHz/4096_patch_width_256_batch_size/Shuffle_On/convnext_base.fb_in22k_ft_in1k_384/110128_0.001_neighbour_select32_b_similarity_start_epoch_17_new_fabian_version/weights_e34_14.1678.pth' # None # 'soundingearth/training/128_mel_48_kHz/4096_patch_width_256_batch_size/Shuffle_On/convnext_base.fb_in22k_ft_in1k_384/nice ones/After_episode_7_110137_0.001_lr_loaded_checkpoint_start_epoch_8_neighbour_select8/weights_e12_9.4714.pth'  
  
    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else len(gpu_ids)//2
    
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

    model_path = f'{config.model_path}/{config.model}/{time.strftime("%H%M%S")}_{config.lr:.3g}_lr_256_sat'

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

    # Print inforamtions of used batch sizes
    print(f'Model training batch size: {config.batch_size}')
    print(f'Model evaluation batch size: {config.batch_size_eval}')

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
    train_dataset = SatSpectroTrainDataset(data_folder=config.data_folder,
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

    # Satellite Images (Reference)
    sat_dataset_test = SatEvalDataset(data_folder=config.data_folder,
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
                                              )
    
    spectro_dataloader_test = DataLoader(spectro_dataset_test,
                                         batch_size=config.batch_size_eval,
                                         num_workers=config.num_workers,
                                         shuffle=False,
                                         pin_memory=True
                                         )
    
    print('\nReference (Sat) Images Test:', len(sat_dataset_test))
    print('Query (Spectro) Images Test:', len(spectro_dataset_test))
    
    #-----------------------------------------------------------------------------#
    # GPS Sample                                                                  #
    #-----------------------------------------------------------------------------#

    if config.gps_sample:
        with open(config.gps_dict_path, 'rb') as f:
            gps_sim_dict = pickle.load(f)
    else:
        gps_sim_dict = None

    #-----------------------------------------------------------------------------#
    # Sim Sample                                                                  #
    #-----------------------------------------------------------------------------#
    
    if config.sim_sample:
    
        # Spectrogram Images (Query): train-split for simsampling
        spectro_dataset_train = SpectroSimDataset(data_folder=config.data_folder ,
                                                  split_csv=config.train_csv,
                                                  transforms=spectro_transforms_val,
                                                  patch_time_steps=config.patch_time_steps,
                                                  sr_kHz=config.sr_kHz,
                                                  n_mels=config.n_mels,
                                                  )

        #spectro_dataset_train = SpectroEvalDataset(data_folder=config.data_folder ,
        #                                           split_csv=config.train_csv, 
        #                                           transforms=spectro_transforms_val,
        #                                           patch_time_steps=config.patch_time_steps,
        #                                           sr_kHz=config.sr_kHz,
        #                                           n_mels=config.n_mels,
        #                                           #stride=config.patch_time_steps//2,
        #                                           stride=None,
        #                                           min_frame=None,
        #                                           chunking=True,
        #                                           dB_power_weights=False,
        #                                           use_power_weights=False,
        #                                           )
            
        spectro_dataloader_train = DataLoader(spectro_dataset_train,
                                              batch_size=config.batch_size_eval,
                                              num_workers=config.num_workers,
                                              shuffle=False,
                                              pin_memory=True
                                              )
        
        sat_dataset_train = SatEvalDataset(data_folder=config.data_folder,
                                           split_csv=config.train_csv,
                                           transforms=sat_transforms_val,
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
                                          min_bound_km=config.min_bound_km,
                                          ranks=[1, 5, 10, 50, 100],
                                          step_size=1000,
                                          cleanup=True
                                          )
    else:
        sim_dict = deepcopy(gps_sim_dict)
                
    #-----------------------------------------------------------------------------#
    # Shuffle                                                                     #
    #-----------------------------------------------------------------------------#     
           
    if config.custom_sampling and config.gps_sample:
        train_dataloader.dataset.shuffle(sim_dict,
                                         neighbour_select=config.neighbour_select,
                                         neighbour_range=config.neighbour_range
                                         )
            
    #-----------------------------------------------------------------------------#
    # Train                                                                       #
    #-----------------------------------------------------------------------------#

    best_score = 0

    # Setting up seed for first epoch
    train_dataloader.dataset.set_random_seed(config.start_epoch)
    if config.sim_sample:
        spectro_dataloader_train.dataset.set_random_seed(config.start_epoch)

    if config.start_epoch > 1:
        print(f'Initializing scheduler until epoch {config.start_epoch-1}')
        train_steps_per_epoch = len(train_dataloader)
        for epoch in range(1, config.start_epoch):
            for _ in range(train_steps_per_epoch):
                scheduler.step()  # Advances the learning rate scheduler.

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {config.start_epoch-1}: Learning Rate = {current_lr:.6f}')

    if config.custom_sampling and config.sim_sample:
        print(f'neighbour_select (before starting Training) = {config.neighbour_select}')
        if config.sim_sample and config.start_epoch > 4:
            r1_train, sim_dict = calc_sim(config=config,
                                          model=model,
                                          reference_dataloader=sat_dataloader_train,
                                          query_dataloader=spectro_dataloader_train, 
                                          min_bound_km=config.min_bound_km,
                                          ranks=[1, 5, 10, 50, 100],
                                          step_size=1000,
                                          cleanup=True
                                          )         
        else:
            sim_dict = None 

        train_dataloader.dataset.shuffle(sim_dict,
                                         neighbour_select=config.neighbour_select,
                                         neighbour_range=config.neighbour_range
                                         )
            
    
    for epoch in range(config.start_epoch, config.epochs+1):

        if config.custom_sampling:
            neighbour_select_epoch = config.neighbour_select
            print(f'neighbour_select (current) = {neighbour_select_epoch}')
        
        print('\n{}[Epoch: {}]{}'.format(30*'-', epoch, 30*'-'))

        train_loss = train(config,
                           model,
                           dataloader=train_dataloader,
                           loss_function=loss_function,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           scaler=scaler
                           )
        
        print(f'Epoch: {epoch}, Train Loss = {train_loss:.3f}, Lr = {optimizer.param_groups[0]["lr"]:.6f}')
        
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

            if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                torch.save(model.module.state_dict(), f'{model_path}/weights_e{epoch}_{r1_test:.4f}.pth')
            else:
                torch.save(model.state_dict(), f'{model_path}/weights_e{epoch}_{r1_test:.4f}.pth')
                
            #if r1_test > best_score:
            #
            #    best_score = r1_test
            #
            #    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
            #        torch.save(model.module.state_dict(), f'{model_path}/weights_e{epoch}_{r1_test:.4f}.pth')
            #    else:
            #        torch.save(model.state_dict(), f'{model_path}/weights_e{epoch}_{r1_test:.4f}.pth')

        # Setting up seed for next epoch
        train_dataloader.dataset.set_random_seed(epoch+1)
        if config.sim_sample:
            spectro_dataloader_train.dataset.set_random_seed(epoch+1)

        if config.custom_sampling:
            if config.sim_sample and epoch > 4 and epoch < config.epochs:
                r1_train, sim_dict = calc_sim(config=config,
                                              model=model,
                                              reference_dataloader=sat_dataloader_train,
                                              query_dataloader=spectro_dataloader_train, 
                                              min_bound_km=config.min_bound_km,
                                              ranks=[1, 5, 10, 50, 100],
                                              step_size=1000,
                                              cleanup=True
                                              )         
            elif config.gps_sample:
                sim_dict = gps_sim_dict
            else:
                sim_dict = None 

            train_dataloader.dataset.shuffle(sim_dict,
                                             neighbour_select=neighbour_select_epoch,
                                             neighbour_range=config.neighbour_range
                                             )
            
                
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), f'{model_path}/weights_end.pth')
    else:
        torch.save(model.state_dict(), f'{model_path}/weights_end.pth')            