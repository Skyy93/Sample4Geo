import os
import sys
import time
import torch
import pickle
import shutil

from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

from spectrum4geo.loss import InfoNCE
from spectrum4geo.model import TimmModelWav2Vec2
from spectrum4geo.utils import setup_system, Logger
from spectrum4geo.trainer import train_wav2vec2 as train
from spectrum4geo.transforms import get_transforms_val_sat
from spectrum4geo.evaluate.soundingearth import evaluate, calc_sim
from spectrum4geo.dataset.training import SatSpectroTrainDataset, SatWavTrainDataset
from spectrum4geo.transforms import get_transforms_train_sat, get_transforms_train_wave 
from spectrum4geo.dataset.evaluation import SatEvalDataset, WavEvalDataset, WavEvalDataLoader


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

    # Training 
    batch_size: int = 64                                        # keep in mind real_batch_size = 2 * batch_size
    mixed_precision: bool = True
    seed = 42
    epochs: int = 50
    verbose: bool = True
    gpu_ids: tuple =  (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)   # GPU ids for training

    # Eval
    batch_size_eval: int = 64
    eval_every_n_epoch: int = 4                 # eval every n Epoch
    normalize_features: bool = True

    # Similarity Sampling
    custom_sampling: bool = False               # use custom sampling instead of random     -> To False for not using shuffle function of dataloader!
    gps_sample: bool = False                    # use gps sampling                          -> To False for not using shuffle function of dataloader!
    sim_sample: bool = False                    # use similarity sampling                   -> To False for not using shuffle function of dataloader!
    neighbour_select: int = 64                  # max selection size from pool
    neighbour_range: int = 128                  # pool size for selection
    gps_dict_path: str = 'data/gps_dict.pkl'    # path to pre-computed distances
 
    # Optimizer 
    clip_grad = 100.                    # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False    # Gradient Checkpointing
    
    # Loss
    label_smoothing: float = 0.1
    
    # Learning Rate
    lr_base: float = 0.001              # 1 * 10^-4 for ViT | 1 * 10^-1 for CNN
    lr_wav2vec2: float = 0.00001
    scheduler_base: str = 'cosine'      # 'polynomial' | 'cosine' | 'constant' | None
    scheduler_wav2vec2: str = 'cosine'
    lr_base_end: float = 0.0001         #  only for 'polynomial'
    lr_wav2vec2_end: float = 0.000001   #  only for 'polynomial'
    warmup_epochs: int = 5

    # Dataset
    data_folder = 'data'     
    train_csv = 'train_df.csv'
    evaluate_csv = 'validate_df.csv'

    # Augment Images
    prob_rotate: float = 0.75           # rotates the sat image 
    prob_flip: float = 0.5              # flipping the sat image 
    
    # Savepath for model checkpoints
    if custom_sampling and gps_sample and sim_sample:
        model_path: str = f'./soundingearth_wav2vec2/training/{sr_kHz}_kHz/{audio_length_s}_audio_length_s_{batch_size}_batch_size/Shuffle_On' 
    else:
        model_path: str = f'./soundingearth_wav2vec2/training/{sr_kHz}_kHz/{audio_length_s}_audio_length_s_{batch_size}_batch_size/Shuffle_Off' 

    # Eval before training
    zero_shot: bool = False 
    
    # Checkpoint to start from
    checkpoint_start = None   
  
    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else len(gpu_ids)
    
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

    model_path = f'{config.model_path}/{config.model}/{time.strftime("%H%M%S")}'

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
        
    print(f'\nModel 1: {config.model} (base/img)')
    print(f'Model 2: {config.model_wav2vec2} (wav2vec2/audio)\n')

    model = TimmModelWav2Vec2(config.model,
                              config.model_wav2vec2,
                              pretrained=True,
                              img_size=config.img_size  # no image size needed, this wis more important for an implementation of an ViT (base_model)
                              ) 
                          
    data_config = model.get_config()
    print(data_config)
    mean = data_config['mean']
    std = data_config['std']
    img_size = config.img_size
    
    img_size_sat = (img_size, img_size)
    
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

    # Transforms
    sat_transforms_train = get_transforms_train_sat(img_size_sat,
                                                    mean=mean,
                                                    std=std,
                                                    )
    
    wave_transforms_train = get_transforms_train_wave()
                                                                                                                                  
    # Train
    train_dataset = SatWavTrainDataset(data_folder=config.data_folder,
                                       split_csv=config.train_csv,
                                       transforms_sat_image=sat_transforms_train,
                                       transforms_wave=wave_transforms_train,
                                       audio_length_s=config.audio_length_s,
                                       sr_kHz=config.sr_kHz,
                                       processor_wav2vec2=config.model_wav2vec2,
                                       prob_flip=config.prob_flip,
                                       prob_rotate=config.prob_rotate,
                                       shuffle_batch_size=config.batch_size
                                       )
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True,
                                  collate_fn=train_dataset.collate_fn
                                  )
    
    # Eval
    sat_transforms_val = get_transforms_val_sat(img_size_sat,
                                                mean=mean,
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
    
    wav_dataloader_test = DataLoader(wave_dataset_test,
                                     batch_size=config.batch_size_eval,
                                     num_workers=config.num_workers,
                                     shuffle=False,
                                     pin_memory=True,
                                     chunking=False
                                     )
    
    print('\nReference (Sat) Images Test:', len(sat_dataset_test))
    print('Reference (Wave) Wav2Vec Test:', len(wave_dataset_test))
    
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
    
        # Wav Data (Query): train-split for simsampling
        wav_dataset_train = WavEvalDataset(data_folder=config.data_folder,
                                           split_csv=config.train_csv,
                                           transforms=None,
                                           audio_length_s=config.audio_length_s,
                                           sr_kHz=config.sr_kHz,
                                           processor_wav2vec2=config.model_wav2vec2
                                           )

        wav_dataloader_train = WavEvalDataLoader(wav_dataset_train,
                                                 batch_size=config.batch_size_eval,
                                                 num_workers=config.num_workers,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 chunking=False
                                                 )
        
        sat_dataset_train = SatEvalDataset(data_folder=config.data_folder,
                                           split_csv=config.train_csv,
                                           transforms=sat_transforms_val,
                                           )
        
        sat_dataloader_train = DataLoader(sat_dataset_train,
                                          batch_size=config.batch_size_eval,
                                          num_workers=config.num_workers,
                                          shuffle=False,
                                          pin_memory=True,
                                          )

        print('\nReference Images Train:', len(sat_dataset_train))
        print('Query Audio (Wave) Train:', len(wav_dataset_train))        

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

    decay_parameters = []

    if config.decay_exclue_bias:  
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias']
        decay_parameters = [
            { 
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01,
            },
            { 
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]

    optimizer_base = torch.optim.AdamW([
                                       {'params': model.module.base_model.parameters(), 'lr': config.lr_base},
                                       {'params': model.module.logit_scale, 'lr': config.lr_base},
                                       ] + decay_parameters 
                                       )

    optimizer_wav2vec2 = torch.optim.AdamW([
                                            {'params': model.module.wav2vec2_model.parameters(), 'lr': config.lr_wav2vec2},
                                            {'params': model.module.projection.parameters(), 'lr': config.lr_wav2vec2},]
                                            )

    #-----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    #-----------------------------------------------------------------------------#

    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs

    def create_scheduler(optimizer, scheduler_type, train_steps, warmup_steps, lr_end=None, power=None):
        if scheduler_type == 'polynomial':
            return get_polynomial_decay_schedule_with_warmup(optimizer,
                                                             num_training_steps=train_steps,
                                                             lr_end=lr_end,
                                                             power=power,
                                                             num_warmup_steps=warmup_steps
                                                             )

        elif scheduler_type == 'cosine':
            return get_cosine_schedule_with_warmup(optimizer,
                                                   num_training_steps=train_steps,
                                                   num_warmup_steps=warmup_steps
                                                   )

        elif scheduler_type == 'constant':
            return get_constant_schedule_with_warmup(optimizer,
                                                     num_warmup_steps=warmup_steps
                                                     )

        else:
            return None

    # Separate schedulers for base_model and wav2vec2_model
    scheduler_base = create_scheduler(optimizer_base, config.scheduler_base, train_steps, warmup_steps, lr_end=config.lr_base_end, power=1.5)
    scheduler_wav2vec2 = create_scheduler(optimizer_wav2vec2, config.scheduler_wav2vec2, train_steps, warmup_steps, lr_end=config.lr_wav2vec2_end, power=1.5)
    
    print(f'\nScheduler (img/base): {config.scheduler_base}')
    print(f'Lr (base/img): {config.lr_base}')
    if config.scheduler_base == 'polynomial':
        print(f'Lr_end (base/img): {config.lr_base_end}')

    print(f'Scheduler (audio/wav2vec2): {config.scheduler_wav2vec2}')
    print(f'Lr (wav2vec2/audio): {config.lr_wav2vec2}')
    if config.scheduler_wav2vec2 == 'polynomial':
        print(f'Lr_end (wav2vec2/audio): {config.lr_wav2vec2_end}')

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
                           query_dataloader=wav_dataloader_test, 
                           ranks=[1, 5, 10, 50, 100],
                           step_size=1000,
                           cleanup=True
                           )
        
        if config.sim_sample:
            r1_train, sim_dict = calc_sim(config=config,
                                          model=model,
                                          reference_dataloader=sat_dataloader_train,
                                          query_dataloader=wav_dataloader_train, 
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
                           optimizer_list=[optimizer_base, optimizer_wav2vec2],
                           scheduler_list=[scheduler_base, scheduler_wav2vec2],
                           lr_monitor=[('lr_base'), ('lr_wav2vec2')],
                           scaler=scaler
                           )

        print('Epoch: {}, Train Loss = {:.3f}, Lr: base/img = {:.6f},  Lr: wav2vec2/audio = {:.6f}'.format(
              epoch,
              train_loss,
              optimizer_base.param_groups[0]['lr'],
              optimizer_wav2vec2.param_groups[0]['lr'])
              )
        
        # evaluate
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:
        
            print('\n{}[{}]{}'.format(30*'-', 'Evaluate', 30*'-'))
        
            r1_test = evaluate(config=config,
                               model=model,
                               reference_dataloader=sat_dataloader_test,
                               query_dataloader=wav_dataloader_test, 
                               ranks=[1, 5, 10, 50, 100],
                               step_size=1000,
                               cleanup=True
                               )
            
            if config.sim_sample:
                r1_train, sim_dict = calc_sim(config=config,
                                              model=model,
                                              reference_dataloader=sat_dataloader_train,
                                              query_dataloader=wav_dataloader_train, 
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