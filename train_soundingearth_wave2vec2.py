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

from spectrum4geo.dataset.soundingearth_wav2vec2 import Wav2Vec2SoundingEarthDatasetTrain, Wav2Vec2SoundingEarthDatasetEval
from spectrum4geo.transforms import get_transforms_train_sat, get_transforms_train_wave 
from spectrum4geo.transforms import get_transforms_val_sat

from spectrum4geo.utils import setup_system, Logger
from spectrum4geo.trainer import train_wave2vec2 as train
from spectrum4geo.evaluate.soundingearth import evaluate, calc_sim
from spectrum4geo.loss import InfoNCE
from spectrum4geo.model import TimmModelWav2Vec2

@dataclass
class Configuration:
    
    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k_384' 
    model_wav2vec: str = 'facebook/wav2vec2-base-960h'
    
    # Override model image size
    img_size: int = 384         # for satallite images
    sr_kHz = 16
    audio_length_s = 15

    # Training 
    mixed_precision: bool = True
    seed = 42
    epochs: int = 40
    batch_size: int = 48         # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = True
    gpu_ids: tuple = (0,1,2,3,4,5,6,7)   # GPU ids for training
    
    
    # Similarity Sampling
    custom_sampling: bool = False   # use custom sampling instead of random     -> To False for not using shuffle function of dataloader!
    gps_sample: bool = False        # use gps sampling                          -> To False for not using shuffle function of dataloader!
    sim_sample: bool = False        # use similarity sampling                   -> To False for not using shuffle function of dataloader!
    neighbour_select: int = 64     # max selection size from pool
    neighbour_range: int = 128     # pool size for selection
    gps_dict_path: str = "data/gps_dict.pkl"   # path to pre-computed distances
 
    # Eval
    batch_size_eval: int = 64
    eval_every_n_epoch: int = 4        # eval every n Epoch
    normalize_features: bool = True

    # Optimizer 
    clip_grad = 100.                   # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False   # Gradient Checkpointing
    
    # Loss
    label_smoothing: float = 0.1
    
    # Learning Rate
    lr_base: float = 0.001                  # 1 * 10^-4 for ViT | 1 * 10^-1 for CNN
    lr_wave2vec2: float = 0.0001
    scheduler_base: str = "cosine"          # "polynomial" | "cosine" | "constant" | None
    scheduler_wave2vec2: str = "cosine"
    lr_base_end: float = 0.0001             #  only for "polynomial"
    lr_wave2vec2_end: float = 0.0001        #  only for "polynomial"
    warmup_epochs: int = 1

    # Dataset
    data_folder = "data"     
    
    # Augment Images
    prob_rotate: float = 0.75          # rotates the sat image 
    prob_flip: float = 0.5             # flipping the sat image 
    
    # Savepath for model checkpoints
    model_path: str = "./soundingearth_wav2vec"
    model_sr_kHz: float = 16
    model_sr: float = model_sr_kHz * 1e3
    
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

    model_path = "{}/{}/{}".format(config.model_path,
                                   config.model,
                                   time.strftime("%H%M%S"))

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(__file__, os.path.join(model_path, "train.py"))

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)
    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    print("\nModel: {}".format(config.model))


    model = TimmModelWav2Vec2(config.model,
                      config.model_wav2vec,
                      pretrained=True,
                      img_size=config.img_size)     # no image size needed, this wis more important for an implementation of an ViT
                          
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = config.img_size
    
    img_size_sat = (img_size, img_size)
    
    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)
     
    # Load pretrained Checkpoint    
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

    print("\nImage Size Sat:", img_size_sat)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std)) 

    print(f"\nAudio segment length: {config.audio_length_s} s")
    print(f"Lr (base/img): {config.lr_base}")
    print(f"Lr (wave2vec2/audio): {config.lr_wave2vec2}")
    if config.scheduler_base == "polynomial":
        print(f"Lr_end (base/img): {config.lr_base_end}")
    if config.scheduler_wave2vec2 == "polynomial":
        print(f"Lr_end (wave2vec2/audio): {config.lr_wave2vec2_end}")

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
    train_dataset = Wav2Vec2SoundingEarthDatasetTrain(data_folder=config.data_folder ,
                                          split_csv='train_df.csv',
                                          transforms_sat_image=sat_transforms_train,
                                          transforms_wave=wave_transforms_train,
                                          audio_length_s=config.audio_length_s,
                                          sr_kHz=config.sr_kHz,
                                          processor_wav2vec2=config.model_wav2vec,
                                          prob_flip=config.prob_flip,
                                          prob_rotate=config.prob_rotate,
                                          shuffle_batch_size=config.batch_size,
                                                                                    )
    
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True,
                                  collate_fn=train_dataset.collate_fn)
    
    
    # Eval
    sat_transforms_val = get_transforms_val_sat(img_size_sat,
                                            mean=mean,
                                            std=std,
                                            )
          


    # Reference Satellite Images
    sat_dataset_test = Wav2Vec2SoundingEarthDatasetEval(data_folder=config.data_folder ,
                                        split_csv='test_df.csv',
                                        query_type = "sat",
                                        transforms=sat_transforms_val,
                                        audio_length_s=config.audio_length_s,
                                        sr_kHz=config.sr_kHz,
                                        processor_wav2vec2=config.model_wav2vec,
                                        )
    
    sat_dataloader_test = DataLoader(sat_dataset_test,
                                           batch_size=config.batch_size_eval,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=True,
                                           collate_fn=sat_dataset_test.collate_fn)
    
    
    # Reference wave Data
    wave_dataset_test = Wav2Vec2SoundingEarthDatasetEval(data_folder=config.data_folder ,
                                        split_csv='test_df.csv',
                                        query_type = "audio",
                                        transforms=None,
                                        audio_length_s=config.audio_length_s,
                                        sr_kHz=config.sr_kHz,
                                        processor_wav2vec2=config.model_wav2vec,
                                        )
        
    wave_dataloader_test = DataLoader(wave_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True,
                                       collate_fn=wave_dataset_test.collate_fn)
    
    
    print("Reference (Sat) Images Test:", len(sat_dataset_test))
    print("Reference (Wave) Wav2Vec Test:", len(wave_dataset_test))
    
    
    #-----------------------------------------------------------------------------#
    # GPS Sample                                                                  #
    #-----------------------------------------------------------------------------#
    if config.gps_sample:
        with open(config.gps_dict_path, "rb") as f:
            sim_dict = pickle.load(f)
    else:
        sim_dict = None

    #-----------------------------------------------------------------------------#
    # Sim Sample                                                                  #
    #-----------------------------------------------------------------------------#
    
    if config.sim_sample:
    
        # Query audiogram Images Train for simsampling
        wave_dataset_train = Wav2Vec2SoundingEarthDatasetEval(data_folder=config.data_folder ,
                                    split_csv='test_df.csv',
                                    query_type = "audio",
                                    transforms=None,
                                    audio_length_s=config.audio_length_s,
                                    sr_kHz=config.sr_kHz,
                                    processor_wav2vec2=config.model_wav2vec,
                                    )
            
        wave_dataloader_train = DataLoader(wave_dataset_train,
                                            batch_size=config.batch_size_eval,
                                            num_workers=config.num_workers,
                                            shuffle=False,
                                            pin_memory=True,
                                            collate_fn=wave_dataset_train.collate_fn)
        
        
        sat_dataset_train = Wav2Vec2SoundingEarthDatasetEval(data_folder=config.data_folder ,
                                    split_csv='test_df.csv',
                                    query_type = "sat",
                                    transforms=sat_transforms_val,
                                    audio_length_s=config.audio_length_s,
                                    sr_kHz=config.sr_kHz,
                                    processor_wav2vec2=config.model_wav2vec,
                                    )
        
        sat_dataloader_train = DataLoader(sat_dataset_train,
                                                batch_size=config.batch_size_eval,
                                                num_workers=config.num_workers,
                                                shuffle=False,
                                                pin_memory=True,
                                                collate_fn=sat_dataset_train.collate_fn)


        print("\nReference Images Train:", len(sat_dataset_train))
        print("Query Audio (Wave) Train:", len(wave_dataset_train))        

    
    #-----------------------------------------------------------------------------#
    # Loss                                                                        #
    #-----------------------------------------------------------------------------#

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loss_function = InfoNCE(loss_function=loss_fn,
                            device=config.device,
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
        no_decay = ["bias", "LayerNorm.bias"]
        decay_parameters = [
            { 
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            { 
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

    optimizer_base = torch.optim.AdamW([
            {'params': model.module.base_model.parameters(), 'lr': config.lr_base},
            {'params': model.module.logit_scale, 'lr': config.lr_base},
            ] + decay_parameters )

    optimizer_wave2vec2 = torch.optim.AdamW([
            {'params': model.module.wav2vec2_model.parameters(), 'lr': config.lr_wave2vec2},
            {'params': model.module.projection.parameters(), 'lr': config.lr_wave2vec2},
            ])

    #-----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    #-----------------------------------------------------------------------------#

    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs

    def create_scheduler(optimizer, scheduler_type, train_steps, warmup_steps, lr_end=None, power=None):
        if scheduler_type == "polynomial":
            return get_polynomial_decay_schedule_with_warmup(optimizer,
                                num_training_steps=train_steps,
                                lr_end=lr_end,
                                power=power,
                                num_warmup_steps=warmup_steps)

        elif scheduler_type == "cosine":
            return get_cosine_schedule_with_warmup(optimizer,
                                num_training_steps=train_steps,
                                num_warmup_steps=warmup_steps)

        elif scheduler_type == "constant":
            return get_constant_schedule_with_warmup(optimizer,
                                num_warmup_steps=warmup_steps)

        else:
            return None

    # Separate schedulers for base_model and wav2vec2_model
    scheduler_base = create_scheduler(optimizer_base, config.scheduler_base, train_steps, warmup_steps, lr_end=config.lr_base_end, power=1.5)
    scheduler_wave2vec2 = create_scheduler(optimizer_wave2vec2, config.scheduler_wave2vec2, train_steps, warmup_steps, lr_end=config.lr_wave2vec2_end, power=1.5)
    
    print(f"\n(img/base) Scheduler: {config.scheduler_base}")
    print(f"(audio/wave2vec2) Scheduler: {config.scheduler_wave2vec2}")
    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))
        
    #-----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    #-----------------------------------------------------------------------------#
    if config.zero_shot:
        print("\n{}[{}]{}".format(30*"-", "Zero Shot", 30*"-"))  

      
        r1_test, median_rank_test, mean_dist_test, roc_auc_test = evaluate(config=config,
                           model=model,
                           reference_dataloader=sat_dataloader_test,
                           query_dataloader=wave_dataloader_test, 
                           ranks=[1, 5, 10, 50, 100],
                           step_size=1000,
                           cleanup=True)
        
        if config.sim_sample:
            r1_train, median_rank_train, mean_dist_train, roc_auc_train, sim_dict = calc_sim(config=config,
                                                    model=model,
                                                    reference_dataloader=sat_dataloader_train,
                                                    query_dataloader=wave_dataloader_train, 
                                                    ranks=[1, 5, 10, 50, 100],
                                                    step_size=1000,
                                                    cleanup=True)
                
    #-----------------------------------------------------------------------------#
    # Shuffle                                                                     #
    #-----------------------------------------------------------------------------#            
    if config.custom_sampling:
        train_dataloader.dataset.shuffle(sim_dict,
                                         neighbour_select=config.neighbour_select,
                                         neighbour_range=config.neighbour_range)
            
    #-----------------------------------------------------------------------------#
    # Train                                                                       #
    #-----------------------------------------------------------------------------#
    start_epoch = 0   
    best_score = 0
    

    for epoch in range(1, config.epochs+1):
        
        print("\n{}[Epoch: {}]{}".format(30*"-", epoch, 30*"-"))
        
        train_loss = train(config,
                           model,
                           dataloader=train_dataloader,
                           loss_function=loss_function,
                           optimizer_list=[optimizer_base, optimizer_wave2vec2],
                           scheduler_list=[scheduler_base, scheduler_wave2vec2],
                           lr_monitor=[("lr_base"), ("lr_wave2vec2")],
                           scaler=scaler)

        print("Epoch: {}, Train Loss = {:.3f}, Lr: base/img = {:.6f},  Lr: wave2vec2/audio = {:.6f}".format(epoch,
                                                                   train_loss,
                                                                   optimizer_base.param_groups[0]['lr'],
                                                                   optimizer_wave2vec2.param_groups[0]['lr']))
        
        # evaluate
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:
        
            print("\n{}[{}]{}".format(30*"-", "Evaluate", 30*"-"))
        
            r1_test, median_rank_test, mean_dist_test, roc_auc_test = evaluate(config=config,
                               model=model,
                               reference_dataloader=sat_dataloader_test,
                               query_dataloader=wave_dataloader_test, 
                               ranks=[1, 5, 10, 50, 100],
                               step_size=1000,
                               cleanup=True)
            
            if config.sim_sample:
                r1_train, median_rank_train, mean_dist_train, roc_auc_train, sim_dict = calc_sim(config=config,
                                                            model=model,
                                                            reference_dataloader=sat_dataloader_train,
                                                            query_dataloader=wave_dataloader_train, 
                                                            ranks=[1, 5, 10, 50, 100],
                                                            step_size=1000,
                                                            cleanup=True)
                
            if r1_test > best_score:

                best_score = r1_test

                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                else:
                    torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                

        if config.custom_sampling:
            train_dataloader.dataset.shuffle(sim_dict,
                                             neighbour_select=config.neighbour_select,
                                             neighbour_range=config.neighbour_range)
                
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    else:
        torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))            
