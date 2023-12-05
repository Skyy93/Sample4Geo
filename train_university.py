import os
import time
import math
import shutil
import sys
import torch
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.utils.data.distributed import DistributedSampler

from sample4geo.dataset.university import U1652DatasetEval, U1652DatasetTrain, get_transforms
from sample4geo.utils import setup_system, Logger, print_dist
from sample4geo.trainer import train
from sample4geo.evaluate.university import evaluate
from sample4geo.loss import InfoNCE
from sample4geo.model import TimmModel


@dataclass
class Configuration:
    
    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    
    # Override model image size
    img_size: int = 384
    
    # Training 
    mixed_precision: bool = True
    custom_sampling: bool = True         # use custom sampling instead of random
    seed = 1
    epochs: int = 1
    batch_size: int = 32                # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = True
    gpu_ids: tuple = (0,1,2,3)           # GPU ids for training
    ddp: bool = False    
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
    
    # Dataset
    dataset: str = 'U1652-D2S'           # 'U1652-D2S' | 'U1652-S2D'
    data_folder: str = "./data/U1652"
    
    # Augment Images
    prob_flip: float = 0.5              # flipping the sat image and drone image simultaneously
    
    # Savepath for model checkpoints
    model_path: str = "./university"
    
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

if 'LOCAL_RANK' in os.environ:
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])
    config.ddp = True
else:
    print("*" * 30)
    print("*")
    print("*   WARNING: You are not using DDP, training is slower and results might be worse")
    print("*")
    print("*" * 30)
    LOCAL_RANK = -1
    WORLD_SIZE = 1

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


if __name__ == '__main__':

    if config.ddp:
        torch.distributed.init_process_group(backend='nccl', rank=WORLD_RANK, world_size=WORLD_SIZE)

    model_path = "{}/{}/{}".format(config.model_path,
                                   config.model,
                                   time.strftime("%H%M%S"))
    if LOCAL_RANK < 1:
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        shutil.copyfile(os.path.basename(__file__), "{}/train.py".format(model_path))

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    print_dist("\nModel: {}".format(config.model), LOCAL_RANK)


    model = TimmModel(config.model,
                          pretrained=True,
                          img_size=config.img_size)
                          
    data_config = model.get_config()
    print_dist(data_config, LOCAL_RANK)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (config.img_size, config.img_size)
    
    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)
    
    # Load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print_dist("Start from:", config.checkpoint_start, LOCAL_RANK)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=False)     


    # Model to device
    if config.ddp:
        device = torch.device("cuda:{}".format(LOCAL_RANK))
        config.device = device
        model = model.to(config.device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK])
    else:
        model = model.to(config.device)

    print_dist("\nImage Size Sat: " + str(img_size), LOCAL_RANK)
    print_dist("Mean: {}".format(mean), LOCAL_RANK)
    print_dist("Std:  {}\n".format(std), LOCAL_RANK) 



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
                                      shuffle_batch_size=config.batch_size * WORLD_SIZE,
                                      )
    if config.ddp:
        sampler = DistributedSampler(dataset=train_dataset, shuffle=not config.custom_sampling)
    else: 
        sampler = None
 
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=False,
                                  sampler=sampler,
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
    
    print_dist("Query Images Test: " + str(len(query_dataset_test)), LOCAL_RANK)
    print_dist("gallery Images Test: " + str(len(gallery_dataset_test)), LOCAL_RANK)
 
    #-----------------------------------------------------------------------------#
    # Loss                                                                        #
    #-----------------------------------------------------------------------------#

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loss_function = InfoNCE(loss_function=loss_fn,
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
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
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
       
    if config.scheduler == "polynomial":
        print_dist("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end), LOCAL_RANK)  
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end = config.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)
        
    elif config.scheduler == "cosine":
        print_dist("\nScheduler: cosine - max LR: {}".format(config.lr), LOCAL_RANK)   
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)
        
    elif config.scheduler == "constant":
        print_dist("\nScheduler: constant - max LR: {}".format(config.lr), LOCAL_RANK)   
        scheduler =  get_constant_schedule_with_warmup(optimizer,
                                                       num_warmup_steps=warmup_steps)
           
    else:
        scheduler = None
        
    print_dist("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps), LOCAL_RANK)
    print_dist("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps), LOCAL_RANK)
        
        
    #-----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    #-----------------------------------------------------------------------------#
    if config.zero_shot and LOCAL_RANK < 1:
        print("\n{}[{}]{}".format(30*"-", "Zero Shot", 30*"-"))  
        torch.cuda.synchronize(torch.cuda.current_device())

        r1_test = evaluate(config=config,
                           model=model,
                           query_loader=query_dataloader_test,
                           gallery_loader=gallery_dataloader_test, 
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)
                
    #-----------------------------------------------------------------------------#
    # Shuffle                                                                     #
    #-----------------------------------------------------------------------------#            
    if config.custom_sampling:
        train_dataloader.dataset.shuffle(LOCAL_RANK)
            
    #-----------------------------------------------------------------------------#
    # Train                                                                       #
    #-----------------------------------------------------------------------------#
    start_epoch = 0   
    best_score = 0
    

    for epoch in range(1, config.epochs+1):
        
        print_dist("\n{}[Epoch: {}]{}".format(30*"-", epoch, 30*"-"), LOCAL_RANK)
        

        train_loss = train(config,
                           model,
                           dataloader=train_dataloader,
                           loss_function=loss_function,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           scaler=scaler,
                           rank=LOCAL_RANK)
        
        print_dist("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(epoch,
                                                                   train_loss,
                                                                   optimizer.param_groups[0]['lr']), LOCAL_RANK)
        
        # evaluate
        if ((epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs) and LOCAL_RANK < 1:
            torch.cuda.synchronize(torch.cuda.current_device())
       
            print("\n{}[{}]{}".format(30*"-", "Evaluate", 30*"-"))
        
            r1_test = evaluate(config=config,
                               model=model,
                               query_loader=query_dataloader_test,
                               gallery_loader=gallery_dataloader_test, 
                               ranks=[1, 5, 10],
                               step_size=1000,
                               cleanup=True)
                
            if r1_test > best_score:

                best_score = r1_test

                if config.ddp and LOCAL_RANK==0:

                    torch.save(model.module.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                else:
                    torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                

        if config.custom_sampling:
            train_dataloader.dataset.shuffle(LOCAL_RANK)
                
    if config.ddp and LOCAL_RANK == 0:
        torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    else:
        torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))            
