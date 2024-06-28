import time
import torch

import torch.nn.functional as F

from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast


def train(train_config, model, dataloader, loss_function, optimizer, scheduler, scaler=None):
    # set model train mode
    model.train()
    
    losses = AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    # for loop over one epoch
    for sat_img, reference, ids in bar: 
        if scaler:
            with autocast():
                # data (batches) to device   
                sat_img = sat_img.to(train_config.device)
                features1, features2 = model(sat_img, reference.to(train_config.device))

                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    loss = loss_function(features1, features2, model.module.logit_scale.exp())
                else:
                    loss = loss_function(features1, features2, model.logit_scale.exp()) 
                losses.update(loss.item())
                
                  
            scaler.scale(loss).backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            scheduler.step()

        else:
        
            # data (batches) to device   
            sat_img = sat_img.to(train_config.device)
            features1, features2 = model(sat_img, reference.to(train_config.device))

            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                loss = loss_function(features1, features2, model.module.logit_scale.exp())
            else:
                loss = loss_function(features1, features2, model.logit_scale.exp()) 
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()
               
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            scheduler.step()

        if train_config.verbose:
            
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def train_wav2vec2(train_config, model, dataloader, loss_function, optimizer_list, scheduler_list, scaler=None, lr_monitor:list[str]=None):
    # set model train mode
    model.train()
    
    losses = AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    for optimizer in optimizer_list:
        optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    # for loop over one epoch
    for sat_img, reference, ids in bar: 
        if scaler:
            with autocast():
                # data (batches) to device   
                sat_img = sat_img.to(train_config.device)

                waveform = reference[0].to(train_config.device)
                attention_mask = reference[1].to(train_config.device)
                features1, features2 = model(sat_img, waveform, attention_mask=attention_mask)

                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    loss = loss_function(features1, features2, model.module.logit_scale.exp())
                else:
                    loss = loss_function(features1, features2, model.logit_scale.exp()) 
                losses.update(loss.item())
                
                  
            scaler.scale(loss).backward()
            # Gradient clipping 
            if train_config.clip_grad:
                for optimizer in optimizer_list:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            for optimizer in optimizer_list:
                # Update model parameters (weights)
                scaler.step(optimizer)
                scaler.update()
                # Zero gradients for next step
                optimizer.zero_grad()
            
            # Scheduler
            for scheduler in scheduler_list:
                scheduler.step()

        else:
        
            # data (batches) to device   
            sat_img = sat_img.to(train_config.device)

            waveform = reference[0].to(train_config.device)
            attention_mask = reference[1].to(train_config.device)
            features1, features2 = model(sat_img, waveform, attention_mask=attention_mask)

            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                loss = loss_function(features1, features2, model.module.logit_scale.exp())
            else:
                loss = loss_function(features1, features2, model.logit_scale.exp()) 
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            for optimizer in optimizer_list:
                # Update model parameters (weights)
                optimizer.step()
                # Zero gradients for next step
                optimizer.zero_grad()
            
            # Scheduler
            for scheduler in scheduler_list:
                scheduler.step()

        if train_config.verbose:
            
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg)}

            # Parameter to print lr of different modules, blocks, functions, etc.
            if lr_monitor:
                lr_dict = {text: "{:.7f}".format(optimizer.param_groups[0]['lr'])
                        for text, optimizer in zip(lr_monitor, optimizer_list)}
            else:
                if len(optimizer_list) > 1:
                    lr_dict = {f"lr_{index}": "{:.7f}".format(optimizer.param_groups[0]['lr'])
                            for index, optimizer in enumerate(optimizer_list)}
                else:
                    lr_dict = {f"lr": "{:.7f}".format(optimizer_list[0].param_groups[0]['lr'])}

            monitor = {**monitor, **lr_dict}
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def predict(train_config, model, dataloader):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    item_features_list = []
    ids_list = []
    
    ids_list = []
    with torch.no_grad():
        
        for item, ids in bar:
        
            ids_list.append(ids)
            
            with autocast():
         
                if not isinstance(item, tuple) or item[1] == None:
                    item = item.to(train_config.device)
                    item_feature = model(item)
                else:
                    attention_mask = item[1].to(train_config.device)
                    item = item[0].to(train_config.device)
                    item_feature = model(item, attention_mask=attention_mask)

                # normalize is calculated in fp32
                if train_config.normalize_features:
                    item_feature = F.normalize(item_feature, dim=-1)
            
            # save features in fp32 for sim calculation
            item_features_list.append(item_feature.to(torch.float32))
        # keep Features on GPU
        item_features = torch.cat(item_features_list, dim=0) 
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
        
    if train_config.verbose:
        bar.close()
        
    return item_features, ids_list
