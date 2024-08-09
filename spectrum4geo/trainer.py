import time
import torch

import torch.nn.functional as F

from math import ceil
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
from collections import defaultdict

from spectrum4geo.dataset.training import SpectroSimDataset
from spectrum4geo.dataset.evaluation import SatEvalDataset, SpectroEvalDataset, WavEvalDataset



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
    for sat_img, query, ids in bar: 
        if scaler:
            with autocast():
                # data (batches) to device   
                sat_img = sat_img.to(train_config.device)
                features1, features2 = model(sat_img, query.to(train_config.device))
                # Forward pass
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
            features1, features2 = model(sat_img, query.to(train_config.device))

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
            
            monitor = {'loss': '{:.4f}'.format(loss.item()),
                       'loss_avg': '{:.4f}'.format(losses.avg),
                       'lr' : '{:.6f}'.format(optimizer.param_groups[0]['lr'])}
            
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
    for sat_img, query, attention_mask, ids in bar: 
        if scaler:
            with autocast():
                # data (batches) to device   
                sat_img = sat_img.to(train_config.device)

                waveform = query.to(train_config.device)
                attention_mask = attention_mask.to(train_config.device)
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

            waveform = query.to(train_config.device)
            attention_mask = attention_mask.to(train_config.device)
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
            
            monitor = {'loss': '{:.4f}'.format(loss.item()),
                       'loss_avg': '{:.4f}'.format(losses.avg)}

            # Parameter to print lr of different modules, blocks, functions, etc.
            if lr_monitor:
                lr_dict = {text: '{:.7f}'.format(optimizer.param_groups[0]['lr'])
                        for text, optimizer in zip(lr_monitor, optimizer_list)}
            else:
                if len(optimizer_list) > 1:
                    lr_dict = {f'lr_{index}': '{:.7f}'.format(optimizer.param_groups[0]['lr'])
                            for index, optimizer in enumerate(optimizer_list)}
                else:
                    lr_dict = {f'lr': '{:.7f}'.format(optimizer_list[0].param_groups[0]['lr'])}

            monitor = {**monitor, **lr_dict}
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def predict_basic_image(train_config, model, dataloader, tqdm_desc):
    model.eval()
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, desc=tqdm_desc, total=len(dataloader))
    else:
        bar = dataloader
        
    item_features_list = []
    ids_list = []
    with torch.no_grad():
        
        for item, ids in bar:
            ids_list.append(ids)
            
            with autocast():
                item = item.to(train_config.device)
                item_feature = model(item)

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


def predict_chunked_spectrogram(train_config, model, dataloader):
    model.eval()
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, desc="[CHUNKING ENABLED] Extract Query Features (Mel Spectrograms)", total=len(dataloader))
    else:
        bar = dataloader

    with torch.no_grad(): # Disable gradient computation
        # first initialisation (One Run):
        sample = dataloader.dataset[0][0].unsqueeze(0).to(train_config.device)
        sample_feature = model(sample)
        item_feature_size = sample_feature.shape[1]
        del sample_feature

        # initialisation of resulting spectro_features_accumulated:
        num_label_ids = dataloader.dataset.len_of_label_ids()
        spectro_features_accumulated = torch.zeros(num_label_ids, item_feature_size, device=train_config.device)

        for spectro_batch, label_id_batch, weight_batch in bar:
            # Enable mixed precision
            with autocast(): 
                spectro_batch = spectro_batch.to(train_config.device)
                weight_batch = weight_batch.to(train_config.device)
                label_id_batch = label_id_batch.to(train_config.device)

                spectro_features = model(spectro_batch)
                weight_batch = weight_batch.unsqueeze(1).expand_as(spectro_features)
                spectro_features_weighted = weight_batch * spectro_features

                spectro_features_accumulated.index_add_(0, label_id_batch, spectro_features_weighted)

        # normalize is calculated in fp32
        if train_config.normalize_features:
            spectro_features_accumulated = F.normalize(spectro_features_accumulated, dim=-1)

        # save features in fp32 for sim calculation
        spectro_features_accumulated = spectro_features_accumulated.to(torch.float32)

        # creating label_ids using torch.arange() since spectro_features_accumulated is in order
        label_ids = torch.arange(num_label_ids, dtype=torch.long).to(train_config.device)
        
    if train_config.verbose:
        bar.close()
        
    return spectro_features_accumulated, label_ids


def predict_basic_wav(train_config, model, dataloader, tqdm_desc):
    model.eval()
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, desc=tqdm_desc, total=len(dataloader))
    else:
        bar = dataloader
        
    items_features_list = []
    ids_list = []
    with torch.no_grad():
        
        for item, attention_mask, ids in bar:
            ids_list.append(ids)
            
            with autocast():
                item = item.to(train_config.device)
                item_feature = model(item, attention_mask=attention_mask)

                # normalize is calculated in fp32
                if train_config.normalize_features:
                    item_feature = F.normalize(item_feature, dim=-1)
            
            # save features in fp32 for sim calculation
            items_features_list.append(item_feature.to(torch.float32))
            
        # keep Features on GPU
        items_features = torch.cat(items_features_list, dim=0) 
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
        
    if train_config.verbose:
        bar.close()
        
    return items_features, ids_list


def predict_chunked_wav(train_config, model, dataloader):
    model.eval()
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, desc="[CHUNKING ENABLED] Extract Query Features (Waveforms)", total=len(dataloader))
    else:
        bar = dataloader

    with torch.no_grad(): # Disable gradient computation
        # first initialisation (One Run):
        wav, _, _ = dataloader.dataset[0]
        wav_padded = dataloader.dataset.processor(wav, 
                                                  sampling_rate=dataloader.dataset.sample_rate, 
                                                  return_tensors="pt", 
                                                  padding=True, 
                                                  truncation=True, 
                                                  return_attention_mask=True,
                                                  max_length=dataloader.dataset.sample_length
                                                  )
        sample = wav_padded['input_values']
        sample_attention_mask = wav_padded['attention_mask']
        sample = sample.to(train_config.device)
        sample_attention_mask = sample_attention_mask.to(train_config.device)
        sample_feature = model(sample, attention_mask=sample_attention_mask)
        item_feature_size = sample_feature.shape[1]
        del sample_feature, sample_attention_mask, wav_padded

        # initialisation of resulting wav_features_accumulated:
        num_label_ids = dataloader.dataset.len_of_label_ids()
        wav_features_accumulated = torch.zeros(num_label_ids, item_feature_size, device=train_config.device)

        for wav_batch, attention_mask_batch, label_id_batch, weight_batch in bar:
            # Enable mixed precision
            with autocast(): 
                wav_batch = wav_batch.to(train_config.device)
                attention_mask_batch = attention_mask_batch.to(train_config.device)
                weight_batch = weight_batch.to(train_config.device)
                label_id_batch = label_id_batch.to(train_config.device)

                wav_features = model(wav_batch, attention_mask=attention_mask_batch)
                weight_batch = weight_batch.unsqueeze(1).expand_as(wav_features)
                wav_features_weighted = weight_batch * wav_features

                wav_features_accumulated.index_add_(0, label_id_batch, wav_features_weighted)

        # normalize is calculated in fp32
        if train_config.normalize_features:
            wav_features_accumulated = F.normalize(wav_features_accumulated, dim=-1)

        # save features in fp32 for sim calculation
        wav_features_accumulated = wav_features_accumulated.to(torch.float32)

        # creating label_ids using torch.arange() since spectro_features_accumulated is in order
        label_ids = torch.arange(num_label_ids, dtype=torch.long).to(train_config.device)
        
    if train_config.verbose:
        bar.close()
        
    return wav_features_accumulated, label_ids


def get_predict_fct(dataloader):
    """Retrieve the appropriate prediction function based on the dataset configuration."""
    if isinstance(dataloader.dataset, SpectroEvalDataset):
        if dataloader.dataset.chunking == True:
            return predict_chunked_spectrogram
        else: 
            return lambda train_config, model, dataloader: predict_basic_image(train_config, model, dataloader, tqdm_desc="Extract Query Features (Mel Spectrograms)")

    elif isinstance(dataloader.dataset, SpectroSimDataset):
        return lambda train_config, model, dataloader: predict_basic_image(train_config, model, dataloader, tqdm_desc="[FOR SIMILARITY SAMPLING] Extract Query Features (Mel Spectrograms)")

    elif isinstance(dataloader.dataset, SatEvalDataset):
        return lambda train_config, model, dataloader: predict_basic_image(train_config, model, dataloader, tqdm_desc="Extract Reference Features (Sat Images)")

    elif isinstance(dataloader.dataset, WavEvalDataset):
        if dataloader.dataset.chunking == True:
            return predict_chunked_wav
        else: 
            return lambda train_config, model, dataloader: predict_basic_wav(train_config, model, dataloader, tqdm_desc="Extract Query Features (Waveforms)")
    else:
        raise ValueError("No predict function for used dataset configuration found.")