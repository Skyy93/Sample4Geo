import time
import torch

import torch.nn.functional as F

from math import ceil
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
from functools import wraps

from spectrum4geo.dataset.soundingearth_eval import SatEvalDataset, SpectroEvalDataset, SpectroTestDataset, WavEvalDataset

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













test=True
def predict_wav_test(train_config, model, dataloader):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    item_features_list = []
    ids_list = []
    with torch.no_grad():
        
        for item, ids in bar:
            ids_list.append(ids)
            
            with autocast():

                # for the wav2vec2 approach (audio)
                if dataloader.dataset.query_type == 'audio':
                    attention_mask = item[1].to(train_config.device)
                    item = item[0].to(train_config.device)
                    if not test:
                        item_feature = model(item, attention_mask=attention_mask)
                    else:
                        item_features_dict = {int(id): [] for id in ids} 
                        first_features = model(item, attention_mask=attention_mask)
                        for id, feature in zip(ids,first_features):
                            item_features_dict[int(id)].append(feature.squeeze())

                        for id in ids:
                            counter = 1
                            #print(f"{int(id)}\: " ,end="")
                            next_item = dataloader.dataset.get_next_chunk(int(id), counter)
                            while next_item is not None:
                                #print(f"{counter} ",end="")
                                item = next_item[0].to(train_config.device)
                                attention_mask = next_item[1].to(train_config.device)
                                feature = model(item, attention_mask=attention_mask)

                                item_features_dict[int(id)].append(feature.squeeze())

                                counter += 1
                                next_item = dataloader.dataset.get_next_chunk(int(id), counter)

                            #print()

                        item_feature = mymean(item_features_dict).to(train_config.device)

                # for images (sat/spectrogram)
                else:
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


def mymean(feature_dict):
    """Calculate the mean of lists of tensors stored in a dictionary."""
    aggregated_features = []
    for features in feature_dict.values():
        if features:  # ensure the list is not empty
            try:
                stacked_features = torch.stack(features)
                mean_features = torch.mean(stacked_features, dim=0)
                aggregated_features.append(mean_features)
            except RuntimeError as e:
                print("Error stacking features:", e)
                print("Shapes of features being stacked:", [f.shape for f in features])
    return torch.stack(aggregated_features)






def predict_basic(train_config, model, dataloader, tqdm_desk):
    model.eval()
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, desc=tqdm_desk, total=len(dataloader))
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



def predict_spectrograms_test(train_config, model, dataloader):
    model.eval()
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, desc="Extract Query Features (Mel Spectrograms)", total=len(dataloader))
    else:
        bar = dataloader
        
    item_features_list = []
    ids_list = []
    with torch.no_grad():
        
        for item, ids in bar:
            ids_list.append(ids)
            dataloader.set_label_ids(ids)
            
            with autocast():
                item = item.to(train_config.device)
                ids = ids.to(train_config.device)
                first_item_feature = model(item)
                
                chunks_weights = dataloader.get_chunks_weights(ids)
                chunks_weights = chunks_weights.to(train_config.device)

                item_feature = first_item_feature * chunks_weights
                dataloader.increment_next_chunk_counter() # to get next chunks in the next batch

                next_chunks_batched = dataloader.get_next_chunks_batched(ids)
                total_chunks = ceil(chunks_weights.reciprocal().sum().item() / dataloader.batch_size)  # Total number of chunk batches
                chunk_pbar = tqdm(total=total_chunks, desc="Processing chunks", leave=False, initial=1)

                while next_chunks_batched is not None:
                    next_chunks, next_chunks_ids = next_chunks_batched
                    next_chunks = next_chunks.to(train_config.device)
                    next_chunks_ids = next_chunks_ids.to(train_config.device)
                    next_chunks_feature = model(next_chunks)

                    item_feature_summand = torch.zeros_like(item_feature)
                    for i, id in enumerate(ids):
                        mask = (next_chunks_ids == id)
                        if mask.any():
                            item_feature_summand[i] += next_chunks_feature[mask].sum(dim=0)

                    item_feature += item_feature_summand * chunks_weights
                    chunk_pbar.update(1)  

                    next_chunks_batched = dataloader.get_next_chunks_batched(ids)

                # normalize is calculated in fp32
                if train_config.normalize_features:
                    item_feature = F.normalize(item_feature, dim=-1)
            
            # save features in fp32 for sim calculation
            dataloader.reset_next_chunk_counter()
            item_features_list.append(item_feature.to(torch.float32))
        # keep Features on GPU
        item_features = torch.cat(item_features_list, dim=0) 
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
        
    if train_config.verbose:
        bar.close()
        
    return item_features, ids_list



def get_predict_fct(dataloader):

    if isinstance(dataloader.dataset, SpectroTestDataset):
        return predict_spectrograms_test

    elif isinstance(dataloader.dataset, SpectroEvalDataset):
        return lambda **args: predict_basic(**args, tqdm_desc="Extract Query Features (Mel Spectrograms)")

    elif isinstance(dataloader.dataset, WavEvalDataset):
        return predict_wav_test

    elif isinstance(dataloader.dataset, SatEvalDataset):
        return lambda **args: predict_basic(**args, tqdm_desc="Extract Reference Features (Sat Images)")

    else:
        return None