import torch
#import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.transforms.functional import resize
from dataclasses import dataclass
import os
import numpy as np

from spectrum4geo.dataset.soundingearth_wav2vec2 import Wav2Vec2SoundingEarthDatasetEval, Wav2Vec2SoundingEarthDatasetTrain
from torch.utils.data import DataLoader
from spectrum4geo.transforms import get_transforms_train_sat
from spectrum4geo.transforms import get_transforms_val_sat

@dataclass
class Configuration:
    img_size: int = 384                # for satallite images
    model_wav2vec: str = 'facebook/wav2vec2-base-960h'
    sr_kHz = 16
    audio_length_s = 20

    batch_size: int = 48               # keep in mind real_batch_size = 2 * batch_size
    data_folder = "data"               # Dataset
    
    # Augment Images
    prob_rotate: float = 0.75          # rotates the sat image 
    prob_flip: float = 0.5             # flipping the sat image 
    num_workers: int = 0 if os.name == 'nt' else 4 
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    mean=[1, 1, 1]
    std=[1, 1, 1]
    
config = Configuration()

img_size = config.img_size
img_size_sat = (img_size, img_size)

# Transforms
sat_transforms_train = get_transforms_train_sat(img_size_sat,
                                                    mean=config.mean,
                                                    std=config.std,
                                                    )


train_dataset = Wav2Vec2SoundingEarthDatasetTrain(data_folder=config.data_folder ,
                                          split_csv='train_df.csv',
                                          transforms_sat_image=sat_transforms_train,
                                          audio_length_s=config.audio_length_s,
                                          sr_kHz=config.sr_kHz,
                                          processor_wav2vec2=config.model_wav2vec,
                                          prob_flip=config.prob_flip,
                                          prob_rotate=config.prob_rotate,
                                          shuffle_batch_size=config.batch_size,
                                                                                    )

train_dataloader = DataLoader(train_dataset,
                              batch_size=12,  
                              shuffle=True,
                              num_workers=config.num_workers,
                              pin_memory=True,
                              collate_fn=train_dataset.collate_fn)


# Eval
sat_transforms_val = get_transforms_val_sat(img_size_sat,
                                                mean=config.mean,
                                                std=config.std,
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
                                        batch_size=config.batch_size,
                                        num_workers=config.num_workers,
                                        shuffle=False,
                                        pin_memory=True,
                                        collate_fn=sat_dataset_test.collate_fn)
    
    

# Reference Spectogram Images
wave_dataset_test = Wav2Vec2SoundingEarthDatasetEval(data_folder=config.data_folder ,
                                    split_csv='test_df.csv',
                                    query_type = "audio",
                                    transforms=sat_transforms_val,
                                    audio_length_s=config.audio_length_s,
                                    sr_kHz=config.sr_kHz,
                                    processor_wav2vec2=config.model_wav2vec,
                                    )

wave_dataloader_test = DataLoader(wave_dataset_test,
                                    batch_size=config.batch_size,
                                    num_workers=config.num_workers,
                                    shuffle=False,
                                    pin_memory=True,
                                    collate_fn=wave_dataset_test.collate_fn)


print("Reference (Sat) Images Test:", len(sat_dataset_test))
print("Reference (wave) Images Test:", len(wave_dataset_test))
    
dataloader_val_tup = (wave_dataloader_test, sat_dataloader_test)


def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def plot_images(dataloader, num_images, sat_den_b=True):
    #fig, axs = plt.subplots(nrows=num_images, ncols=1, figsize=(20, 5 * num_images))  
    #axs = axs.flatten() 

    image_count = 0
    for data in dataloader:
        sat_images, waves_tensors, _ = data  

        for sat_image, waves_tensor in zip(sat_images, waves_tensors):
            if image_count >= num_images:
                break

            if sat_den_b:
                sat_image = denormalize(sat_image, config.mean, config.std)

            # Convert wave tensor to image
            waves_image = waves_tensor.unsqueeze(0).repeat(3, 1, 1)  # Convert to 3-channel image
            target_height = sat_image.shape[1]
            waves_image_resized = resize(waves_image, (target_height, waves_image.shape[-1]))

            # Concatenate satellite image and wave image horizontally
            combined_image = torch.cat((sat_image, waves_image_resized), dim=2)

            #ax = axs[image_count]
            #ax.imshow(combined_image.permute(1, 2, 0).numpy())  
            #ax.axis('off')
            image_count += 1

        if image_count >= num_images:
            break

    #plt.tight_layout()
    #plt.show()  <- habe ich deaktiviert, da du es in der Shell ausführst

plot_images(train_dataloader, 3, sat_den_b = True)
