import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import random
import numpy as np


class Cut(ImageOnlyTransform):
    def __init__(self, 
                 cutting=None,
                 always_apply=False,
                 p=1.0):
        
        super(Cut, self).__init__(always_apply, p)
        self.cutting = cutting
    
    
    def apply(self, image, **params):
        
        if self.cutting:
            image = image[self.cutting:-self.cutting,:,:]
            
        return image
            
    def get_transform_init_args_names(self):
        return ("size", "cutting")     


class SpectroTimeMaskTransform(ImageOnlyTransform):
    def __init__(self, 
                 time_mask_range = (15, 50),
                 always_apply=False,
                 p=1.0):
        
        super(SpectroTimeMaskTransform, self).__init__(always_apply, p)
        self.time_mask_range = time_mask_range
    
    def apply(self, image, **params):
        time_mask_param = np.random.randint(*self.time_mask_range)
        num_time_bins = image.shape[1]

        t = np.random.randint(0, num_time_bins - time_mask_param)
        image[:, t:t + time_mask_param] = 0
        
        return image
            
    def get_transform_init_args_names(self):
        return ("size", "time_mask_range")     
    

class SpectroFrequencyMaskTransform(ImageOnlyTransform):
    def __init__(self, 
                 freq_mask_range = (5, 15),
                 always_apply=False,
                 p=1.0):
        
        super(SpectroFrequencyMaskTransform, self).__init__(always_apply, p)
        self.freq_mask_range = freq_mask_range
    
    def apply(self, image, **params):
        freq_mask_param = np.random.randint(*self.freq_mask_range)
        num_freq_bins = image.shape[0]
        
        f = np.random.randint(0, num_freq_bins - freq_mask_param)
        image[f:f + freq_mask_param, :] = 0

        return image
            
    def get_transform_init_args_names(self):
        return ("size", "freq_mask_range")   



def get_transforms_train_sat(image_size_sat,
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],):
    
    
    satellite_transforms = A.Compose([
                                      A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0),
                                               A.CoarseDropout(max_holes=25,
                                                               max_height=int(0.2*image_size_sat[0]),
                                                               max_width=int(0.2*image_size_sat[0]),
                                                               min_holes=10,
                                                               min_height=int(0.1*image_size_sat[0]),
                                                               min_width=int(0.1*image_size_sat[0]),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])
            
    return satellite_transforms


def get_transforms_train_ground(img_size_ground,
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        ground_cutting=0):
    
    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                   A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                           ], p=0.3),
                                   A.OneOf([
                                            A.GridDropout(ratio=0.5, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2*img_size_ground[0]),
                                                            max_width=int(0.2*img_size_ground[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1*img_size_ground[0]),
                                                            min_width=int(0.1*img_size_ground[0]),
                                                            p=1.0),
                                           ], p=0.3),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   ])
                
    return ground_transforms


def get_transforms_train_spectro(img_size_spectro,   
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225],
                                spectro_cutting=0,):
            
    spectro_transforms = A.Compose([Cut(cutting=spectro_cutting, p=1.0),
                                A.ColorJitter(brightness=(0.85, 1.15), contrast=(0.75, 1.25), saturation=(0.75, 1.25), hue=(-0.02,0.02), p=0.75),
                                A.OneOf([
                                    A.AdvancedBlur(p=1.0),
                                    A.GaussianBlur(p=1.0),
                                    A.Sharpen(p=1.0),
                                ], p=0.45),
                                A.OneOf([
                                            A.GridDropout(ratio=0.175, p=1.0),
                                            A.CoarseDropout(max_holes=12,
                                                            max_height=int(0.2*img_size_spectro[1]),
                                                            max_width=int(0.2*img_size_spectro[0]),
                                                            min_holes=5,
                                                            min_height=int(0.1*img_size_spectro[1]),
                                                            min_width=int(0.1*img_size_spectro[0]),
                                                            p=1.0),
                                        ], p=0.3),
                                SpectroTimeMaskTransform(time_mask_range=(0.095*img_size_spectro[0],0.175*img_size_spectro[0]), p=0.9),
                                SpectroFrequencyMaskTransform( freq_mask_range=(0.085*img_size_spectro[1],0.15*img_size_spectro[1]), p=0.9),
                                A.Normalize(mean, std, max_pixel_value=1), 
                                ToTensorV2(),
                                ])

    return spectro_transforms


def get_transforms_val_sat(image_size_sat,
                       mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225],):
    
    satellite_transforms = A.Compose([A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])
                   
    return satellite_transforms


def get_transforms_val_ground(img_size_ground,
                       mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225],
                       ground_cutting=0):

    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                    A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                    A.Normalize(mean, std),
                                    ToTensorV2(),
                                    ])
    
    return ground_transforms
            

def get_transforms_val_spectro(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225],
                       spectro_cutting=0,):
    
    spectro_transforms = A.Compose([Cut(cutting=spectro_cutting, p=1.0),
                                    A.Normalize(mean, std, max_pixel_value=1), 
                                    ToTensorV2(),
                                    ])
               
    return spectro_transforms


