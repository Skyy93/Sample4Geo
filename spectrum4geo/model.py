import torch
import timm
import numpy as np
import torch.nn as nn

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

class TimmModel(nn.Module):

    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=383):
                 
        super(TimmModel, self).__init__()
        
        self.img_size = img_size
        
        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        
    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

        
    def forward(self, img1, img2=None):
        if img2 is not None:
            image_features1 = self.model(img1)     
            image_features2 = self.model(img2)
            return image_features1, image_features2            
        else:
            image_features = self.model(img1)             
            return image_features



class TimmModelWav2Vec(TimmModel):

    def __init__(self, 
                 model_name,
                 model_name_wav2vec,
                 pretrained=True,
                 img_size=383):
                 
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
        self.wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")    

        super(TimmModelWav2Vec, self).__init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=383)

    def forward(self, img, wav):
        image_features = self.model(img) if img is not None else None
        wave_features = self.model_wav2vec(wav) if wav is not None else None
             
        return image_features, wave_features
    