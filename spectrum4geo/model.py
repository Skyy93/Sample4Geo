import torch
import timm
import numpy as np
import torch.nn as nn

from transformers import Wav2Vec2Model

class TimmModel(nn.Module):

    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=383):
                 
        super(TimmModel, self).__init__()
        
        self.img_size = img_size
        
        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.base_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
        else:
            self.base_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        
    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.base_model)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        self.base_model.set_grad_checkpointing(enable)

        
    def forward(self, img1, img2=None):
        if img2 is not None:
            image_features1 = self.base_model(img1)     
            image_features2 = self.base_model(img2)
            return image_features1, image_features2            
        else:
            image_features = self.base_model(img1)             
            return image_features



class TimmModelWav2Vec2(TimmModel):

    def __init__(self, 
                 model_name,
                 model_name_wav2vec,
                 pretrained=True,
                 img_size=383):
                 
        super(TimmModelWav2Vec2, self).__init__(model_name, pretrained, img_size)

        self.wav2vec2_model = Wav2Vec2Model.from_pretrained(model_name_wav2vec)
        self.projection = nn.Linear(768, 1024)  # assuming Wav2Vec2 has 768 hidden units, adjust if it's 1024        

    def forward(self, item_1, item_2=None):
        # Identification of the audio/img tensor
        if item_1.dim() == 4:  
            img = item_1
            waveform_data, attention_mask = item_2
        else:
            waveform_data, attention_mask = item_1
            img = item_2
        
        # Processing of the img tensor
        image_features = self.base_model(img) if img is not None else None

        # Processing of the waveform data
        if waveform_data is not None: 
            outputs = self.wav2vec2_model(input_values=waveform_data, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            hidden_states = self.projection(hidden_states)  # project to match image feature size

            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

            audio_features = self.classifier(pooled_output)

        else:
            audio_features = None
            
        # Return logic
        if image_features is not None and audio_features is not None:
            return image_features, audio_features
        elif image_features is None:
            return audio_features
        else:
            return image_features
    