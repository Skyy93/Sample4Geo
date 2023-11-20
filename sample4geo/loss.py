import torch
import torch.nn as nn
from torch.nn import functional as F

import torch.distributed.nn
from torch import distributed as dist

def gather_features(features1, features2):
    all_features1 = torch.cat(torch.distributed.nn.all_gather(features1), dim=0)
    all_features2 = torch.cat(torch.distributed.nn.all_gather(features2), dim=0)
    return all_features1, all_features2

class InfoNCE(nn.Module):

    def __init__(self, loss_function):
        super().__init__()
        self.loss_fn = loss_function

    def forward(self, features1, features2, logit_scale):
        device = features1.device

        features1, features2 = gather_features(features1, features2)
        
        features1 = F.normalize(features1, dim=-1)
        features2 = F.normalize(features2, dim=-1)
        
        logits_per_feat1 = logit_scale * features1 @ features2.T
        logits_per_feat2 = logits_per_feat1.T

        labels = torch.arange(len(logits_per_feat1), dtype=torch.long, device=device)
        
        total_loss = (
            self.loss_fn(logits_per_feat1, labels) +
            self.loss_fn(logits_per_feat2, labels)
        ) / 2

        return total_loss 
