import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothing(nn.Module):
    
    def __init__(self, smoothing, pad_idx):
        super(LabelSmoothing, self).__init__()
        self.smoothing = smoothing
        self.pad_idx = pad_idx
        
    def forward(self, pred, target):  # pred (B, S, V), target (B, S)
        # Note: preds are expected to be after log
        B, S, V = pred.shape
        # (B, S, V) -> (B * S, V); (B, S) -> (B * S)
        pred = pred.contiguous().view(-1, V)
        target = target.contiguous().view(-1)
        
        # prior (uniform)
        dist = self.smoothing * torch.ones_like(pred) / (V - 2)
        # add smoothed ground-truth to prior (args: dim, index, src (value))
        dist.scatter_(1, target.unsqueeze(-1).long(), 1-self.smoothing)
        # make the padding token to have zero probability
        dist[:, self.pad_idx] = 0
        # ?? mask: 1 if target == pad_idx; 0 otherwise
        mask = torch.nonzero(target == self.pad_idx)
        
        if mask.sum() > 0 and len(mask) > 0:
            # dim, index, val
            dist.index_fill_(0, mask.squeeze(), 0)
            
        return F.kl_div(pred, dist, reduction='sum')
