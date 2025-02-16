import torch
import torch.nn as nn

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DetectionHead, self).__init__()
        self.cls_fc = nn.Linear(in_channels, num_classes)  
        self.bbox_fc = nn.Linear(in_channels, 4, bias=False)  

    def forward(self, x):
        x = torch.flatten(x, 1)  
        class_logits = self.cls_fc(x)
        bbox_preds = self.bbox_fc(x)
        return class_logits, bbox_preds
