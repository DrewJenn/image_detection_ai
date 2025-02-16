import torch.nn as nn
from .detection_head import DetectionHead

class CustomBackboneWithHead(nn.Module):
    def __init__(self, num_classes):
        super(CustomBackboneWithHead, self).__init__()

        # Backbone layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.detection_head = DetectionHead(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.maxpool(x)
        x = self.global_avg_pool(x)

        x = x.view(x.size(0), -1)

        class_logits, bbox_preds = self.detection_head(x)
        return class_logits, bbox_preds
