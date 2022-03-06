import sys
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        print('num_classes =', num_classes)
        self.fc = nn.Sequential(
            nn.Linear(1280, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, num_classes, bias=True)
        )
    def forward(self, x):
        # print('x0: ', x.shape)              # [64, 3, 224, 224]
        
        x = self.model.extract_features(x)
        # print('x1: ', x.shape)              # [64, 1280, 7, 7]
        
        x = self.model._avg_pooling(x)
        # print('x2: ', x.shape)              # [64, 1280, 1, 1]
        
        x = x.flatten(start_dim=1)
        # print('x3: ', x.shape)              # [64, 1280]
        
        x = self.model._dropout(x)
        # print('x4: ', x.shape)              # [64, 1280]
        
        x = self.fc(x)
        # print('x5: ', x.shape)              # [64, 1]
        # sys.exit()
        
        return x
