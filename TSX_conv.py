import numpy as np                    	# 수학 계산 관련 라이브러리
import torch                          	# 파이토치 관련 라이브러리
import torch.nn as nn                 	# neural network 관련 라이브러리
import torchsummaryX

batch_size = 99

class My_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.zConv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1,  out_channels=16, kernel_size=5),  # [bs, 1, 28, 28] -> [bs, 16, 24, 24]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                      # [bs, 16, 24, 24] -> [bs, 16, 12, 12]
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),  # [bs, 16, 12, 12] -> [bs, 32, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                      # [bs, 32, 8, 8] -> [bs, 32, 4, 4]
        )
        self.zFC_layer = nn.Sequential(
            nn.Linear(32*4*4, 10),                                      # [bs, 32*4*4] -> [bs, 10]
        )
        
    def forward(self, x):                           # x.shape = (bs, 1, 28, 28)
        out_data = self.zConv_layer(x)               # out_data.shape = (bs, 32, 4, 4)
        out_data = out_data.view(batch_size, -1)  	# out_data.shape = (bs, 32*4*4) = (bs, 512)
        out_data = self.zFC_layer(out_data)          # out_data.shape = (bs, 10)
        return out_data

# torchsummaryX.summary(My_Model(), torch.zeros((batch_size, 1, 28, 28)))


print()
model = My_Model()
print(model)

torchsummaryX.summary(model, torch.zeros((batch_size, 1, 28, 28)))

