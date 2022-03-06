import numpy as np                    	# 수학 계산 관련 라이브러리
import torch                          	# 파이토치 관련 라이브러리
import torch.nn as nn                 	# neural network 관련 라이브러리
import torchsummaryX

batch_size = 99

class My_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.zTest_layer = nn.Sequential(
            # nn.Linear(784, 50, bias=True),
            nn.Linear(784, 50, bias=False),
            nn.ReLU(),		# nn.Sigmoid() 함수를 사용할 수도 있음.
            nn.Linear(50, 10)
        )       
        
    def forward(self, x):			# x.shape = (batch_size, 1, 28, 28)
        in_data = x.view(batch_size, -1)	# in_data.shape = (batch_size, 784)
        out_data = self.zTest_layer(in_data)		# out_data.shape = (batch_size, 10)
        return out_data

torchsummaryX.summary(My_Model(), torch.zeros((batch_size, 1, 28, 28)))
# model = My_Model() 라인 후에... torchsummaryX.summary(model, .....) 방식으로 사용할 수도 있음.

print()
model = My_Model()
print(model)
