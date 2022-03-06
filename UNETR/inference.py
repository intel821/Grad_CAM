from dataset import MaskDataset
from torch.utils.data import DataLoader
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import numpy as np

data_dir = './Test'
data_set = MaskDataset(data_dir)
data_loader = DataLoader(
    data_set,
    batch_size=15,
    shuffle=True,
)

from model import MyModel
import os
import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
m = torch.nn.Sigmoid()

model = MyModel(num_classes=1)

model_path = './model/exp3/best.pth'
model_path = './model/exp4/best.pth'
model.load_state_dict(torch.load(model_path, map_location=device))  # 모델 불러오기

model = model.to(device)
model.eval()

d = next(iter(data_loader))                      # loader에서 배치사이즈 만큼 random하게 뽑아온다.
# print('type(d) =', type(d))         # type(d) = <class 'list'>
# print('d.shape =', np.shape(d))     # d.shape = (2,)
# print(d[0].shape)   # torch.Size([15, 3, 224, 224]) -> 참고로 (15 = bs)
# print(d[1].shape)   # torch.Size([15])
# print(d[2].shape)   # list index out of range
# print(len(d))       # 2
# print(len(d[0]))    # 15 (=bs)

SHOW_PLOT = True
SHOW_PLOT = False

if SHOW_PLOT:
    fig, ax = plt.subplots(3, 5, figsize=(20,8))
    ax = ax.flatten()

with torch.no_grad():

    for i in range(15):
        if (SHOW_PLOT == False):
            data = d[0][i]
            data = data.to(device)
            break
        
        data = d[0][i]
        label = d[1][i]
        # print(data.shape)       # torch.Size([3, 224, 224])
        # print(label.shape)      # torch.Size([])
        
        ax[i].imshow(data.permute(1,2,0))   # plt에 맞춰서 permute
        
        data = data.to(device)

        # print(data.unsqueeze(0).shape)    # torch.Size([1, 3, 224, 224])
        out = model(data.unsqueeze(0))      # 기존 3차원에서 배치까지 포함된 4차원에 맞추기 위해 unsqueeze
        out = m(out)

        if round(out.item()) == 0:     
            title = 'Mask'
        else:
            title = 'normal'
            
        ax[i].set_title(title)
        ax[i].axis('off')

if SHOW_PLOT:
    plt.show()
    print('data.shape =', data.shape)   # torch.Size([3, 224, 224])


print('abc')
print(model.model._blocks[-1])
print('defg')
sys.exit()

### Grad-CAM
save_feat=[]
def hook_feat(module, input, output):   # feature 뽑기
  save_feat.append(output)
  return output


save_grad=[]
def hook_grad(grad):    # Gradient 뽑기
  save_grad.append(grad)
  return grad


def vis_gradcam(model, img):
  model.eval()

  model.model._blocks[-1]._swish.register_forward_hook(hook_feat)
  print('bb')
  print(save_feat)
  print(len(save_feat))         # 0
  # print(save_feat[0].shape)     # error
  
  # print(img.shape)  # torch.Size([3, 224, 224])
  img = img.unsqueeze(0)
  # print(img.shape)  # torch.Size([1, 3, 224, 224])
  print(len(save_feat))         # 0

  s = model(img)[0]
  # print(s.shape)    # torch.Size([1])
  
  print(len(save_feat))         # 3
  print(save_feat[0].shape)     # torch.Size([1, 1152, 7, 7])
  # print(save_feat[0])
  
  save_feat[0].register_hook(hook_grad)
  # print(save_feat.shape)
  # print(save_feat[0])
  # sys.exit()
  
  y = torch.argmax(s).item()
  s_y = s[y]
  s_y.backward()


  gap_layer  = torch.nn.AdaptiveAvgPool2d(1)
  alpha = gap_layer(save_grad[0][0].squeeze())
  A = save_feat[0].squeeze()

  relu_layer = torch.nn.ReLU()

  weighted_sum = torch.sum(alpha*A, dim=0)
  grad_CAM = relu_layer(weighted_sum)

  grad_CAM = grad_CAM.unsqueeze(0)
  grad_CAM = grad_CAM.unsqueeze(0)

  upscale_layer = torch.nn.Upsample(scale_factor=img.shape[-1]/grad_CAM.shape[-1], mode='bilinear')

  grad_CAM = upscale_layer(grad_CAM)
  grad_CAM = grad_CAM/torch.max(grad_CAM)

  # Plotting  
  img = img.squeeze()
  img = img.cpu()
  img = img.permute(1,2,0)
  grad_CAM = grad_CAM.cpu()
  grad_CAM = grad_CAM.squeeze().detach().numpy()

  plt.figure(figsize=(8, 8))
  plt.imshow(img)
  plt.imshow(grad_CAM, cmap='jet', alpha = 0.5)
  plt.show

  return grad_CAM

vis_gradcam(model, data)

