import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import *


class MaskDataset(Dataset):
    image_paths = []
    mask_labels = []

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        
        self.data_dir = data_dir

        self.transform = transforms.Compose([
            Resize((224,224), Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])
        self.setup()

    def setup(self):
        image_folders = os.listdir(self.data_dir)
        for image_folder in image_folders:      #WithMask, WithoutMask
            image_path = os.path.join(self.data_dir, image_folder)
            for filename in os.listdir(image_path):
                self.image_paths.append(os.path.join(image_path, filename))
                if image_folder == 'WithMask':
                    self.mask_labels.append(0)
                else:
                    self.mask_labels.append(1)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        mask_label = self.mask_labels[index]

        image_transform = self.transform(image)
        return image_transform, torch.tensor(mask_label, dtype=torch.float32)

    def __len__(self):
        return len(self.image_paths)


