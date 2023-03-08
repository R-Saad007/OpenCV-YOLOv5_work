import os
import cv2 as cv
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.io import read_image

print(torch.cuda.is_available())


image_path = r'/home/adlytic/datasets/Train_Data/images/hush_puppies_1.jpg'


class CustomImageDataset():
    def __init__(self, img_dir, transform=None):
        # self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        # self.target_transform = target_transform

    # def __len__(self):
    #     return len(self.img_labels)

    def __getitem__(self):
        print("get item: ", self.img_dir)
        img_path = os.path.join(self.img_dir)
        image = read_image(img_path)
        # label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image
    
# target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
tf=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512,640)), #(Height, Width)
    transforms.ToTensor()
])

custom_dataset = CustomImageDataset(img_dir = image_path, transform = tf)
img = custom_dataset.__getitem__()
img = img.numpy()
print(img)
# cv.imshow('1',img)
# cv.waitKey(0)