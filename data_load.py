import os
import torch
from torchvision import transforms
from torchvision.io import read_image

image_path = r'/home/adlytic/datasets/Train_Data/images/'

# Function to check Cuda Functionality
def check_cuda():
    if torch.cuda.is_available(): 
        dev = "cuda:0" 
    else: 
        dev = "cpu"
    device = torch.device(dev)
    return device

class CustomImageDataset():
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        train_images = sorted(os.listdir(self.img_dir))
        self.len = len(train_images)
        return self.len

    def __getitem__(self):
        tensor_output = []
        train_images = sorted(os.listdir(self.img_dir))
        for x in range(self.__len__()):
            img_path = os.path.join(self.img_dir, train_images[x])
            image = read_image(img_path)
            if self.transform:
                image = self.transform(image)
                device = check_cuda()
                image = image.to(device)
                tensor_output.append(image)
        return tensor_output
    
# Transformation on images
tf=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512,640)), #(Height, Width)
    transforms.ToTensor()
])


# Main Function
if __name__ == "__main__":
    custom_dataset = CustomImageDataset(img_dir = image_path, transform = tf)
    # Returns a list of all image tensors
    img_tensors = custom_dataset.__getitem__()
    print("Tensor Output:", len(img_tensors),"images")
    print('\nDevice Name:',torch.cuda.get_device_name())