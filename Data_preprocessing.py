# Imports
import os 
import random
import shutil

random.seed(108)
# Dataset already contains the text files required for the dataset (txt format with class, xcentre, ycentre, widht, height)
# folder path
dir_path = r'/home/adlytic/yolov5/classify/YOLOv5_Dataset'

# list to store files names of annotations
txt_file_names = []
# list to store file names of images
img_file_names = []
# This function takes only 1000 data points
list_dir = sorted(os.listdir(dir_path))
for file in list_dir:
    # check only text files
    if file.endswith('.txt'):
        txt_file_names.append(file)
    #check only img files
    elif file.endswith('.jpg'):
        img_file_names.append(file)

#splitting data into train, validation, and test sets (80,10,10)
train_data = []
validation_data = []
test_data = []

# Training Data
train_data = txt_file_names[:int(len(txt_file_names)*0.8)]
train_data_imgs = img_file_names[:int(len(txt_file_names)*0.8)]
# Validation Data
validation_data = txt_file_names[int(len(txt_file_names)*0.8):int(len(txt_file_names)*0.9)]
validation_data_imgs = img_file_names[int(len(txt_file_names)*0.8):int(len(txt_file_names)*0.9)]
# Test Data
test_data = txt_file_names[int(len(txt_file_names)*0.9):]
test_data_imgs = img_file_names[int(len(txt_file_names)*0.9):]

# Moving files to their respective directories

source_folder = r'/home/adlytic/datasets/Test_Data/'
img_folder = r'/home/adlytic/datasets/Test_Data/images/'
label_folder = r'/home/adlytic/datasets/Test_Data/labels/'
train_folder = r'/home/adlytic/yolov5/classify/Train_Data/'
validation_folder = r'/home/adlytic/yolov5/classify/Validation_Data/'
test_folder = r'/home/adlytic/yolov5/classify/Test_Data/'

Test_data = sorted(os.listdir(source_folder))
# Moving all Training images to images folder
# for file in Test_data:
#     if file.endswith('.jpg'):
#         source = source_folder + file
#         destination = img_folder + file
#         shutil.copy(source, destination)
#     if file.endswith('.txt'):
#         source = source_folder + file
#         destination = label_folder + file
#         shutil.copy(source, destination)

# Moving all Train img files
for x in range(0, len(train_data)):
    # construct full file path
    source = source_folder + train_data_imgs[x]
    destination = train_folder + train_data_imgs[x]
    shutil.copy(source, destination)
    source = source_folder + train_data[x]
    destination = train_folder + train_data[x]
    shutil.copy(source, destination)

# Moving all Validation Data files
for x in range(0, len(validation_data)):
    # construct full file path
    source = source_folder + validation_data_imgs[x]
    destination = validation_folder + validation_data_imgs[x]
    shutil.copy(source, destination)
    source = source_folder + validation_data[x]
    destination = validation_folder + validation_data[x]
    shutil.copy(source, destination)

# Moving all Test Data files
for x in range(0, len(test_data)):
    # construct full file path
    source = source_folder + test_data_imgs[x]
    destination = test_folder + test_data_imgs[x]
    shutil.copy(source, destination)
    source = source_folder + test_data[x]
    destination = test_folder + test_data[x]
    shutil.copy(source, destination)