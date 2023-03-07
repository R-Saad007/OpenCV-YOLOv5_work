import cv2 as cv
import os
import statistics

cropped_img_save_path = r'/home/adlytic/datasets/Validation_Data/cropped_images/' # Path for saving the cropped images
cropped_images = sorted(os.listdir(cropped_img_save_path)) # Cropped Images
width_data = [] # list to store width of all cropped images
height_data = [] # list to store height of all cropped images
for x in range(len(cropped_images)):
    # Reading Image
    img = cv.imread(os.path.join(cropped_img_save_path, cropped_images[x]))
    # (Height, Width, Depth)
    dimension = img.shape
    height_data.append(dimension[0])
    width_data.append(dimension[1])
# Calculating mean width and height
mean_width = statistics.mean(width_data)
mean_height = statistics.mean(height_data)
# Calculating standard deviation of width and height
sd_width = statistics.stdev(width_data, mean_width)
sd_height = statistics.stdev(height_data, mean_height)
print("Mean width: ", mean_width,"\nMean height: ",mean_height,"\nStandard Deviation of width: ", sd_width,"\nStandard Deviation of height: ", sd_height)