import cv2 as cv
import os

train_img_path = r'/home/adlytic/datasets/Validation_Data/images/' # Path for all training images
train_label_path = r'/home/adlytic/datasets/Validation_Data/labels/' # Path for all training labels (annotations)
img_save_path = r'/home/adlytic/datasets/Validation_Data/bbox_images/' # Path for saving the drawn BBox

# Function to draw bounding box
def draw_Bbox(train_img_path, train_label_path):
    train_images = sorted(os.listdir(train_img_path)) # All training images
    train_labels = sorted(os.listdir(train_label_path)) # All training labels
    for x in range(0, len(train_images)):
        # Read Image
        img = cv.imread(os.path.join(train_img_path, train_images[x]))
        dimension = img.shape # (height, width, depth)
        with open(os.path.join(train_label_path, train_labels[x]), 'r') as f:
            file_data = [line.strip() for line in f]
        # For each object detected
        for y in range(0,len(file_data)):
            box_coords = file_data[y].split(' ')
            # Converting all string elements to floating point
            box_coords = [float(element) for element in box_coords] 
            startx = int((box_coords[1]*dimension[1]) - ((box_coords[3]/2)*dimension[1]))
            starty = int((box_coords[2]*dimension[0]) - ((box_coords[4]/2)*dimension[0]))
            endx = int((box_coords[1]*dimension[1]) + ((box_coords[3]/2)*dimension[1]))
            endy = int((box_coords[2]*dimension[0]) + ((box_coords[4]/2)*dimension[0]))
            start = (startx, starty)
            end = (endx, endy)
            # Drawing BBox
            cv.rectangle(img, start, end,(0,0,255),3)
        # To save the image somewhere
        img_path = os.path.join(img_save_path, train_images[x])
        cv.imwrite(img_path, img)
    print("BBox Generation Completed!")

if __name__ == "__main__":
    draw_Bbox(train_img_path=train_img_path, train_label_path=train_label_path)