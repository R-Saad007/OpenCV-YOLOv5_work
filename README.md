# I have added files to track my personal progress with these libraries.

## **Image Classification:**
It is the process of classifying the entire image on the basis of classes used to train the model before. It does not take into account the localization factor. The widely used model in contemporary world is the **VGG-19 CNN** (Convolution Neural Networks). **VGG-19** is a convolutional neural network that is 19 layers deep and can classify images into 1000 object categories such as a keyboard, mouse, and many animals. The model trained on more than a million images from the Imagenet database with an accuracy of 92%. Other models which are used include **YOLOv5, the Vision Transformer, and Resnet34**.

## **Object Detection:**
It is the process of detecting elements/objects within the same image based on the localization of their features. For instance, detecting a dog and cat in the same picture. These models work by using BBox(Bounding Boxes), drawing boxes, around the element. We can make use of **Image Segmentation** to include only the necessary pixels for each object i.e. pixel wise mapping. This will exclude the background, noise, and redundancies etc. **YOLO** is considered to be the best Object Detection model as of 2023. However, the **SSD** model is also in competition alongside the **R-CNN** (Region-based CNN).

## **Image Segmentation:**
An object detection technique which involves the use of the object's pixels in order to create detection regions instead of using BBox.

## **Image Augmentation:**
A technique used to increase a user's dataset. It might be due to a small dataset, or can act as one of the important measures to reduce overfitting problems. This is done by recreating images from previously available ones in the dataset by rotating, shearing, or transforming them. This technique generally increases the accuracy of the model and helps to reduce data labelling and cleaning costs.
