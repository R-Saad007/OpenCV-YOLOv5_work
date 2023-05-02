import torch
import cv2
import argparse
import json
import os
import pandas
from ultralytics import YOLO
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# yolov8 handler for object detection in images
# handler class
class handler():
    def __init__(self, img_path, model_path):
        self.model = None                       # yolov8 model
        self.img_path = img_path                # file path for images
        self.model_path = model_path            # path for model weight

    def load_model(self):
        # loading pytorch yolov8 model for inference
        self.model = YOLO(self.model_path)
        # shifting model to GPU/CPU depending on resource available
        self.model.to(device)

    def img_processing(self):
        print("Starting image processing...")
        # obtaining all images from the file path
        images = sorted(os.listdir(self.img_path))
        print("Inferencing...")
        for x in range(len(images)):
            # reading image from entire file path
            img = cv2.imread(f'{self.img_path}/{images[x]}')
            # inferencing
            results = self.model.predict(source=img)
            for res in results:
                # processing for each bounding box in the image
                boxes = res.boxes
                if not boxes:
                    self.write_output((x+1), '','')
                    break
                for box in boxes:
                    # bringing the tensor output from GPU to CPU and converting it to numpy array to get the bounding box coordinates
                    bbox = box.cpu().numpy().xyxy[0]
                    # extracting classname
                    classname = self.model.names[int(box.cls)]
                    # converting the numpy array to a dataframe for JSON processing
                    data = pandas.DataFrame(bbox)
                    # writing the coordinates to a JSON file
                    self.write_output((x+1), data[0].to_json(orient='records'), classname)
        print("JSON file saved!")
        print("Images processed: ", len(images))

    def write_output(self, frameno, data, name):
        # writing JSON format output to a file
        with open("output.json", "a") as outfile:
            # indenting the JSON data according to the YOLOv8 JSON output parameters
            if data == "":
                data = 'No DETECTION'
                result = f'Image: {frameno}\tDetections: ' + data + '\n'
            else:
                data = json.dumps(data)
                result = f'Image: {frameno}\tClass: {name}\t Detections: ' + data + '\n'
            outfile.write(result)
        outfile.close()

    def __del__(self):
        # object destructor
        self.model = None                                                   # yolov8 model
        self.vid_path = None                                                # file path for images
        self.model_path = None                                              # path for model weights
        print("Exiting...")

# main function
if __name__ == '__main__':
    # Arguments from CLI
    '''    
        
        Method to execute code
         
        python yolov8_coordinate_parser.py -img_path (folder path containing images) -model_path (direct path to the weights i.e. yolov8n.pt)

    '''

    parser = argparse.ArgumentParser(description = 'I/O and model file paths required.')
    parser.add_argument('-img_path', type = str, dest = 'img_path', required =True)
    parser.add_argument('-model_path', type = str, dest = 'model_path', required =True)
    args = parser.parse_args()

    # For calculating execution time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    # whatever you are timing goes here
    vid_handler = handler(args.img_path, args.model_path)
    vid_handler.load_model()
    vid_handler.img_processing()
    del vid_handler
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()
    print("Execution Time:","%.3f" % (start.elapsed_time(end)/1000), "seconds")  # seconds
