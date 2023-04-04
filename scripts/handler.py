import torch
import cv2
import argparse
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# yolov5 handler for object dection in videos
# handler class
class handler():
    def __init__(self, vid):
        self.model = None                       # yolov5 model
        self.vid_path = vid                     # video path
        self.frame_list = None                  # list of frames/images in video
        self.inference_frame_list = None        # list of inferenced frames/images in video

    def load_model(self):
        # loading pytorch yolov5 model for inference
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
        # shifting model to GPU
        self.model.to(device)
        # for running inference on the person class only
        self.model.classes = [0]

    def frame_conversion(self):
        # converting video to image frames for inference
        # self.vid
        # update frame_list
        pass
        

    def inference(self):
        # inference on each video frame
        # update inference_frame_list
        pass

    def save_output(self):
        # storing each inferenced image frame
        pass

    def view_output(self):
        # to view the inferenced image
        pass

    def __del__(self):
        # object destructor
        print("Handler destructor invoked")

# main function
if __name__ == '__main__':
    # Argument from CLI
    parser = argparse.ArgumentParser(description = 'Input file path required.')
    parser.add_argument('-img_path', type = str, dest = 'img_path', required =True)
    args = parser.parse_args()

    # For calculating execution time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    print("Starting Video Inferencing")

    # whatever you are timing goes here
    vid_handler = handler(args.img_path)
    vid_handler.load_model()
    vid_handler.frame_conversion()
    vid_handler.inference()
    vid_handler.save_output()
    del vid_handler
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()
    print("Execution Time:","%.3f" % (start.elapsed_time(end)/1000), "seconds")  # seconds