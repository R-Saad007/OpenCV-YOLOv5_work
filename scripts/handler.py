import torch
import cv2
import argparse
import time
import pandas
import json
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# yolov5 handler for object dection in videos
# handler class
class handler():
    def __init__(self, vid):
        self.model = None                       # yolov5 model
        self.vid_path = vid                     # video path
        self.frame_list = list()                # list of frames/images in video

    def load_model(self):
        # loading pytorch yolov5 model for inference
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
        # shifting model to GPU
        self.model.to(device)
        # for running inference on the person class only
        self.model.classes = [0]

    def frame_conversion(self):
        print("Starting Video Processing...")
        # video capture object
        cap = cv2.VideoCapture(self.vid_path)
        # capturing first frame to check whether video exists for processing below
        ret, frame = cap.read()
        # video processing loop
        while ret:
            # appending each frame to the frame list
            self.frame_list.append(frame)
            # checking for user's exit command
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            # error handling for last frame
            try:
                ret, frame = cap.read()                                
            except Exception:
                pass
        # release the video capture object
        cap.release()
        # Closes all the windows currently opened.
        cv2.destroyAllWindows()
        print("Video Processing Completed!")
        print('Number of frames: ',len(self.frame_list))

    def inference(self):
        # inference on all video frames
        print("Starting Frame Inferencing...")
        for x in range(len(self.frame_list)):
            results = self.model(self.frame_list[x]) # changed code to allow for CUDA memory usage / multiple runs problem
            self.write_output(x+1, results.pandas().xyxy[0].to_json(orient='records')) # converting each frame to a JSON object for the JSON file
            results = np.array(results.render()) # selecting the frame from the inferenced output (YOLOv5 Detection class)
        print("Inferencing Completed!")
    
    def view_output(self):
        # display inferenced output
        # calculating time between frames to display fps information
        prev_time = 0.0
        new_frame_time = 0.0
        font = cv2.FONT_HERSHEY_SIMPLEX
        for frame in self.frame_list:
            new_frame_time = time.time()
            fps = 'FPS: ' + str(int(1/(new_frame_time-prev_time)))
            cv2.putText(frame, fps, (7, 70), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Frame" , frame)
            prev_time = new_frame_time
            if cv2.waitKey(25) and 0xFF == ord("q"):
                break
        # Closes all the windows currently opened.
        cv2.destroyAllWindows()

    def write_output(self, frameno, data):
        # writing JSON format output to a file
        with open("sample.json", "a") as outfile:
            # indenting the JSON data according to the YOLOv5 JSON output parameters
            if data == "[]":
                data = 'No DETECTION'
            else:
                data = json.dumps(data)
            result = f'Frame: {frameno} \t Detections: ' + data + '\n'
            outfile.write(result)
        outfile.close()

    def __del__(self):
        # object destructor
        self.model = None                                                   # yolov5 model
        self.vid_path = None                                                # video path
        self.frame_list = self.frame_list.clear()                           # list of frames/images in video
        print("Handler destructor invoked!")

# main function
if __name__ == '__main__':
    # Argument from CLI
    parser = argparse.ArgumentParser(description = 'I/O file paths required.')
    parser.add_argument('-img_path', type = str, dest = 'img_path', required =True)
    args = parser.parse_args()

    # For calculating execution time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    # whatever you are timing goes here
    vid_handler = handler(args.img_path)
    vid_handler.load_model()
    vid_handler.frame_conversion()
    vid_handler.inference()
    vid_handler.view_output()
    del vid_handler
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()
    print("Execution Time:","%.3f" % (start.elapsed_time(end)/1000), "seconds")  # seconds
