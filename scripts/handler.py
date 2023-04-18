import torch
import cv2
import argparse
import time
import pandas
import json
import numpy as np
from yolox.tracker.byte_tracker import BYTETracker
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ByteTrack args class
class bytetrackerargs():
    track_thresh: float = 0.25                  # tracking threshold
    track_buffer: int = 30                      # frames in buffer
    match_thresh: float = 0.8                   # matching threshold
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0                   # smallest possible bbox
    mot20: bool = False                         # not using mot20

# yolov5 handler for object dection in videos
# handler class
class handler():
    def __init__(self, vid):
        self.model = None                       # yolov5 model
        self.vid_path = vid                     # video path
        self.frame_list = list()                # list of frames/images in video
        self.targets = list()                   # list of tracker outputs

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
        tracker = BYTETracker(bytetrackerargs) # byte tracker object
        for x in range(len(self.frame_list)):
            results = self.model(self.frame_list[x], size = 640) # changed code to allow for CUDA memory usage / multiple runs problem
            detections = self.bytetrackconverter(results)
            online_targets = tracker.update(detections, self.frame_list[x].shape[:2], self.frame_list[x].shape[:2]) # tracker output
            self.targets.append(online_targets)
            self.write_output(x+1, results.pandas().xyxy[0].to_json(orient='records')) # converting each frame to a JSON object for the JSON file
            results = np.array(results.render()) # selecting the frame from the inferenced output (YOLOv5 Detection class)
        print("Inferencing Completed!")
        print("JSON file created!")
    
    def view_output_YOLOv5(self):
        # display inferenced output
        # calculating time between frames to display fps information
        prev_time = 0.0
        new_frame_time = 0.0
        font = cv2.FONT_HERSHEY_SIMPLEX
        # coordinates for the marked region (footfall counter)
        start = (620,500)
        end = (1000,650)
        for frame in self.frame_list:
            new_frame_time = time.time()
            fps = 'FPS: ' + str(int(1/(new_frame_time-prev_time)))
            # FPS text
            cv2.putText(frame, fps, (7, 70), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
            # Footfall region
            cv2.rectangle(frame, start, end,(0,255,0),-1)
            cv2.imshow("Frame" , frame)
            prev_time = new_frame_time
            if cv2.waitKey(25) and 0xFF == ord("q"):
                break
        # Closes all the windows currently opened.
        cv2.destroyAllWindows()

    def view_output_ByteTrack(self):
        # display inferenced output
        # calculating time between frames to display fps information
        prev_time = 0.0
        new_frame_time = 0.0
        font = cv2.FONT_HERSHEY_SIMPLEX
        # coordinates for the marked region (footfall counter)
        start = (620,500)
        end = (1000,650)
        for x in range(len(self.frame_list)):
            new_frame_time = time.time()
            fps = 'FPS: ' + str(int(1/(new_frame_time-prev_time)))
            # FPS text
            cv2.putText(self.frame_list[x], fps, (7, 70), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
            # Footfall region
            cv2.rectangle(self.frame_list[x], start, end,(0,255,0),-1)
            # Drawing all ByteTrack bbox
            for tracklet in self.targets[x]:
                # the top left bbox coordinates
                xmin_coord = int(tracklet._tlwh[0])
                ymin_coord = int(tracklet._tlwh[1])
                bbox_coord_start = (xmin_coord, ymin_coord)
                # the bottom right bbox coordinates
                xmax_coord = int(bbox_coord_start[0] + tracklet._tlwh[2])
                ymax_coord = int(bbox_coord_start[1] + tracklet._tlwh[3])
                bbox_coord_end = (xmax_coord, ymax_coord)
                cv2.rectangle(self.frame_list[x], bbox_coord_start, bbox_coord_end,(0,0,255),2)
            cv2.imshow("Frame",self.frame_list[x])
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

    def bytetrackconverter(self, results):
        # converts yolov5 output to bytetrack input
        df = results.pandas().xyxy[0] # yolov5 output as a dataframe
        # list of the processed input for tracking
        detections = []
        # xmin values
        xmin_vals = df['xmin'].tolist()
        # ymin values
        ymin_vals = df['ymin'].tolist()
        # xmax values
        xmax_vals = df['xmax'].tolist()
        # ymax values
        ymax_vals = df['ymax'].tolist()
        # confidence values
        conf_values = df['confidence'].tolist()
        # formatting values
        for x in range(len(df)):
            detections.append([xmin_vals[x],ymin_vals[x],xmax_vals[x],ymax_vals[x],conf_values[x]])
        return np.array(detections, dtype=float)

    def __del__(self):
        # object destructor
        self.model = None                                                   # yolov5 model
        self.vid_path = None                                                # video path
        self.frame_list = self.frame_list.clear()                           # list of frames/images in video
        self.targets = self.targets.clear()                                 # list of tracker outputs
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
    vid_handler.view_output_YOLOv5()
    #vid_handler.view_output_ByteTrack()
    del vid_handler
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()
    print("Execution Time:","%.3f" % (start.elapsed_time(end)/1000), "seconds")  # seconds
