import torch
import cv2
import argparse
import time
import pandas
import json
import numpy as np
from yolox.tracker.byte_tracker import BYTETracker
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# For RTSP
# rtsp://admin:hik12345@192.168.10.240:554
USERNAME = 'admin'
PASSWORD = 'hik12345'
IP = '192.168.10.240'
ENDPOINT = '554'

# ByteTrack args class
class bytetrackerargs():
    track_thresh: float = 0.25                  # tracking threshold
    track_buffer: int = 30                      # frames in buffer
    match_thresh: float = 0.8                   # matching threshold
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0                   # smallest possible bbox
    mot20: bool = False                         # not using mot20

# yolov5 handler for object detection in videos
# handler class
class handler():
    def __init__(self, vid):
        self.model = None                       # yolov5 model
        self.inside_ids = set()                 # set to store tracking ids of entered tracked objects
        self.vid_path = vid                     # video path

    def load_model(self):
        # loading pytorch yolov5 model for inference
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        # shifting model to GPU
        self.model.to(device)
        # for running inference on the person class only
        self.model.classes = [0]

    def frame_processing(self):
        print("Starting Video Processing...")
        # fps calculators
        prev_time = 0.0
        new_frame_time = 0.0
        font = cv2.FONT_HERSHEY_SIMPLEX
        # coordinates for the exit region
        start_exit = (620,400)
        end_exit = (1000,550)
        # coordinates for the entrance region
        start_entrance = (620,650)
        end_entrance = (1000,800)
        # video capture object
        #rtsp_cap = cv2.VideoCapture(f'rtsp://{USERNAME}:{PASSWORD}@{IP}/{ENDPOINT}')
        cap = cv2.VideoCapture(self.vid_path)
        # byte tracker object
        tracker = BYTETracker(bytetrackerargs)
        frame_count = 0
        entrances = 0
        exits = 0
        # video processing loop
        while cap.isOpened():
            # capturing first frame to check whether video exists for processing below
            ret, frame = cap.read()
            # if no frame exists, simply end the capturing process
            if not ret:
                break
            frame_count +=1
            # inference + tracking
            results = self.model(frame, size=640)
            detections = self.bytetrackconverter(results)
            '''
            Uncomment the lines below for:
            1)  JSON file output
            2)  YOLOv5 Output
            '''
            # converting each frame to a JSON object for the JSON file
            # self.write_output(frame_count, results.pandas().xyxy[0].to_json(orient='records'))
            # results = np.array(results.render()) # selecting the frame from the inferenced output (YOLOv5 Detection class Output)
            online_targets = tracker.update(detections, (640,640), (640,640))
            new_frame_time = time.time()
            fps = 'FPS: ' + str(int(1/(new_frame_time-prev_time)))
            entrance_counter = 'Entrances:' + str(entrances)
            exit_counter = 'Exits:' + str(exits)
            # FPS text
            cv2.putText(frame, fps, (7, 70), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
            # Counters
            cv2.putText(frame, entrance_counter, (8, 120), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, exit_counter, (8, 170), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
            # Exit region
            cv2.rectangle(frame, start_exit, end_exit,(0,0,255),-1)
            cv2.putText(frame, "Exit", (770, 475), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
            # Entrance region
            cv2.rectangle(frame, start_entrance, end_entrance,(0,255,0),-1)
            cv2.putText(frame, "Entrance", (740, 725), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
            for tracklet in online_targets:
                    # the top left bbox coordinates
                    xmin_coord = int(tracklet.tlwh[0])
                    ymin_coord = int(tracklet.tlwh[1])
                    bbox_coord_start = (xmin_coord, ymin_coord)
                    # the bottom right bbox coordinates
                    xmax_coord = int(bbox_coord_start[0] + tracklet.tlwh[2])
                    ymax_coord = int(bbox_coord_start[1] + tracklet.tlwh[3])
                    bbox_coord_end = (xmax_coord, ymax_coord)
                    trackletID = "ID:" + str(tracklet.track_id)
                    # adding the track id to the respective set
                    if ymax_coord > end_entrance[1]:
                        self.inside_ids.add(tracklet.track_id)
                    # adding tracking ID to object
                    cv2.putText(frame, trackletID, (xmin_coord, ymin_coord - 2), font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    # drawing bbox
                    cv2.rectangle(frame, bbox_coord_start, bbox_coord_end,(255,0,0),2)
                    # calculating the bottom mid x coordinate
                    botxcoord = (xmin_coord+xmax_coord)/2.0
                    # counting logic
                    if tracklet.track_id not in self.inside_ids:
                        # bounding the vertical coloumn for the designated region
                        if botxcoord > start_entrance[0] and botxcoord < end_entrance[0]:
                            if ymax_coord < end_entrance[1] and ymax_coord > start_entrance[1]:
                                entrances += 1
                                self.inside_ids.add(tracklet.track_id)  
                    if tracklet.track_id in self.inside_ids:
                        # bounding the vertical coloumn for the designated region
                        if botxcoord > start_entrance[0] and botxcoord < end_entrance[0]:
                            if ymax_coord < end_exit[1] and ymax_coord > start_exit[1]:
                                exits += 1
                                self.inside_ids.remove(tracklet.track_id)
            # visualizing output frame
            cv2.imshow("ByteTrack Output" , frame)
            '''Remove the statement below(only for testing purposes)'''
            # cv2.imwrite('frame.jpg',frame)
            prev_time = new_frame_time
            cv2.waitKey(1)
        # release the video capture object
        cap.release()
        # Closes all the windows currently opened.
        cv2.destroyAllWindows()
        print("Video Processing Completed!")
        print("Inferencing Completed!")
        print('Number of frames: ', frame_count)
    
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
        self.inside_ids = self.inside_ids.clear()                           # set to store tracking ids of entered tracked objects
        self.vid_path = None                                                # video path
        print("Handler destructor invoked!")

# main function
if __name__ == '__main__':
    # Argument from CLI
    parser = argparse.ArgumentParser(description = 'I/O file paths required.')
    parser.add_argument('-vid_path', type = str, dest = 'vid_path', required =True)
    args = parser.parse_args()

    # For calculating execution time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    # whatever you are timing goes here
    vid_handler = handler(args.vid_path)
    vid_handler.load_model()
    vid_handler.frame_processing()
    del vid_handler
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()
    print("Execution Time:","%.3f" % (start.elapsed_time(end)/1000), "seconds")  # seconds
