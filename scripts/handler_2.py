import torch
import cv2
import time
import numpy as np
from yolox.tracker.byte_tracker import BYTETracker
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def script():
    # ByteTrack args class
    class bytetrackerargs():
        track_thresh: float = 0.25                  # tracking threshold
        track_buffer: int = 30                      # frames in buffer
        match_thresh: float = 0.8                   # matching threshold
        aspect_ratio_thresh: float = 3.0
        min_box_area: float = 1.0                   # smallest possible bbox
        mot20: bool = False                         # not using mot20

    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model.to(device)
    model.classes = [0]

    print("Starting Video Processing...")
    prev_time = 0.0
    new_frame_time = 0.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    # coordinates for the marked region (footfall counter)
    start = (620,500)
    end = (1000,650)
    # video capture object
    cap = cv2.VideoCapture('./adlytic_videos/footfall.mp4')
    tracker = BYTETracker(bytetrackerargs) # byte tracker object
    # capturing first frame to check whether video exists for processing below
    ret, frame = cap.read()
    # video processing loop
    while ret:
        # inference + tracking
        results = model(frame, size=640)
        detections = bytetrackconverter(results)
        # results = np.array(results.render()) # selecting the frame from the inferenced output (YOLOv5 Detection class)
        online_targets = tracker.update(detections, (640,640), (640,640)) # tracker output
        new_frame_time = time.time()
        fps = 'FPS: ' + str(int(1/(new_frame_time-prev_time)))
        # FPS text
        cv2.putText(frame, fps, (7, 70), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
        # Footfall region
        cv2.rectangle(frame, start, end,(0,255,0),-1)
        for tracklet in online_targets:
                # the top left bbox coordinates
                xmin_coord = int(tracklet._tlwh[0])
                ymin_coord = int(tracklet._tlwh[1])
                bbox_coord_start = (xmin_coord, ymin_coord)
                # the bottom right bbox coordinates
                xmax_coord = int(bbox_coord_start[0] + tracklet._tlwh[2])
                ymax_coord = int(bbox_coord_start[1] + tracklet._tlwh[3])
                bbox_coord_end = (xmax_coord, ymax_coord)
                trackletID = "ID: " + str(tracklet.track_id)
                cv2.putText(frame, trackletID, (xmin_coord, ymin_coord - 2), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.rectangle(frame, bbox_coord_start, bbox_coord_end,(0,0,255),2)
        cv2.imshow("ByteTrack Output" , frame)
        prev_time = new_frame_time
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
    print("Inferencing Completed!")

def bytetrackconverter(results):
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


# Main
if __name__ == "__main__":
    script()
