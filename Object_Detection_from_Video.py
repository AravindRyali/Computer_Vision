from ultralytics import YOLO
import cv2 as cv 

model = YOLO('yolov8n.pt')

#load the video file
video_path = '/Users/aravindryali/Desktop/Studies/Computer_Vision/odv.mp4'
cap = cv.VideoCapture(video_path)

ret = True
while ret:
    ret, frame = cap.read()
    if ret:
        results = model.track(frame, persist = True)
    # Draw bounding boxes and labels on the frame
    frame_ = results[0].plot()

    cv.imshow('Video Object Detection', frame_)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

    