import cv2
import numpy as np
from PIL import Image

def get_limits(colour):
    c = np.uint8([[colour]])  # BGR values
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    hue = hsvC[0][0][0]  # Get the hue value

    # Define the range for green hue
    lowerLimit = np.array([hue - 30, 100, 100], dtype=np.uint8)
    upperLimit = np.array([hue + 30, 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit

# Define the green color in BGR colorspace
green = [0, 255, 0]

# Get the HSV limits for the green color
lowerLimit, upperLimit = get_limits(green)
# Define the yellow color in BGR colorspace
green = [0, 255, 0]

# Initialize video capture (0 is usually the default camera, use 1 if you have multiple cameras)
cap = cv2.VideoCapture(0)

# Check if the video capture is initialized properly
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Check if frame is read correctly
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to HSV colorspace
    try:
        hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    except cv2.error as e:
        print(f"Error converting frame to HSV: {e}")
        continue

    # Get the color limits for yellow
    lowerLimit, upperLimit = get_limits(colour=green)

    # Create a mask for the yellow color in the frame
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    # Convert the mask to a PIL image for bounding box extraction
    mask_ = Image.fromarray(mask)

    # Get the bounding box of the mask
    bbox = mask_.getbbox()

    # If a bounding box is found, draw a rectangle on the frame
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Display the frame with the rectangle
    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()