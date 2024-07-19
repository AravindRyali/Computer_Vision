import cv2 as cv 
import mediapipe as mp
import os
 
out_dir = '/Users/aravindryali/Desktop/Studies/Computer_Vision'
img_path = '/Users/aravindryali/Desktop/Studies/Computer_Vision/AS.jpg'
img = cv.imread(img_path)
H, W, _ = img.shape

#face recognition
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection = 0, min_detection_confidence=0.5) as face_detection:
    
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)
    
    print(out.detections)

for detections in out.detections:
   location_data = detections.location_data
   bbox = location_data.relative_bounding_box
   x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
   
   x1, y1 = int(x1 * W), int(y1 * H)
   w, h = int(w * W), int(h * H)
   img = cv.rectangle(img, (x1, y1), (x1+w, y1+h), 10)
   
   
   img[y1:y1 + h, x1:x1 + w, :] = cv.blur(img[y1:y1 + h, x1:x1 + w, :],(100, 100))
cv.imwrite(os.path.join(out_dir, 'BluredImage.png'), img)
cv.imshow('Face Detection', img)
cv.waitKey(0)




