import cv2 as cv
import matplotlib.pyplot as plt
import easyocr

img_path = '/Users/aravindryali/Desktop/Studies/Computer_Vision/image_with_text.jpeg'

# Load the image
img = cv.imread(img_path)

# Convert the image to grayscale    
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#instance of text detector
reader = easyocr.Reader(['en'])

# Detect and read text from image
text_ = reader.readtext(gray)

for line in text_:
    
    print(line)
    
    bbox, text, score = line
    
    cv.rectangle(img, bbox[0], bbox[2], (0, 255,0), 1)
    cv.putText(img, text, bbox[0], cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
   

plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.show()   

    

