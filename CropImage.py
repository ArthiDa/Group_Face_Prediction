# import the modules
import cv2
import os
from os import listdir

cn = 1
# get the path/directory
folder_dir = "C:/Users/Hp/Desktop/AI Project/trainData/haiman"
for images in os.listdir(folder_dir):
    # check if the image ends with png
    file_name = folder_dir + "/" + images
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    print(file_name)
    # Read the image
    image = cv2.imread(file_name)
    tempImage = image
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        continue
    print(len(faces))
    for (x, y, w, h) in faces:
        crop_img = tempImage[y:y + h, x:x + w]
        new_file_name = f"img{cn}.jpg"
        cv2.imwrite(new_file_name, crop_img)
        cn = cn + 1

