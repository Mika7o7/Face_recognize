import numpy as np
import cv2
import pickle


haarcascades = cv2.data.haarcascades

face_cascade = cv2.CascadeClassifier(haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person_name": 1}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}


camera = cv2.VideoCapture('images/video.MOV')
count = 0

while True:
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    for x,y,w,h in face:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 25:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)       
        img_item = "my_image.pgm"        
        cv2.imwrite(img_item, roi_gray)
        
        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        
    
    cv2.imshow("camera", frame)
    if cv2.waitKey(20) & 0xff==ord("q"):
        break
    
    
camera.release()
cv2.destroyAllWindows()