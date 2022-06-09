import cv2
filename = "./images/famil.jpg"
haarcascades = cv2.data.haarcascades

def detect():
    face_cascade = cv2.CascadeClassifier(haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(haarcascades + "haarcascade_eye.xml")
    camera = cv2.VideoCapture('images/video.MOV')
    while True:
        ret, frame = camera.read()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        face = face_cascade.detectMultiScale(gray, 1.2, 6)
    
        for (x, y, w, h) in face:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=3)
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.05, 4, 0, (40, 40))
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(img, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0,255,0), 2)

        cv2.imshow("camera", frame)
        
        if cv2.waitKey(int(1000/12)) & 0xff == ord("q"):
            break
    
    camera.release()
    cv2.destroyAllWindows()
detect()
