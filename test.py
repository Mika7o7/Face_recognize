import cv2 
import numpy as np 
import sys 
import os

names = ['mika', 'isay'] # имя лица, соответствующего изображению
if len(sys.argv) < 2:
    print("USAGE: xxx.py </path/to/images> [</path/to/store/images/at>]")
    sys.exit()

[X,y] = read_images(sys.argv[1])
print(sys.argv[1])
# print(y)
# print(X)
y = np.asarray(y, dtype=np.int32)

if len(sys.argv) == 3:
    out_dir = sys.argv[2]

        # Вот три различных метода идентификации, которые основаны на Eigenfaces, Fisherfaces и LBPH.
model = cv2.face.EigenFaceRecognizer_create()#cv2.face.createEigenFaceRecognizer()
# model = cv2.face.FisherFaceRecognizer_create()#cv2.face.createFisherFaceRecognizer()
# model = cv2.face.LBPHFaceRecognizer_create()#cv2.face.createLBPHFaceRecognizer()

model.train(np.asarray(X), np.asarray(y))
camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_default.xml')

while (True):
    read, img = camera.read()
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        roi = gray[x:x+w, y:y+h]
        try:
            roi = cv2.resize(roi, (200,200), interpolation=cv2.INTER_LINEAR)
            params = model.predict(roi)
            print("Label:%s, Confidence:%.2f" % (params[0], params[1]))
            cv2.putText(img, names[params[0]], (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        except:
            continue
    cv2.imshow('camera', img)
    if cv2.waitKey(int(1000/12)) & 0xff == ord('q'):
        break
cv2.destroyAllWindows()