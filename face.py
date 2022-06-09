import cv2


cap = cv2.VideoCapture("images/video.MOV")


while True:
    success, img = cap.read()

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    fases = cv2.CascadeClassifier('face.xml')

    results = fases.detectMultiScale(grey, scaleFactor=2, minNeighbors=6)

    for (x, y, w, h) in results:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=3)

    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
