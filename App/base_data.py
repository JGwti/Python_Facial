import cv2
import os
import imutils

personName ='Javier'
dataPath = 'D:\Desktop\Facial_Python\Data'
personPath = dataPath+'/'+personName

if not os.path.exists(personPath):
    print('Carpeta Creada:', personPath)
    os.makedirs(personPath)

cap =cv2.VideoCapture(0, cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier('D:\Desktop\Facial_Python\App\haarcascade_frontalface_default.xml')
count = 0

while True:

    ret, frame = cap.read()
    if ret == False:
        break
    frame = imutils.resize(frame, width=320)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)


    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0),2)
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro, (720, 720), interpolation= cv2.INTER_CUBIC )
        cv2.imwrite(personPath+'/Rostro_{}.jpg'.format(count),rostro)
        count = count + 1

    cv2.imshow('frame',frame)

    k = cv2.waitKey(1)
    if k == 27 or count ==300:
        break
cap.release()
cv2.destroyAllWindows()