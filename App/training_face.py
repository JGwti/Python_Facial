import cv2
import os
import numpy as np

dataPath = 'D:\Desktop\Facial_Python\Data'
peopleList = os.listdir(dataPath)
print('Lista de personas:' , peopleList)

labels = []
faceData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath+'/'+nameDir
    print('Leyendo Imagenes')

    for fileName in os.listdir(personPath):
        print('Rostros: ', nameDir+'/'+fileName)
        labels.append(label)

        faceData.append(cv2.imread(personPath+'/'+fileName, 0))
        image = cv2.imread(personPath+'/'+fileName, 0)

        #cv2.imshow('image',image)
        #cv2.waitKey(10)

    label = label + 1

#cv2.destroyAllWindows()

#print('labels = ',labels)
#print('Numero de etiquetas 0:', np.count_nonzero(np.array(labels)==0))
#print('Numero de etiquetas 0:', np.count_nonzero(np.array(labels)==1))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

print("Entrenando...")
face_recognizer.train(faceData, np.array(labels))

face_recognizer.write('ModeloFaceFrontalData2023.xml')
print("Modelo guardado")


