import cv2
import os
import numpy as np

def emotionImage(emotion):
    # Ajusta la ruta a .jpg en lugar de .jpeg
    path = f'Emojis/{emotion}.jpg'
    image = cv2.imread(path)
    if image is None:
        print(f"Error al cargar la imagen: {path}")
        return np.zeros((480, 300, 3), dtype=np.uint8)  # Imagen negra de relleno
    return image

def resize_image(image, height):
    aspect_ratio = image.shape[1] / image.shape[0]
    new_width = int(height * aspect_ratio)
    return cv2.resize(image, (new_width, height))

method = 'EigenFaces'
if method == 'EigenFaces': emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
if method == 'FisherFaces': emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
if method == 'LBPH': emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

emotion_recognizer.read('modelo'+method+'.xml')

dataPath = 'C:\\Users\\ASUS\\Documents\\Sustentacion IA\\PlanMejoramientoIA\\Deep learning\\ReconocimientoEmociones\\Data'
imagePaths = os.listdir(dataPath)
print('imagePaths=', imagePaths)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    height = 480
    nFrame = cv2.hconcat([frame, np.zeros((height, 300, 3), dtype=np.uint8)])

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = emotion_recognizer.predict(rostro)

        cv2.putText(frame, '{}'.format(result), (x, y-5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

        if method == 'EigenFaces':
            if result[1] < 5700:
                emotion = imagePaths[result[0]].split('.')[0]  # Obtener el nombre de la emoción sin la extensión
                cv2.putText(frame, '{}'.format(emotion), (x, y-25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                image = emotionImage(emotion)
                image_resized = resize_image(image, height)
                nFrame = cv2.hconcat([frame, image_resized])
            else:
                cv2.putText(frame, 'No identificado', (x, y-20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                nFrame = cv2.hconcat([frame, np.zeros((height, 300, 3), dtype=np.uint8)])
        # Repite la lógica para FisherFaces y LBPH

    cv2.imshow('nFrame', nFrame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
