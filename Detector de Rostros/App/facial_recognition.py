import cv2
import os

class FacialRecognition:
    def __init__(self):
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        self.dataPath = "ruta donde se almaceno trining"
        self.imagePaths = os.listdir(self.dataPath)
        # Leer el modelo
        self.face_recognizer.read('modeloLBPHFace.xml')

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceClassif.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            rostro = gray[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            result = self.face_recognizer.predict(rostro)
            confidence = result[1]
            if confidence < 70:
                label = self.imagePaths[result[0]]
            else:
                label = "Desconocido"
            cv2.putText(frame, str(label), (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return frame

    def generate_frames(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # or provide a video file path
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.detect_faces(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

