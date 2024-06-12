import cv2
import os
import imutils
from PyQt5.QtWidgets import QApplication, QFileDialog, QInputDialog, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

class FaceRecognizer:
    def __init__(self, max_images_per_person=100):
        self.max_images_per_person = max_images_per_person
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def create_person_folder(self, person_name, data_path):
        person_path = os.path.join(data_path, person_name)
        if not os.path.exists(person_path):
            print('Carpeta creada:', person_path)
            os.makedirs(person_path)
        return person_path

    def capture_images(self, person_name, data_path, label):
        count = 0
        person_path = self.create_person_folder(person_name, data_path)

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: No se puede acceder a la cámara.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo obtener el fotograma.")
                break

            frame = imutils.resize(frame, width=640)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_roi = frame[y:y + h, x:x + w]
                face_roi = cv2.resize(face_roi, (150, 150), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(person_path, 'rostro_{}.jpg'.format(count)), face_roi)
                count += 1
                if count >= self.max_images_per_person:
                    cap.release()
                    cv2.destroyAllWindows()
                    return  # Termina la captura cuando se alcanza el límite

            if count >= self.max_images_per_person:
                break

def main():
    try:
        num_people = int(input("Ingrese el número de personas que desea registrar: "))
        recognizer = FaceRecognizer()

        # Usar QFileDialog para obtener la ruta de la carpeta Data
        app = QApplication([])
        data_path = QFileDialog.getExistingDirectory(None, "Seleccione la carpeta Data", ".", QFileDialog.ShowDirsOnly)

        # Configurar la ventana para mostrar los fotogramas
        window = QWidget()
        window.setWindowTitle('Captura de Rostros')
        layout = QVBoxLayout()
        label = QLabel()
        layout.addWidget(label)
        window.setLayout(layout)
        window.show()

        for i in range(num_people):
            person_name, _ = QInputDialog.getText(None, "Registro de Persona", "Ingrese el nombre de la persona {} a registrar: ".format(i + 1))
            recognizer.capture_images(person_name, data_path, label)
            print("Registro de {} completado.".format(person_name))

        app.exec_()

    except ValueError:
        print("Error: Ingrese un número válido para el número de personas.")

if __name__ == "__main__":
    main()
