import cv2
import os
import imutils
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

class FaceRecognizer:
    def __init__(self, max_images_per_person=300):
        self.max_images_per_person = max_images_per_person
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def create_person_folder(self, person_name, data_path):
        person_path = os.path.join(data_path, person_name)
        if not os.path.exists(person_path):
            print('Carpeta creada:', person_path)
            os.makedirs(person_path)
        return person_path

    def capture_images(self, person_name, data_path):
        count = 0
        person_path = self.create_person_folder(person_name, data_path)

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # cap = cv2.VideoCapture('Video.mp4')

        while True:
            ret, frame = cap.read()
            if ret == False:
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
                    return

            cv2.imshow('frame', frame)

            k = cv2.waitKey(1)
            if k == 27 or count >= self.max_images_per_person:
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    try:
        num_people = int(input("Ingrese el número de personas que desea registrar: "))
        recognizer = FaceRecognizer()

        # Usar QFileDialog para obtener la ruta de la carpeta Data
        app = QApplication([])
        data_path = QFileDialog.getExistingDirectory(None, "Seleccione la carpeta Data", ".", QFileDialog.ShowDirsOnly)

        for i in range(num_people):
            person_name, _ = QInputDialog.getText(None, "Registro de Persona", "Ingrese el nombre de la persona {} a registrar: ".format(i + 1))
            recognizer.capture_images(person_name, data_path)
            print("Registro de {} completado.".format(person_name))

    except ValueError:
        print("Error: Ingrese un número válido para el número de personas.")

if __name__ == "__main__":
    main()
