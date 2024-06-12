# Aplicacion de deteccion de Rostros usando python,opencv y Flask
El objetivo principal de este proyecto es realizar una aplicacion que permita capturar imagenes, entrenar y utilizar un modelo de deteccion de rostros usando opencv

## Capture
Este módulo se enfoca en la captura de imágenes. Imagina que eres una empresa y necesitas utilizar la detección de rostros para autorizar o denegar el acceso de tus empleados a áreas específicas. Este módulo permite ingresar el nombre de la persona y simultáneamente captura las imágenes de dicho individuo.

## Training
Este módulo permite seleccionar las imágenes con las cuáles entrenaras el módelo, luego de esto el módelo sabra que personas estan registradas y cuáles no.

## App/facial_recognition
Basicamente toma el modelo entrenado para poder implementarlo

## run

ChatGPT
Este script utiliza el framework Flask de Python para implementar un servicio de reconocimiento facial. La aplicación consta de tres rutas principales: /capture, /train y /video_feed.

La ruta /capture está diseñada para capturar imágenes faciales de personas específicas. Cuando se envía una solicitud POST a esta ruta, se instancia un objeto FaceRecognizer, se recibe el número de personas a capturar y la ruta donde se almacenarán las imágenes. Luego, se itera sobre el número de personas y se capturan sus imágenes utilizando el método capture_images() del objeto FaceRecognizer. Si ocurre algún error durante el proceso, se maneja y se devuelve un mensaje de error correspondiente.

La ruta /train se utiliza para entrenar el modelo de reconocimiento facial. Al recibir una solicitud POST, se instancia un objeto FaceTrainer, se lee la ruta del directorio de datos desde la solicitud y se utiliza para leer las imágenes. Luego, se entrena el reconocedor facial y se guarda el modelo resultante en un archivo XML. Si hay algún error durante el proceso, se devuelve un mensaje de error junto con el código de estado correspondiente.

La ruta /video_feed proporciona un flujo de video en tiempo real que muestra el proceso de reconocimiento facial. Esta ruta devuelve un objeto Response con el contenido del flujo de video, generado por el método generate_frames() de un objeto FacialRecognition.

Finalmente, el script verifica si se está ejecutando directamente como un script principal y, en ese caso, inicia la aplicación Flask en modo de depuración.

Este conjunto de rutas y funciones permite la captura de imágenes faciales, el entrenamiento del modelo de reconocimiento facial y la transmisión en tiempo real del proceso de reconocimiento facial a través de una interfaz web.

```python
from flask import Flask, request, jsonify,Response
from training import FaceTrainer
from capture import FaceRecognizer
from App import FacialRecognition

app = Flask(__name__)
facial_recognition = FacialRecognition()

@app.route('/capture', methods=['POST'])
def capture():
    recognizer = FaceRecognizer()
    num_people = int(request.form['num_people'])
    data_path = './data'  # Ruta donde se almacenarán las imágenes
    try:
        for i in range(num_people):
            person_name = request.form['person_name_{}'.format(i)]
            recognizer.capture_images(person_name, data_path, None)
            print("Registro de {} completado.".format(person_name))
        return 'Captura de imágenes completada exitosamente.'
    except ValueError:
        return 'Error: Ingrese un número válido para el número de personas.'

@app.route('/train', methods=['POST'])
def train_model():
    try:
        face_recognizer = FaceTrainer()
        data_path = request.json.get('data_path')
        if not data_path:
            return jsonify({'error': 'La ruta del directorio de datos no está especificada'}), 400
        face_recognizer.data_path = data_path
        face_recognizer.read_images()
        recognizer = face_recognizer.train_recognizer()
        model_path = 'modeloLBPHFace.xml'
        face_recognizer.save_model(recognizer, model_path)
        return jsonify({'message': 'Modelo entrenado y almacenado correctamente'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    return Response(facial_recognition.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == "__main__":
    app.run(debug=True)
```