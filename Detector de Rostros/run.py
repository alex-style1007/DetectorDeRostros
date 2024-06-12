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
