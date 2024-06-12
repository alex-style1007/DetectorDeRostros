from flask import Flask, request, jsonify
from training import FaceRecognizer

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train_model():
    try:
        face_recognizer = FaceRecognizer()
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

if __name__ == "__main__":
    app.run(debug=True)
