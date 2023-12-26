import librosa
from flask import Flask, request, send_from_directory, render_template, jsonify
import os

import numpy as np
import joblib

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        filename = 'audio.wav'
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        return 'File uploaded successfully', 200

@app.route('/audio')
def serve_audio():
    return send_from_directory(UPLOAD_FOLDER, 'audio.wav')

import librosa

@app.route('/detect_tempo', methods=['GET'])
def detect_tempo():
    try:
        y, sr = librosa.load(os.path.join(UPLOAD_FOLDER, 'audio.wav'))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return {'tempo': tempo}
    except Exception as e:
        return str(e), 500

@app.route('/process_audio', methods=['GET'])
def process_audio():
    try:
        # Load audio
        y, sr = librosa.load(os.path.join(UPLOAD_FOLDER, 'audio.wav'))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        beats_per_bar = request.args.get('beats_per_bar', default=4, type=int)
        bar_length = sr * 60 / tempo * beats_per_bar

        predictions = []
        for i in range(0, len(y), int(bar_length)):
            bar = y[i:i + int(bar_length)]

            # Feature extraction
            chromagram_cqt = librosa.feature.chroma_cqt(y=bar, sr=sr)
            if len(chromagram_cqt[0]) >= 100:
                chromagram_cqt = chromagram_cqt[:, :100]
            elif len(chromagram_cqt[0]) < 100:
                chromagram_cqt = np.pad(chromagram_cqt, ((0, 0), (0, 100 - chromagram_cqt.shape[1])), mode='constant', constant_values=0)
            chromagram_cqt = chromagram_cqt.flatten()

            # Load the trained model (assumed to be in the same folder)
            svc = joblib.load('model.pkl')

            # Make prediction
            y_pred = svc.predict([chromagram_cqt])
            predictions.append(y_pred[0])

        return jsonify(predictions)
    except Exception as e:
        return str(e), 500


if __name__ == '__main__':
    app.run(debug=True)
