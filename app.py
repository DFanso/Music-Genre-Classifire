import numpy as np
import librosa
import tensorflow as tf
import cv2
import os
from pydub import AudioSegment
from flask import Flask, request, jsonify
from flask_cors import CORS



app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

def convert_audio_to_wav(audio_path, target_path=None):
    sound = AudioSegment.from_file(audio_path)
    wav_path = target_path if target_path else os.path.splitext(audio_path)[0] + '_converted.wav'
    sound.export(wav_path, format="wav")
    return wav_path


def extract_and_preprocess_features(audio_path, duration=30):
    wav_path = convert_audio_to_wav(audio_path)
    y, sr = librosa.load(wav_path, sr=None, mono=True, dtype=np.float32, duration=duration)

    # Spectrogram
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    if S.shape[1] > 1293:
        S = S[:, :1293]  # Crop columns if longer than 1293
    if S.shape[1] < 1293:
        padding_width = 1293 - S.shape[1]
        S = np.pad(S, ((0, 0), (0, padding_width)), 'constant', constant_values=(0))  # Pad with zeros
    spec = S.reshape(1, S.shape[0], S.shape[1], 1)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    mfcc_resized = cv2.resize(mfcc, (600, 120), interpolation=cv2.INTER_CUBIC)
    mfcc_normalized = ((mfcc_resized - np.mean(mfcc_resized)) / np.std(mfcc_resized)).reshape(1, 120, 600, 1)

    # Mel-Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_resized = cv2.resize(mel_db, (1293, 128), interpolation=cv2.INTER_CUBIC)
    mel_normalized = mel_resized.reshape(1, 128, 1293, 1)

    return spec, mfcc_normalized, mel_normalized


def get_majority(votes):
    return np.bincount(votes).argmax()


def predict_genre(audio_path):
    spec, mfcc, mel = extract_and_preprocess_features(audio_path)
    y_pred_spec = np.argmax(model_spectrogram.predict(spec), axis=-1)
    y_pred_mel = np.argmax(model_mel_spectrogram.predict(mel), axis=-1)

    # Predictions from the three new MFCC models
    y_pred_mfcc1 = np.argmax(model_mfcc1.predict(mfcc), axis=-1)
    y_pred_mfcc2 = np.argmax(model_mfcc2.predict(mfcc), axis=-1)
    y_pred_mfcc3 = np.argmax(model_mfcc3.predict(mfcc), axis=-1)

    # Ensemble prediction now includes the three MFCC models
    y_pred_ensemble = get_majority([y_pred_spec[0], y_pred_mel[0], y_pred_mfcc1[0], y_pred_mfcc2[0], y_pred_mfcc3[0]])

    num_to_genre = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop',
                    8: 'reggae', 9: 'rock'}
    predicted_genre = num_to_genre[y_pred_ensemble]
    return predicted_genre

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        audio_path = os.path.join(os.getcwd(), 'uploaded_audio', audio_file.filename)
        audio_file.save(audio_path)

        predicted_genre = predict_genre(audio_path)
        return jsonify({'predicted_genre': predicted_genre})

    except FileNotFoundError as e:
        return jsonify({'error': 'File not found', 'message': str(e)}), 404

    except Exception as e:
        return jsonify({'error': 'An error occurred during prediction', 'message': str(e)}), 500

if __name__ == "__main__":
    try:
        # Load the models, including the three new MFCC models
        model_spectrogram = tf.keras.models.load_model(
            os.path.join(os.getcwd(), "models", "new_spec_model_spectrogram1.h5"))
        model_mfcc1 = tf.keras.models.load_model(
            os.path.join(os.getcwd(), "models", "normalized_new_ensemble_classifier_mfcc1.h5"))
        model_mfcc2 = tf.keras.models.load_model(
            os.path.join(os.getcwd(), "models", "normalized_new_ensemble_classifier_mfcc2.h5"))
        model_mfcc3 = tf.keras.models.load_model(
            os.path.join(os.getcwd(), "models", "normalized_new_ensemble_classifier_mfcc3.h5"))
        model_mel_spectrogram = tf.keras.models.load_model(
            os.path.join(os.getcwd(), "models", "final_melspectrogram_model.h5"))

        app.run(host='0.0.0.0')

    except FileNotFoundError as e:
        print(f"Error: Model file not found - {str(e)}")

    except Exception as e:
        print(f"Error: An error occurred during model loading - {str(e)}")