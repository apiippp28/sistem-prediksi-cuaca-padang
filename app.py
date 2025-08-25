# app.py
# Versi baru dengan fungsionalitas pencatatan (logging) otomatis ke file CSV.

from flask import Flask, request, jsonify, send_file
import joblib
import numpy as np
import os
from datetime import datetime
import csv

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# --- Memuat Model dan Encoder ---
MODEL_PATH = 'weather_model.pkl'
ENCODER_PATH = 'label_encoder.pkl'
LOG_FILE = 'log_prediksi.csv'

model = None
label_encoder = None

try:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print(">>> Model dan encoder berhasil dimuat.")
except Exception as e:
    print(f"!!! KESALAHAN saat memuat model: {e}")

# --- Endpoint Aplikasi ---

@app.route('/')
def index():
    """Halaman utama yang memberikan status API."""
    return "API Sistem Prediksi Cuaca aktif. Endpoint prediksi ada di /predict. Log data ada di /log."

@app.route('/predict', methods=['POST'])
def predict():
    """Menerima data sensor, memberikan prediksi, dan mencatatnya ke file CSV."""
    if not model or not label_encoder:
        return jsonify({'error': 'Model tidak tersedia.'}), 500

    try:
        iot_data = request.get_json(force=True)
        
        # Susun data untuk prediksi
        features = np.array([[
            iot_data['suhu'],
            iot_data['kelembaban'],
            iot_data['kecepatan_angin'],
            iot_data['tekanan_udara']
        ]])

        # Lakukan prediksi
        prediction_encoded = model.predict(features)
        prediction_text = label_encoder.inverse_transform(prediction_encoded)
        
        # --- FUNGSI BARU: Mencatat data ke file CSV ---
        try:
            # Dapatkan waktu saat ini dengan format yang jelas
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Siapkan data untuk ditulis ke log
            log_data = [
                timestamp,
                iot_data['suhu'],
                iot_data['kelembaban'],
                iot_data['kecepatan_angin'],
                iot_data['tekanan_udara'],
                prediction_text[0]
            ]
            
            # Cek apakah file log sudah ada untuk menentukan perlu header atau tidak
            file_exists = os.path.isfile(LOG_FILE)
            
            # Buka file dalam mode 'append' (menambah baris baru)
            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                # Jika file baru dibuat, tulis header-nya dulu
                if not file_exists:
                    header = ['Waktu', 'Suhu', 'Kelembaban', 'Kecepatan Angin', 'Tekanan Udara', 'Prediksi']
                    writer.writerow(header)
                # Tulis baris data log
                writer.writerow(log_data)

        except Exception as e:
            print(f"!!! Gagal menulis ke file log: {e}")
            
        # Kirim balasan seperti biasa
        response = {
            'prediksi_cuaca': prediction_text[0],
            'data_diterima': iot_data
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan saat prediksi: {str(e)}'}), 500

# --- ENDPOINT BARU UNTUK DOWNLOAD LOG ---
@app.route('/log', methods=['GET'])
def get_log():
    """Menyediakan file log_prediksi.csv untuk di-download."""
    try:
        return send_file(LOG_FILE, as_attachment=True, download_name='log_prediksi.csv')
    except FileNotFoundError:
        return "File log belum dibuat. Kirimkan setidaknya satu data prediksi terlebih dahulu.", 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
