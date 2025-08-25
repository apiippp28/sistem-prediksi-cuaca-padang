# app.py
# Versi final dengan dashboard real-time dan pencatatan latency.

from flask import Flask, request, jsonify, send_file, render_template_string
import joblib
import numpy as np
import os
from datetime import datetime
import csv
import time

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# --- Konfigurasi Path ---
MODEL_PATH = 'weather_model.pkl'
ENCODER_PATH = 'label_encoder.pkl'
LOG_FILE = 'log_prediksi.csv'

# --- Memuat Model dan Encoder ---
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
    """Halaman utama yang memberikan status dan link ke dashboard."""
    return """
    <h1>API Sistem Prediksi Cuaca Aktif</h1>
    <p>Endpoint prediksi ada di <code>/predict</code>.</p>
    <p>Lihat dashboard real-time di <a href="/dashboard">/dashboard</a>.</p>
    <p>Download log data di <a href="/log">/log</a>.</p>
    """

@app.route('/predict', methods=['POST'])
def predict():
    """Menerima data sensor, memberikan prediksi, dan mencatatnya ke file CSV."""
    start_time = time.time() # Catat waktu mulai proses

    if not model or not label_encoder:
        return jsonify({'error': 'Model tidak tersedia.'}), 500

    try:
        iot_data = request.get_json(force=True)
        features = np.array([[
            iot_data['suhu'],
            iot_data['kelembaban'],
            iot_data['kecepatan_angin'],
            iot_data['tekanan_udara']
        ]])

        prediction_encoded = model.predict(features)
        prediction_text = label_encoder.inverse_transform(prediction_encoded)
        
        # Hitung latency (waktu proses di server)
        end_time = time.time()
        latency_ms = round((end_time - start_time) * 1000)

        # Mencatat data ke file CSV
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_data = [
                timestamp, iot_data['suhu'], iot_data['kelembaban'],
                iot_data['kecepatan_angin'], iot_data['tekanan_udara'],
                prediction_text[0], latency_ms # Menambahkan latency ke log
            ]
            
            file_exists = os.path.isfile(LOG_FILE)
            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    header = ['Waktu', 'Suhu', 'Kelembaban', 'Kecepatan Angin', 'Tekanan Udara', 'Prediksi', 'Latency (ms)']
                    writer.writerow(header)
                writer.writerow(log_data)
        except Exception as e:
            print(f"!!! Gagal menulis ke file log: {e}")
            
        response = {
            'prediksi_cuaca': prediction_text[0],
            'data_diterima': iot_data
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan saat prediksi: {str(e)}'}), 500

# --- Endpoint Baru untuk Dashboard ---

@app.route('/latest_data')
def latest_data():
    """Menyediakan data log terakhir dalam format JSON untuk dashboard."""
    try:
        with open(LOG_FILE, 'r') as f:
            # Membaca semua baris dan mengambil yang terakhir
            last_line = f.readlines()[-1]
            data = last_line.strip().split(',')
            json_data = {
                "waktu": data[0],
                "suhu": data[1],
                "kelembaban": data[2],
                "angin": data[3],
                "tekanan": data[4],
                "prediksi": data[5],
                "latency": data[6]
            }
            return jsonify(json_data)
    except (FileNotFoundError, IndexError):
        return jsonify({"error": "Belum ada data log."}), 404

@app.route('/dashboard')
def dashboard():
    """Menampilkan halaman dashboard HTML."""
    # Kode HTML dan JavaScript untuk dashboard ada di sini
    # Ini membuat kita tidak perlu file HTML terpisah
    return render_template_string(open('dashboard.html').read())

@app.route('/log')
def get_log():
    """Menyediakan file log_prediksi.csv untuk di-download."""
    try:
        return send_file(LOG_FILE, as_attachment=True, download_name='log_prediksi.csv')
    except FileNotFoundError:
        return "File log belum dibuat. Kirimkan setidaknya satu data prediksi terlebih dahulu.", 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
