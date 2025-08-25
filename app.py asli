# app.py
# Aplikasi web (API) yang akan di-deploy ke server Render.
# Sesuai arsitektur pada Bab I dan Bab III dokumen penelitian Anda.

# Mengimpor library yang dibutuhkan
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# --- Inisialisasi Aplikasi Flask ---
# Baris ini sangat penting. 
# Ini membuat variabel 'app' yang akan dicari oleh server Gunicorn.
# Perintah 'gunicorn app:app' artinya: jalankan file 'app.py' dan cari variabel 'app'.
app = Flask(__name__)

# --- Memuat Model dan Encoder ---
# Model dan encoder dimuat hanya sekali saat server pertama kali dijalankan.
# Ini membuat proses prediksi menjadi sangat cepat.

MODEL_PATH = 'weather_model.pkl'
ENCODER_PATH = 'label_encoder.pkl'

model = None
label_encoder = None

try:
    # Memuat model machine learning yang sudah dilatih
    model = joblib.load(MODEL_PATH)
    # Memuat encoder untuk mengubah prediksi angka menjadi teks (misal: 2 -> 'Hujan')
    label_encoder = joblib.load(ENCODER_PATH)
    print(">>> Model dan encoder berhasil dimuat. Server siap menerima permintaan.")
except FileNotFoundError:
    print(f"!!! KESALAHAN: File model '{MODEL_PATH}' atau '{ENCODER_PATH}' tidak ditemukan.")
    print("!!! Pastikan Anda sudah menjalankan 'train_model.py' dan file .pkl ada di repositori.")
except Exception as e:
    print(f"!!! Terjadi kesalahan saat memuat model: {e}")


# --- Endpoint (URL) Aplikasi ---

# Endpoint utama untuk mengecek apakah server berjalan
@app.route('/')
def index():
    """Halaman utama yang memberikan status API."""
    return "Selamat Datang di API Sistem Prediksi Cuaca GH TRB Padang. Gunakan endpoint /predict untuk membuat prediksi."

# Endpoint untuk prediksi, hanya menerima metode POST
@app.route('/predict', methods=['POST'])
def predict():
    """Menerima data sensor dari ESP32 dan mengembalikan prediksi cuaca."""
    
    # 1. Cek apakah model sudah siap
    if model is None or label_encoder is None:
        return jsonify({'error': 'Model tidak tersedia. Silakan cek log server.'}), 500

    # 2. Ambil data JSON yang dikirim oleh perangkat IoT
    try:
        iot_data = request.get_json(force=True)
    except Exception as e:
        return jsonify({'error': 'Request body tidak valid atau bukan format JSON.', 'details': str(e)}), 400

    # 3. Lakukan Prediksi
    try:
        # Susun data menjadi array numpy sesuai urutan fitur saat pelatihan.
        # PENTING: Urutan harus sama persis dengan skrip training.
        # Fitur yang digunakan: Suhu, Kelembaban, Kecepatan Angin, Tekanan Udara
        features = np.array([[
            iot_data['suhu'],
            iot_data['kelembaban'],
            iot_data['kecepatan_angin'],
            iot_data['tekanan_udara']
        ]])

        # Lakukan prediksi menggunakan model yang sudah dimuat
        prediction_encoded = model.predict(features)

        # Terjemahkan hasil prediksi (yang berupa angka) kembali menjadi teks
        prediction_text = label_encoder.inverse_transform(prediction_encoded)

        # 4. Kirim Hasil Prediksi
        # Hasil ini nantinya bisa digunakan oleh ESP32 untuk memicu notifikasi Telegram.
        response = {
            'prediksi_cuaca': prediction_text[0],
            'data_diterima': iot_data
        }
        
        return jsonify(response)

    except KeyError as e:
        # Error jika ada field yang hilang dari data JSON
        return jsonify({'error': f'Data tidak lengkap. Field yang wajib ada: {e}'}), 400
    except Exception as e:
        # Error umum lainnya
        return jsonify({'error': f'Terjadi kesalahan saat proses prediksi: {str(e)}'}), 500

# Bagian ini untuk menjalankan server saat pengujian di komputer lokal
if __name__ == '__main__':
    # Render akan menggunakan Gunicorn, tapi ini berguna untuk tes lokal.
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
