# app.py
# Versi final dengan dashboard real-time, pencatatan latency, dan komentar penjelas.
# Kode HTML sekarang disematkan langsung di dalam file ini.

# --- Tahap 1: Impor Library yang Dibutuhkan ---
# Mengimpor semua "perkakas" yang diperlukan oleh program.
from flask import Flask, request, jsonify, send_file, render_template_string # Untuk membuat web server
import joblib  # Untuk memuat model machine learning (.pkl)
import numpy as np # Untuk mengelola data numerik (array)
import os # Untuk berinteraksi dengan sistem file (mengecek keberadaan file log)
from datetime import datetime # Untuk mendapatkan waktu saat ini
import csv # Untuk menulis data ke dalam file .csv
import time # Untuk menghitung latency
import pytz # Untuk menangani konversi zona waktu secara akurat

# --- Tahap 2: Inisialisasi Aplikasi Web ---
# Membuat "wadah" utama untuk aplikasi web kita menggunakan Flask.
# Variabel 'app' ini yang akan dijalankan oleh server Gunicorn di Render.
app = Flask(__name__)

# --- Tahap 3: Konfigurasi dan Variabel Global ---
# Mendefinisikan nama-nama file penting agar mudah diubah jika perlu.
MODEL_PATH = 'weather_model.pkl'
ENCODER_PATH = 'label_encoder.pkl'
LOG_FILE = 'log_prediksi.csv'

# --- Tahap 4: Kode HTML untuk Tampilan Dashboard ---
# Semua kode untuk tampilan visual dashboard disimpan dalam satu variabel string.
# Ini membuat proyek lebih ringkas karena tidak memerlukan file .html terpisah.
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard Prediksi Cuaca</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
        .card {
            background-color: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
</head>
<body class="bg-gray-900 text-white antialiased">
    <div class="min-h-screen flex flex-col items-center justify-center p-4">
        
        <div class="text-center mb-8">
            <h1 class="text-4xl font-bold tracking-tight">Dashboard Prediksi Cuaca</h1>
            <p class="text-gray-400 mt-2">Monitoring Real-time untuk GH TRB Padang</p>
        </div>

        <div class="w-full max-w-4xl p-6 rounded-2xl card">
            
            <div class="flex justify-between items-center border-b border-gray-700 pb-4 mb-6">
                <div class="flex items-center space-x-3">
                    <div id="status-dot" class="status-dot bg-yellow-400"></div>
                    <span id="status-text" class="font-medium text-gray-300">Menunggu data pertama...</span>
                </div>
                <div class="text-right">
                    <p class="text-sm text-gray-400">Update Terakhir (WIB)</p>
                    <p id="waktu" class="font-semibold text-lg">-</p>
                </div>
            </div>

            <div class="text-center mb-8">
                <p class="text-gray-400 text-lg">Prediksi Cuaca Saat Ini</p>
                <div id="prediksi-container" class="flex items-center justify-center space-x-4 mt-2">
                    <p id="prediksi" class="text-5xl font-bold">-</p>
                </div>
            </div>

            <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                <div class="p-4 rounded-lg bg-gray-800/50"><p class="text-sm text-gray-400">Suhu</p><p id="suhu" class="text-2xl font-semibold mt-1">-</p></div>
                <div class="p-4 rounded-lg bg-gray-800/50"><p class="text-sm text-gray-400">Kelembaban</p><p id="kelembaban" class="text-2xl font-semibold mt-1">-</p></div>
                <div class="p-4 rounded-lg bg-gray-800/50"><p class="text-sm text-gray-400">Kecepatan Angin</p><p id="angin" class="text-2xl font-semibold mt-1">-</p></div>
                <div class="p-4 rounded-lg bg-gray-800/50"><p class="text-sm text-gray-400">Tekanan Udara</p><p id="tekanan" class="text-2xl font-semibold mt-1">-</p></div>
            </div>
             <div class="text-center mt-6 pt-4 border-t border-gray-700">
                <p class="text-xs text-gray-500">Latensi Server (detik): <span id="latency" class="font-medium">-</span></p>
            </div>
        </div>
        
        <div class="mt-8 text-center text-gray-500 text-sm">
            <p>Dashboard ini akan refresh otomatis setiap 15 detik.</p>
            <a href="/log" class="underline hover:text-white mt-2 inline-block">Download Log Data (CSV)</a>
        </div>
    </div>

    <script>
        // Logika JavaScript untuk mengambil data terbaru dari server dan memperbarui tampilan dashboard.
        const weatherIcons = {
            'Cerah': '<svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12 text-yellow-300" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" /></svg>',
            'Berawan': '<svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z" /></svg>',
            'Hujan': '<svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12 text-blue-300" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 1v6m-4 4l-2 2m12-2l2 2M8 15l-2 2m12-2l2 2m-4-2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>',
            'Hujan Deras': '<svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 1v6m-4 4l-2 2m12-2l2 2M8 15l-2 2m12-2l2 2m-4-2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>'
        };

        async function fetchData() {
            try {
                const response = await fetch('/latest_data');
                if (!response.ok) throw new Error('Gagal mengambil data');
                const data = await response.json();
                if (data.error) throw new Error(data.error);

                document.getElementById('status-dot').className = 'status-dot bg-green-400';
                document.getElementById('status-text').textContent = 'Terhubung ke Server';
                
                const wibTime = new Date(data.waktu + "Z");
                document.getElementById('waktu').textContent = wibTime.toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit', second: '2-digit', timeZone: 'Asia/Jakarta' });

                document.getElementById('prediksi').textContent = data.prediksi;
                document.getElementById('suhu').textContent = `${parseFloat(data.suhu).toFixed(2)} °C`;
                document.getElementById('kelembaban').textContent = `${parseFloat(data.kelembaban).toFixed(2)} %`;
                document.getElementById('angin').textContent = `${parseFloat(data.angin).toFixed(2)} m/s`;
                document.getElementById('tekanan').textContent = `${parseFloat(data.tekanan).toFixed(2)} kPa`;
                document.getElementById('latency').textContent = `${data.latency} s`;

                const iconContainer = document.getElementById('prediksi-container');
                const existingIcon = iconContainer.querySelector('svg');
                if (existingIcon) existingIcon.remove();
                const iconHTML = weatherIcons[data.prediksi] || weatherIcons['Berawan'];
                iconContainer.insertAdjacentHTML('afterbegin', iconHTML);

            } catch (error) {
                console.error('Error:', error);
                document.getElementById('status-dot').className = 'status-dot bg-red-500';
                document.getElementById('status-text').textContent = 'Gagal terhubung';
            }
        }

        fetchData();
        setInterval(fetchData, 15000);
    </script>
</body>
</html>
"""

# --- Tahap 5: Memuat Model dan Encoder ---
# Mencoba memuat file .pkl saat server pertama kali dinyalakan.
model = None
label_encoder = None
try:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print(">>> Model dan encoder berhasil dimuat.")
except Exception as e:
    print(f"!!! KESALAHAN saat memuat model: {e}")

# --- Tahap 6: Mendefinisikan Rute (Endpoint) Aplikasi ---

# Rute untuk halaman utama
@app.route('/')
def index():
    # Menampilkan halaman sederhana dengan link ke dashboard dan log.
    return """
    <h1>API Sistem Prediksi Cuaca Aktif</h1>
    <p>Endpoint prediksi ada di <code>/predict</code>.</p>
    <p>Lihat dashboard real-time di <a href="/dashboard">/dashboard</a>.</p>
    <p>Download log data di <a href="/log">/log</a>.</p>
    """

# Rute utama untuk menerima data dari ESP32 dan membuat prediksi
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time() # Catat waktu mulai untuk menghitung latency
    
    # Cek apakah model sudah berhasil dimuat
    if not model or not label_encoder:
        return jsonify({'error': 'Model tidak tersedia.'}), 500

    try:
        # Ambil data JSON yang dikirim oleh ESP32
        iot_data = request.get_json(force=True)
        
        # Susun data menjadi format array yang bisa dibaca oleh model
        features = np.array([[
            iot_data['suhu'], iot_data['kelembaban'],
            iot_data['kecepatan_angin'], iot_data['tekanan_udara']
        ]])

        # Lakukan prediksi menggunakan model
        prediction_encoded = model.predict(features)
        prediction_text = label_encoder.inverse_transform(prediction_encoded)
        
        end_time = time.time() # Catat waktu selesai
        # Hitung latency dalam detik dengan 3 angka di belakang koma
        latency_s = f"{(end_time - start_time):.3f}"

        # Blok untuk menulis data ke file log CSV
        try:
            # Menggunakan pytz untuk konversi zona waktu yang akurat ke WIB
            utc_now = datetime.now(pytz.utc)
            wib_tz = pytz.timezone('Asia/Jakarta')
            wib_now = utc_now.astimezone(wib_tz)
            timestamp = wib_now.strftime("%Y-%m-%d %H:%M:%S")
            
            # Siapkan satu baris data untuk ditulis ke file log
            log_data = [
                timestamp, iot_data['suhu'], iot_data['kelembaban'],
                iot_data['kecepatan_angin'], iot_data['tekanan_udara'],
                prediction_text[0], latency_s
            ]
            
            # Cek apakah file log sudah ada (untuk menentukan perlu header atau tidak)
            file_exists = os.path.isfile(LOG_FILE)
            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    # Jika file baru, tulis header-nya dulu
                    header = ['Waktu (WIB)', 'Suhu', 'Kelembaban', 'Kecepatan Angin', 'Tekanan Udara', 'Prediksi', 'Latency (s)']
                    writer.writerow(header)
                # Tulis baris data log
                writer.writerow(log_data)
        except Exception as e:
            print(f"!!! Gagal menulis ke file log: {e}")
            
        # Siapkan dan kirim balasan ke ESP32
        response = {'prediksi_cuaca': prediction_text[0], 'data_diterima': iot_data}
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan saat prediksi: {str(e)}'}), 500

# Rute untuk menyediakan data terbaru ke dashboard
@app.route('/latest_data')
def latest_data():
    try:
        # Buka file log dan baca baris terakhir
        with open(LOG_FILE, 'r') as f:
            last_line = f.readlines()[-1]
            data = last_line.strip().split(',')
            # Kemas data baris terakhir menjadi format JSON
            json_data = {
                "waktu": data[0].replace(" ", "T"), 
                "suhu": data[1], "kelembaban": data[2],
                "angin": data[3], "tekanan": data[4], "prediksi": data[5], "latency": data[6]
            }
            return jsonify(json_data)
    except (FileNotFoundError, IndexError):
        # Jika file log belum ada atau kosong
        return jsonify({"error": "Belum ada data log."}), 404

# Rute untuk menampilkan halaman dashboard
@app.route('/dashboard')
def dashboard():
    # Mengembalikan kode HTML yang sudah disimpan di variabel HTML_TEMPLATE
    return render_template_string(HTML_TEMPLATE)

# Rute untuk men-download file log
@app.route('/log')
def get_log():
    try:
        # Mengirim file log_prediksi.csv sebagai attachment yang bisa di-download
        return send_file(LOG_FILE, as_attachment=True, download_name='log_prediksi.csv')
    except FileNotFoundError:
        return "File log belum dibuat. Kirimkan setidaknya satu data prediksi terlebih dahulu.", 404

# --- Tahap 7: Menjalankan Server ---
# Bagian ini hanya berjalan jika Anda menjalankan file ini langsung di komputer (untuk tes).
# Render akan menggunakan Gunicorn, bukan bagian ini.
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
