# Sistem Prediksi Cuaca GH TRB Padang

Proyek Tugas Akhir ini adalah sebuah sistem prediksi cuaca berbasis *machine learning* yang dirancang khusus untuk lingkungan Gardu Hubung (GH) TRB Padang. Sistem ini menggunakan data sensor lokal untuk memprediksi kondisi cuaca dan memberikan peringatan dini.

## 🤖 Arsitektur Sistem

Sistem ini bekerja dengan alur sebagai berikut:
1.  **Akuisisi Data**: Perangkat keras (ESP32 dengan sensor BME280 & Anemometer) mengumpulkan data suhu, kelembaban, tekanan udara, dan kecepatan angin.
2.  **Pengiriman Data**: Data dikirim melalui protokol HTTP ke sebuah API yang di-hosting di server Render.
3.  **Prediksi**: API menerima data, lalu menggunakannya sebagai input untuk model *machine learning* (Decision Tree) yang sudah dilatih.
4.  **Hasil**: Model memberikan hasil prediksi berupa salah satu dari 4 kategori: **Cerah, Berawan, Hujan,** atau **Hujan Deras**.
5.  **Notifikasi**: Hasil prediksi ini kemudian dapat digunakan untuk memicu sistem notifikasi otomatis melalui Telegram.

---

## 📂 Struktur File

-   `app.py`: Aplikasi web (API) utama yang di-deploy ke Render. Bertugas menerima data dan memberikan prediksi.
-   `train_model.py`: Skrip untuk melatih model Decision Tree menggunakan data historis dari BMKG.
-   `requirements.txt`: Daftar library Python yang dibutuhkan agar proyek bisa berjalan.
-   `weather_model.pkl`: File model *machine learning* yang sudah dilatih.
-   `label_encoder.pkl`: "Kamus" untuk mengubah label prediksi dari angka ke teks.
-   `data_bmkg_bersih.csv`: Dataset historis dari BMKG yang digunakan untuk melatih model.

---

## 🚀 Cara Menjalankan Lokal (Testing)

Untuk menguji API di komputer Anda sebelum deploy:

1.  **Install semua library yang dibutuhkan:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Jalankan server API:**
    ```bash
    python app.py
    ```

3.  **Kirim permintaan tes dari terminal lain:**
    ```bash
    curl -X POST [http://127.0.0.1:8080/predict](http://127.0.0.1:8080/predict) \
    -H "Content-Type: application/json" \
    -d '{
        "suhu": 28,
        "kelembaban": 85,
        "kecepatan_angin": 1,
        "tekanan_udara": 94.9
    }'
    ```

---

## ☁️ Informasi Deployment

-   **Platform:** Render
-   **Start Command:** `gunicorn app:app`

