import cv2
import mediapipe as mp
import time
import numpy as np
from scipy.signal import butter, filtfilt

def get_resp(frame, start_time):
    """
    Mengekstraksi sinyal respirasi dari pergerakan vertikal bahu kiri dan kanan menggunakan MediaPipe Pose.

    Parameters:
    - frame: Frame video (BGR) dari webcam.
    - start_time: Waktu awal program (untuk menghitung timestamp relatif).

    Returns:
    - avg_y: Rata-rata posisi vertikal (y) bahu kiri dan kanan dalam piksel.
    - curr_time: Waktu relatif frame dari start_time (dalam detik).
    """

    # Inisialisasi model pose MediaPipe
    mp_pose = mp.solutions.pose
    # Mengaktifkan deteksi pose (realtime, model kompleks, confidence minimal 0.5)
    pose = mp_pose.Pose(
        static_image_mode=False,      # Karena inputnya berupa video, bukan gambar statis
        model_complexity=2,           # Model kompleks untuk akurasi lebih tinggi
        min_detection_confidence=0.5  # Confidence minimum agar landmark diakui valid
    )

    # Konversi frame dari BGR (OpenCV) ke RGB (MediaPipe)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Proses deteksi pose pada frame
    results = pose.process(image_rgb)

    # Jika landmark tubuh berhasil dideteksi
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Ambil nilai koordinat Y (vertikal) dari bahu kiri dan kanan
        left_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0]
        right_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0]

        # Hitung rata-rata dari dua titik bahu tersebut sebagai sinyal respirasi
        avg_y = (left_y + right_y) / 2

        # Hitung timestamp relatif terhadap waktu mulai program
        return avg_y, time.time() - start_time

    # Jika tidak ada landmark, kembalikan None
    return None, None


def butter_bandpass_resp(data, lowcut=0.1, highcut=0.6, fs=15, order=2):
    """
    Menerapkan filter bandpass Butterworth pada sinyal respirasi untuk menghilangkan noise di luar rentang frekuensi pernapasan manusia.

    Parameters:
    - data: Sinyal mentah (rata-rata posisi bahu tiap frame).
    - lowcut: Frekuensi cutoff bawah (Hz). Default 0.1 Hz ≈ 6 napas/menit.
    - highcut: Frekuensi cutoff atas (Hz). Default 0.6 Hz ≈ 36 napas/menit.
    - fs: Sampling rate (frame per second). Default 15 fps.
    - order: Orde filter Butterworth. Orde rendah menghasilkan respons cepat dan stabil.

    Returns:
    - filtered_signal: Sinyal yang sudah difilter dan lebih halus, hanya berisi komponen napas.
    """

    # Buat filter bandpass Butterworth dengan parameter yang ditentukan
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')

    # Terapkan filter dengan metode zero-phase (tidak menggeser sinyal)
    filtered_signal = filtfilt(b, a, data)

    return filtered_signal
