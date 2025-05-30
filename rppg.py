import numpy as np
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
from glob import glob
import scipy.signal as signal

def cpu_POS(signal, **kargs):
    """
    POS (Plane-Orthogonal-to-Skin) method untuk ekstraksi sinyal rPPG.

    Params:
        signal: Numpy array dengan bentuk [#estimators, 3 (RGB), #frames]
        fps: frame per second dari video yang digunakan

    Return:
        H: Sinyal rPPG hasil POS method untuk setiap estimator (baris)

    Referensi:
        Wang, W. et al. (2016). Algorithmic principles of remote PPG.
    """
    eps = 1e-9  # Untuk mencegah pembagian dengan nol
    X = signal
    e, c, f = X.shape            # e = jumlah estimator (biasanya 1), c = 3 channel RGB, f = jumlah frame
    w = int(1.6 * kargs['fps'])  # panjang sliding window, default 1.6 detik

    # Matriks proyeksi POS, diterapkan ke setiap estimator
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    Q = np.stack([P for _ in range(e)], axis=0)

    # Inisialisasi sinyal keluaran H
    H = np.zeros((e, f))
    
    # Iterasi untuk setiap window (dimulai dari indeks ke-w)
    for n in np.arange(w, f):
        m = n - w + 1  # posisi awal window

        # Normalisasi temporer: membagi sinyal RGB dengan nilai rata-ratanya di window tersebut
        Cn = X[:, :, m:(n + 1)]
        M = 1.0 / (np.mean(Cn, axis=2) + eps)
        M = np.expand_dims(M, axis=2)
        Cn = np.multiply(M, Cn)

        # Proyeksi sinyal ke ruang POS
        S = np.dot(Q, Cn)
        S = S[0, :, :, :]
        S = np.swapaxes(S, 0, 1)  # ubah dimensi agar jadi [window, 2, estimator]

        # Penyelarasan sinyal (Tuning POS)
        S1 = S[:, 0, :]
        S2 = S[:, 1, :]
        alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1))
        alpha = np.expand_dims(alpha, axis=1)
        Hn = np.add(S1, alpha * S2)
        Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)

        # Gabungkan sinyal yang tumpang tindih ke sinyal akhir
        H[:, m:(n + 1)] += Hnm

    return H

def get_rgb_roi(frame):
    """
    Deteksi wajah dan ekstraksi rata-rata nilai RGB dari Region of Interest (ROI).

    Params:
        frame: frame gambar dari video (format BGR)

    Return:
        r_signal, g_signal, b_signal: nilai rata-rata channel warna RGB di ROI
        Jika wajah tidak terdeteksi, mengembalikan None untuk semua sinyal.
    """
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
            
    # Konversi ke RGB karena MediaPipe bekerja di RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)
    
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            width, height = int(bbox.width * w), int(bbox.height * h)

            # Tetapkan ROI berbentuk kotak persegi (130x130 px)
            bbox_size_from_center = 65
            bbox_center_x = x + width // 2
            bbox_center_y = y + height // 2
            new_x = bbox_center_x - bbox_size_from_center
            new_y = bbox_center_y - bbox_size_from_center
            new_width = new_height = bbox_size_from_center * 2
                    
            roi = frame[new_y:new_y+new_height, new_x:new_x+new_width]
            # Rata-rata nilai setiap channel
            r_signal = np.mean(roi[:, :, 0])  # channel merah
            g_signal = np.mean(roi[:, :, 1])  # channel hijau
            b_signal = np.mean(roi[:, :, 2])  # channel biru

            return r_signal, g_signal, b_signal
        
    else:
        return None, None, None

def get_rppg(r_signal, g_signal, b_signal):
    """
    Proses akhir untuk menghitung sinyal rPPG dan detak jantung dari sinyal RGB.

    Params:
        r_signal, g_signal, b_signal: sinyal RGB hasil ekstraksi dari ROI selama beberapa frame

    Return:
        filtered_rppg: sinyal rPPG setelah difilter dan dinormalisasi
        heart_rate: detak jantung (BPM)
        peaks: indeks puncak-puncak sinyal
        heart_rate: nilai BPM yang sama (redundan)
    """
    # Gabungkan sinyal RGB ke dalam array [1, 3, N]
    rgb_signals = np.array([r_signal, g_signal, b_signal]).reshape(1, 3, -1)

    # Ekstraksi sinyal rPPG menggunakan metode POS
    rppg_signal = cpu_POS(rgb_signals, fps=15)
    rppg_signal = rppg_signal.reshape(-1)

    # Filter sinyal menggunakan bandpass filter Butterworth
    fs = 15            # frekuensi sampling (FPS video)
    lowcut = 0.9       # cutoff bawah (0.9 Hz ≈ 54 BPM)
    highcut = 2.4      # cutoff atas (2.4 Hz ≈ 144 BPM)
    order = 3          # orde filter
    b, a = signal.butter(order, [lowcut, highcut], btype='band', fs=fs)
    filtered_rppg = signal.filtfilt(b, a, rppg_signal)
    
    # Hilangkan data awal agar hasil lebih stabil
    sliced_filtered_rppg = filtered_rppg[30:]

    # Normalisasi sinyal agar memiliki mean 0 dan deviasi standar 1
    filtered_rppg = (sliced_filtered_rppg - np.mean(sliced_filtered_rppg)) / np.std(sliced_filtered_rppg)

    # Deteksi puncak-puncak pada sinyal
    peaks, _ = signal.find_peaks(filtered_rppg, prominence=0.5)

    # Hitung detak jantung dalam BPM
    heart_rate = 60 * len(peaks) / (len(filtered_rppg) / fs)

    return filtered_rppg, heart_rate, peaks, heart_rate
