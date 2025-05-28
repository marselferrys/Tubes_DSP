import cv2
import time
import os
import numpy as np # Import numpy untuk np.array([0.0])

# Import modul-modul yang telah dibuat
from video_input import VideoInput
from signal_process import SignalProcessor
from feature_extract import FeatureExtractor
from mood import MoodClassifier
from visual import Visualizer

def main():
    """
    Fungsi utama untuk menjalankan program Mood Meter.
    """
    print("Memulai aplikasi Mood Meter...")

    # --- Inisialisasi Model MediaPipe ---
    # Pastikan model-model ini ada di folder 'models/'
    # Unduh dari:
    # blaze_face_short_range.tflite: https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
    # pose_landmarker.task: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker/float16/1/pose_landmarker.task
    if not os.path.exists("models"):
        os.makedirs("models")
        print("Folder 'models' dibuat. Harap unduh model MediaPipe ke folder ini.")
        print("blaze_face_short_range.tflite: https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite")
        print("pose_landmarker.task: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker/float16/1/pose_landmarker.task")
        return # Keluar agar user bisa mengunduh model

    # --- Inisialisasi Modul ---
    video_source = 0 # 0 untuk webcam default
    video_input = VideoInput(video_source)
    
    fps = video_input.get_fps()
    if fps == 0:
        print("Peringatan: FPS kamera tidak terdeteksi, menggunakan default 30 FPS.")
        fps = 30.0 # Fallback jika tidak dapat mendeteksi FPS

    signal_processor = SignalProcessor(fps=fps)
    feature_extractor = FeatureExtractor(fps=fps)
    mood_classifier = MoodClassifier()
    visualizer = Visualizer() # Inisialisasi visualizer

    print(f"Aplikasi siap. Menggunakan FPS: {fps:.2f}")
    print("Tekan 'q' pada jendela video untuk keluar.")

    start_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = video_input.get_frame()
            if not ret:
                print("Gagal membaca frame dari video input. Mengakhiri.")
                break

            # --- Proses Sinyal ---
            # signal_processor.process_frame sekarang mengembalikan face_bbox juga
            rppg_signal, resp_signal, processed_frame_with_overlay, face_bbox = signal_processor.process_frame(frame)

            # --- Ekstraksi Fitur ---
            rppg_features = feature_extractor.extract_rppg_features(rppg_signal)
            respiration_features = feature_extractor.extract_respiration_features(resp_signal)

            # --- Klasifikasi Mood ---
            # mood_classifier.classify_mood sekarang mengembalikan mood dan confidence
            current_mood, mood_confidence = mood_classifier.classify_mood(rppg_features, respiration_features)

            # --- Visualisasi ---
            # Update grafik sinyal dan dapatkan gambar Matplotlib yang dirender
            matplotlib_image = visualizer.update_signals(rppg_signal, resp_signal)

            # Tampilkan frame video dengan overlay mood dan grafik Matplotlib dalam satu jendela
            visualizer.display_frame_and_mood(processed_frame_with_overlay, face_bbox, 
                                              current_mood, mood_confidence, 
                                              rppg_features, respiration_features, 
                                              matplotlib_image)

            frame_count += 1
            
            # Cek untuk keluar
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Tombol 'q' ditekan. Mengakhiri aplikasi.")
                break

    except Exception as e:
        print(f"Terjadi kesalahan fatal: {e}")
    finally:
        # --- Pembersihan ---
        video_input.release()
        visualizer.close()
        print("Aplikasi Mood Meter telah berhenti.")

if __name__ == "__main__":
    main()
