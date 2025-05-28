import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import scipy.signal as signal
import os

class SignalProcessor:
    """
    Kelas untuk memproses sinyal rPPG dan respirasi dari frame video.
    """
    def __init__(self, fps, model_dir="models/"):
        """
        Menginisialisasi SignalProcessor dengan model MediaPipe.

        Args:
            fps (float): Frame rate dari video input.
            model_dir (str): Direktori tempat model MediaPipe disimpan.
        """
        self.fps = fps
        self.model_dir = model_dir

        # Inisialisasi Face Detector untuk rPPG
        face_model_path = f"{self.model_dir}blaze_face_short_range.tflite"
        base_options_face = python.BaseOptions(model_asset_path=face_model_path)
        options_face = vision.FaceDetectorOptions(
            base_options=base_options_face,
            running_mode=vision.RunningMode.IMAGE
        )
        self.face_detector = vision.FaceDetector.create_from_options(options_face)

        # Inisialisasi Pose Landmarker untuk Respirasi
        pose_model_path = f"{self.model_dir}pose_landmarker.task"
        base_options_pose = python.BaseOptions(model_asset_path=pose_model_path)
        options_pose = vision.PoseLandmarkerOptions(
            base_options=base_options_pose,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options_pose)

        # Buffer sinyal
        self.r_signal_buffer = []
        self.g_signal_buffer = []
        self.b_signal_buffer = []
        self.resp_signal_buffer = []

        # Parameter untuk Optical Flow (Respirasi)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.old_gray_frame = None
        self.features_to_track = None
        self.roi_coords = None # (left_x, top_y, right_x, bottom_y)
        self.STANDARD_SIZE = (640, 480) # Ukuran frame untuk pemrosesan optical flow

        # Konfigurasi filter
        self.rppg_lowcut = 0.9 # Hz
        self.rppg_highcut = 2.4 # Hz (sesuai PDF)
        self.filter_order = 3

        # Ukuran window untuk POS (sesuai PDF, 1.6 * fps)
        self.pos_window_length = int(1.6 * self.fps)

    def _POS(self, signal_data, fps):
        """
        Implementasi metode Plane-Orthogonal-to-Skin (POS) untuk ekstraksi rPPG.
        Diadaptasi dari kode yang disediakan di PDF.

        Args:
            signal_data (np.array): Sinyal RGB 3D (estimators, color_channels, frames).
            fps (float): Frame rate.

        Returns:
            np.array: Sinyal rPPG yang diekstraksi.
        """
        eps = 1e-9
        X = signal_data
        e, c, f = X.shape # Number of estimators, color channels, and frames
        w = int(1.6 * fps) # Window Length in frames

        P = np.array([[0, 1, 1], [-2, 1, 1]])
        Q = np.stack([P for _ in range(e)], axis=0)

        H = np.zeros((e, f))

        for n in np.arange(w, f):
            m = n - w + 1 # Start index of sliding window

            # Temporal Normalization (Equation 5 from the paper):
            Cn = X[:, :, m:(n + 1)]
            M = 1.0 / (np.mean(Cn, axis=2) + eps)
            M = np.expand_dims(M, axis=2) # shape [e, c, w]
            Cn = np.multiply(Cn, M)

            # Projection (Equation 6 from the paper):
            S = np.dot(Q, Cn)
            S = S[0, :, :, :] # Assuming 1 estimator for simplicity
            S = np.swapaxes(S, 0, 1)

            # Tuning (Equation 7 from the paper):
            S1 = S[:, 0, :]
            S2 = S[:, 1, :]
            alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1))
            alpha = np.expand_dims(alpha, axis=1)
            Hn = np.add(S1, alpha * S2)
            Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)

            # Overlap-Adding (Equation 8 from the paper):
            H[:, m:(n + 1)] = np.add(H[:, m:(n + 1)], Hnm)

        return H.reshape(-1) # Mengembalikan sinyal 1D

    def _get_initial_roi_for_respiration(self, image):
        """
        Mendapatkan ROI awal berdasarkan posisi bahu menggunakan pose detection.
        Diadaptasi dari kode yang disediakan di PDF.

        Args:
            image (np.array): Frame video input (BGR).

        Returns:
            tuple: (left_x, top_y, right_x, bottom_y) koordinat ROI.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = self.pose_landmarker.detect(mp_image)

        if not detection_result.pose_landmarks:
            raise ValueError("Tidak ada pose terdeteksi di frame untuk respirasi.")

        landmarks = detection_result.pose_landmarks[0]
        
        # Menggunakan landmark bahu kiri (11) dan kanan (12)
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        # Menghitung titik tengah antara bahu
        center_x = int((left_shoulder.x + right_shoulder.x) * width / 2)
        center_y = int((left_shoulder.y + right_shoulder.y) * height / 2)

        # Parameter untuk ukuran dan pergeseran ROI (sesuai PDF)
        x_size = 100
        y_size = 100
        shift_x = 6
        shift_y = 0 # PDF tidak menyebutkan shift_y, jadi 0

        center_x += shift_x
        center_y += shift_y

        left_x = max(0, center_x - x_size)
        right_x = min(width, center_x + x_size)
        top_y = max(0, center_y - y_size)
        bottom_y = min(height, center_y) # PDF menggunakan center_y sebagai bottom_y, perlu disesuaikan jika ingin kotak

        if (right_x - left_x) <= 0 or (bottom_y - top_y) <= 0:
            raise ValueError("Dimensi ROI tidak valid.")

        return (left_x, top_y, right_x, bottom_y)

    def _initialize_features_for_respiration(self, frame):
        """
        Menginisialisasi fitur untuk pelacakan optical flow pada ROI dada.
        Diadaptasi dari kode yang disediakan di PDF.

        Args:
            frame (np.array): Frame sumber dari kamera.
        """
        resized_frame = cv2.resize(frame, self.STANDARD_SIZE)
        
        # Mendapatkan ROI awal jika belum ada
        if self.roi_coords is None:
            try:
                self.roi_coords = self._get_initial_roi_for_respiration(resized_frame)
            except ValueError as e:
                print(f"Peringatan: {e}. Mencoba lagi di frame berikutnya.")
                self.features_to_track = None
                self.old_gray_frame = None
                return

        left_x, top_y, right_x, bottom_y = self.roi_coords
        
        self.old_gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        roi_chest = self.old_gray_frame[top_y:bottom_y, left_x:right_x]

        features = cv2.goodFeaturesToTrack(roi_chest, maxCorners=50, qualityLevel=0.03, minDistance=7)
        if features is None:
            raise ValueError("Tidak ada fitur yang ditemukan untuk dilacak!")

        # Sesuaikan koordinat fitur ke koordinat frame penuh
        features[:, :, 0] += left_x
        features[:, :, 1] += top_y
        self.features_to_track = np.float32(features)


    def process_frame(self, frame):
        """
        Memproses satu frame untuk mengekstrak sinyal rPPG dan respirasi.

        Args:
            frame (np.array): Frame video (BGR).

        Returns:
            tuple: (current_rppg_signal, current_resp_signal, processed_frame, face_bbox)
                   current_rppg_signal: Sinyal rPPG yang telah difilter.
                   current_resp_signal: Sinyal respirasi (rata-rata pergerakan Y).
                   processed_frame: Frame dengan overlay visualisasi (bounding box, dll).
                   face_bbox: Tuple (x, y, w, h) dari bounding box wajah.
        """
        display_frame = frame.copy() # Frame untuk ditampilkan dengan overlay
        face_bounding_box = None # Inisialisasi bounding box wajah

        # --- Pemrosesan rPPG ---
        rppg_signal_out = None
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.face_detector.detect(mp_image)

            if result.detections:
                # Ambil deteksi pertama (asumsi satu wajah)
                detection = result.detections[0]
                bboxC = detection.bounding_box
                x, y, w, h = bboxC.origin_x, bboxC.origin_y, bboxC.width, bboxC.height
                face_bounding_box = (int(x), int(y), int(w), int(h)) # Simpan bounding box

                # Setup tipis tipis biar boxnya pas di tengah wajah (sesuai PDF)
                margin_x = 10
                scaling_factor = 0.8
                new_x = int(x + margin_x)
                new_w = int(w * scaling_factor)
                new_h = int(h * scaling_factor)

                # Dapatkan ROI (Region of Interest)
                # Umumnya rPPG diambil dari dahi atau pipi, di sini kita ambil bagian atas wajah
                # Bisa disesuaikan lebih lanjut untuk area spesifik
                face_roi = rgb_frame[y:y + new_h, new_x:new_x + new_w]

                # Gambar bounding box di frame yang akan ditampilkan (ini akan digambar ulang di visual.py)
                # cv2.rectangle(display_frame, (int(x), int(y)), (int(x + new_w), int(y + new_h)), (0, 255, 0), 2)

                # Hitung nilai rata-rata RGB dari ROI wajah
                mean_rgb = cv2.mean(face_roi)[:3]
                self.r_signal_buffer.append(mean_rgb[0])
                self.g_signal_buffer.append(mean_rgb[1])
                self.b_signal_buffer.append(mean_rgb[2])

                # Batasi buffer sinyal ke ukuran window POS
                if len(self.r_signal_buffer) > self.pos_window_length:
                    self.r_signal_buffer.pop(0)
                    self.g_signal_buffer.pop(0)
                    self.b_signal_buffer.pop(0)

                # Hanya proses rPPG jika buffer cukup penuh
                if len(self.r_signal_buffer) == self.pos_window_length:
                    rgb_signals = np.array([self.r_signal_buffer, self.g_signal_buffer, self.b_signal_buffer])
                    rgb_signals = rgb_signals.reshape(1, 3, -1) # Bentuk (estimators, channels, frames)
                    raw_rppg = self._POS(rgb_signals, self.fps)

                    # Filter rPPG
                    b, a = signal.butter(self.filter_order, [self.rppg_lowcut, self.rppg_highcut],
                                         btype='band', fs=self.fps)
                    rppg_signal_out = signal.filtfilt(b, a, raw_rppg)

        except Exception as e:
            print(f"Error saat pemrosesan rPPG: {e}")
            rppg_signal_out = None # Reset jika ada error
            face_bounding_box = None # Reset bounding box jika ada error

        # --- Pemrosesan Respirasi (Optical Flow) ---
        resp_signal_out = None
        try:
            resized_frame = cv2.resize(frame, self.STANDARD_SIZE)
            if self.features_to_track is None or len(self.features_to_track) < 10:
                # Inisialisasi fitur jika belum ada atau terlalu sedikit
                self._initialize_features_for_respiration(resized_frame)
                self.old_gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
                # Langsung kembali jika fitur baru diinisialisasi
                return rppg_signal_out, resp_signal_out, display_frame, face_bounding_box

            frame_gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

            # Lakukan pelacakan optical flow
            new_features, status, error = cv2.calcOpticalFlowPyrLK(
                self.old_gray_frame, frame_gray, self.features_to_track, None, **self.lk_params
            )

            if new_features is not None and status.any():
                good_old = self.features_to_track[status == 1]
                good_new = new_features[status == 1]

                # Gambar garis pelacakan dan titik pada frame yang ditampilkan
                # Skala koordinat pelacakan kembali ke ukuran frame asli
                scale_x = frame.shape[1] / self.STANDARD_SIZE[0]
                scale_y = frame.shape[0] / self.STANDARD_SIZE[1]

                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    cv2.line(display_frame, (int(c * scale_x), int(d * scale_y)), (int(a * scale_x), int(b * scale_y)), (0, 0, 255), 2)
                    cv2.circle(display_frame, (int(a * scale_x), int(b * scale_y)), 3, (0, 255, 255), -1)

                # Hitung rata-rata pergerakan Y sebagai sinyal respirasi
                avg_y = np.mean(good_new[:, 1])
                self.resp_signal_buffer.append(avg_y)

                # Batasi buffer sinyal respirasi (misalnya, 60 detik data)
                if len(self.resp_signal_buffer) > self.fps * 60:
                    self.resp_signal_buffer.pop(0)
                
                # Sinyal respirasi yang akan dikembalikan adalah rata-rata buffer terakhir
                resp_signal_out = np.array(self.resp_signal_buffer)
                
                # Update fitur dan frame lama untuk iterasi berikutnya
                self.features_to_track = good_new.reshape(-1, 1, 2)
                self.old_gray_frame = frame_gray.copy()
            else:
                # Re-initialize features if tracking fails
                self.features_to_track = None
                self.old_gray_frame = None
                self.roi_coords = None # Reset ROI juga
                print("Pelacakan fitur respirasi gagal, menginisialisasi ulang.")

            # Gambar bounding box ROI respirasi (jika ada)
            if self.roi_coords:
                left_x, top_y, right_x, bottom_y = self.roi_coords
                scale_x = frame.shape[1] / self.STANDARD_SIZE[0]
                scale_y = frame.shape[0] / self.STANDARD_SIZE[1]
                cv2.rectangle(display_frame, 
                              (int(left_x * scale_x), int(top_y * scale_y)), 
                              (int(right_x * scale_x), int(bottom_y * scale_y)), 
                              (255, 0, 0), 2)

        except Exception as e:
            print(f"Error saat pemrosesan respirasi: {e}")
            resp_signal_out = None # Reset jika ada error
            self.features_to_track = None # Pastikan inisialisasi ulang
            self.old_gray_frame = None
            self.roi_coords = None

        return rppg_signal_out, resp_signal_out, display_frame, face_bounding_box

if __name__ == "__main__":
    # Contoh penggunaan:
    # Ini memerlukan model MediaPipe di folder 'models/'
    # Unduh dari:
    # https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
    # https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker/float16/1/pose_landmarker.task
    
    # Buat folder models jika belum ada
    if not os.path.exists("models"):
        os.makedirs("models")
        print("Folder 'models' dibuat. Harap unduh model MediaPipe ke folder ini.")
        print("blaze_face_short_range.tflite: https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite")
        print("pose_landmarker.task: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker/float16/1/pose_landmarker.task")
        exit() # Keluar agar user bisa mengunduh model

    # Simulasi FPS
    simulated_fps = 30.0
    processor = SignalProcessor(simulated_fps)

    cap = cv2.VideoCapture(0) # Gunakan webcam
    if not cap.isOpened():
        print("Error: Tidak dapat membuka webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame.")
            break

        rppg_sig, resp_sig, processed_frame, bbox = processor.process_frame(frame)

        if rppg_sig is not None:
            # print(f"rPPG Signal (last 5 values): {rppg_sig[-5:]}")
            pass # Hanya untuk debugging, tidak perlu print terus menerus

        if resp_sig is not None:
            # print(f"Respiration Signal (last 5 values): {resp_sig[-5:]}")
            pass # Hanya untuk debugging

        # Tambahkan bounding box ke frame untuk contoh penggunaan di sini
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(processed_frame, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Processed Frame", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
