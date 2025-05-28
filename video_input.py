import cv2
import logging
import time

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoInput:
    """
    Kelas untuk mengelola input video dari webcam.
    """
    def __init__(self, source=0):
        """
        Menginisialisasi objek VideoInput.

        Args:
            source (int/str): Sumber video. 0 untuk webcam default,
                              atau path ke file video.
        """
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            error_message = f"Error: Tidak dapat membuka sumber video {source}."
            logging.error(error_message)
            print(error_message)
            exit()
        # Mendapatkan frame rate dari kamera/video
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0: # Perbaikan: Cek jika FPS invalid
            self.fps = 30.0 # Default ke 30 FPS
            logging.warning(f"FPS tidak valid ({self.fps}), menggunakan default 30 FPS.")
        else:
            logging.info(f"Sumber video dibuka. FPS: {self.fps}")
        self.last_frame_time = time.time()  # Tambahkan waktu frame terakhir
        self.frame_count = 0

    def get_frame(self):
        """
        Membaca satu frame dari sumber video.

        Returns:
            tuple: (ret, frame) di mana ret adalah boolean yang menunjukkan
                   apakah frame berhasil dibaca, dan frame adalah gambar.
        """
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            current_time = time.time()
            frame_duration = current_time - self.last_frame_time
            if frame_duration > 1.0 / self.fps: # Cek jika lebih lambat dari FPS
                logging.warning(f"Frame lambat: {frame_duration:.4f} detik (harus < {1.0/self.fps:.4f})")
            self.last_frame_time = current_time
            return True, frame
        else:
            logging.error("Gagal membaca frame dari sumber video.")
            return False, None

    def get_fps(self):
        """
        Mengembalikan frame rate dari sumber video.

        Returns:
            float: Frame rate (frames per second).
        """
        return self.fps

    def release(self):
        """
        Melepaskan sumber video.
        """
        logging.info("Melepaskan sumber video.")
        self.cap.release()

if __name__ == "__main__":
    # Contoh penggunaan:
    video_stream = VideoInput(source=0) # Gunakan webcam
    print(f"FPS Kamera: {video_stream.get_fps()}")

    while True:
        ret, frame = video_stream.get_frame()
        if not ret:
            print("Gagal membaca frame, keluar.")
            break

        cv2.imshow("Webcam Feed", frame)

        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_stream.release()
    cv2.destroyAllWindows()
