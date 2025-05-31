import time
import cv2
import numpy as np
import pygame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 

from resp import get_resp
from rppg import get_rppg, get_rgb_roi


# Fungsi untuk membuat chart dan mengembalikannya dalam bentuk numpy array 
def create_realtime_chart(respiration, rppg, peaks, heart_rate):
    """Membuat plot untuk detak jantung dan pernapasan."""
    fig, ax = plt.subplots(2, 1, figsize=(6, 4), facecolor='#DDDDDD') #
    # Plot Detak Jantung (rPPG)
    ax[0].set_title(f'Heart Rate: {heart_rate:.0f} BPM')
    ax[0].plot(rppg, color='red', label='rPPG Signal')
    ax[0].plot(peaks, [rppg[i] for i in peaks], 'x', color='blue', label='Peaks')
    ax[0].set_ylabel('Intensitas Piksel')
    ax[0].legend(loc='upper right')
    
    # Plot Pernapasan
    ax[1].set_title('Respiration Signal')
    ax[1].plot(respiration, color='green', label='Respiration Signal')
    ax[1].set_xlabel('Sampel Waktu')
    ax[1].set_ylabel('Pergerakan')
    ax[1].legend(loc='upper right')

    plt.subplots_adjust(hspace=0.6)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    
    width, height = canvas.get_width_height()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    chart_img = buf[:, :, :3].copy()
    
    plt.close(fig)
    return chart_img

def main():
    # Inisialisasi webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Gagal membuka webcam.")
        return

    # Inisialisasi pygame
    pygame.init()
    try:
        info = pygame.display.Info()
        screen_width = info.current_w
        screen_height = info.current_h
    except pygame.error:
        print("Tidak bisa mendapatkan info display. Menggunakan ukuran default.")
        screen_width, screen_height = 1280, 720


    # Ukuran jendela (80% lebar, 90% tinggi)
    window_width = max(int(screen_width * 0.8), 800)
    window_height = max(int(screen_height * 0.9), 600)

    # Ukuran area webcam & chart
    webcam_area_height = window_height // 2
    chart_area_height = window_height - webcam_area_height

    # Inisialisasi layar pygame
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Realtime rPPG and Respiration Signal")
    font = pygame.font.SysFont(None, 24)

    # Variabel sinyal
    timestamps, resp, r_signal, g_signal, b_signal = [], [], [], [], []
    start_time = time.time()
    chart_update_interval = 2  # detik
    last_chart_update_time = 0
    chart_surface = None

    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        # Mengubah ukuran frame agar pas dengan area webcam yang sudah ditentukan
        frame_resized = cv2.resize(frame, (window_width, webcam_area_height))

        # Deteksi ROI wajah (menggunakan frame asli untuk akurasi)
        curr_r_signal, curr_g_signal, curr_b_signal = get_rgb_roi(frame)
        if curr_r_signal:
            r_signal.append(curr_r_signal)
            g_signal.append(curr_g_signal)
            b_signal.append(curr_b_signal)

        # Ubah frame yang sudah di-resize ke RGB dan buat surface
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Menggunakan swapaxes untuk konversi yang benar dari array NumPy ke Pygame
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))

        # Update chart jika cukup data dan sudah lewat interval
        current_time = time.time()
        if len(r_signal) > 30 and current_time - last_chart_update_time > chart_update_interval:
            curr_resp, curr_time = get_resp(frame, start_time)
            resp.append(curr_resp)
            timestamps.append(curr_time)

            rppg, heart_rate, rppg_peaks, _ = get_rppg(r_signal, g_signal, b_signal)
            chart_img = create_realtime_chart(resp, rppg, rppg_peaks, heart_rate)
            
            # Resize gambar chart sesuai dengan areanya
            chart_img_resized = cv2.resize(chart_img, (window_width, chart_area_height))
            
            # --- PERUBAHAN 3: KOREKSI ORIENTASI GRAFIK & KODE LEBIH BERSIH ---
            # Tidak ada lagi konversi warna bolak-balik yang tidak perlu
            # Menggunakan swapaxes agar orientasi grafik benar
            chart_surface = pygame.surfarray.make_surface(chart_img_resized.swapaxes(0, 1))
            last_chart_update_time = current_time

        # Gambar semua elemen ke layar pygame
        screen.fill((0, 0, 0))
        screen.blit(frame_surface, (0, 0))
        if chart_surface:
            screen.blit(chart_surface, (0, webcam_area_height))
        
        # Tampilkan tulisan instruksi
        text_surface = font.render("Tekan 'q' atau tutup jendela untuk keluar", True, (255, 255, 255))
        screen.blit(text_surface, (10, 10))

        pygame.display.update()

        # Event handling untuk keluar dari program
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            running = False

    # Bersih-bersih
    cap.release()
    pygame.quit()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
