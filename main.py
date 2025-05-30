import time
import cv2
import numpy as np
import pygame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from resp import get_resp
from rppg import get_rppg, get_rgb_roi

# Fungsi untuk membuat chart dan mengembalikannya dalam bentuk numpy array 
def create_realtime_chart(respiration, rppg, peaks, heart_rate):
    fig, ax = plt.subplots(2, 1, figsize=(6, 4))
    ax[0].set_title(f'Heart Rate: {heart_rate:.0f}')
    ax[0].plot(rppg, color='black')
    ax[0].plot(peaks, [rppg[i] for i in peaks], 'x', color='red')
    ax[1].set_title('Respiration Signal')
    ax[1].plot(respiration, color='black')
    plt.subplots_adjust(hspace=0.5)

    # Gunakan FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    # Ambil buffer RGBA dan ubah ke RGB
    width, height = canvas.get_width_height()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    chart_img = buf[:, :, :3].copy()  # Ambil hanya RGB, buang alpha

    plt.close(fig)
    return chart_img

def main():
    # Inisialisasi webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Gagal membuka webcam.")
        return

    # Ambil resolusi webcam
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame.")
        return
    frame_height, frame_width = frame.shape[:2]
    
    # Dapatkan resolusi layar
    pygame.init()
    info = pygame.display.Info()
    screen_width = info.current_w
    screen_height = info.current_h

    # Ukuran jendela (80% lebar, 90% tinggi)
    window_width = max(int(screen_width * 0.8), 800)
    window_height = max(int(screen_height * 0.9), 600)

    # Ukuran area webcam & chart
    webcam_area_height = window_height // 2
    chart_area_height = window_height - webcam_area_height

    # Inisialisasi pygame
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("rppg dan respirasi Realtime Chart")
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

        # Deteksi ROI wajah
        curr_r_signal, curr_g_signal, curr_b_signal = get_rgb_roi(frame)
        if not curr_r_signal:
            continue

        r_signal.append(curr_r_signal)
        g_signal.append(curr_g_signal)
        b_signal.append(curr_b_signal)

        # Ubah frame ke RGB dan rotate untuk pygame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(np.rot90(frame_rgb))

        # Update chart jika cukup data dan sudah lewat interval
        if len(r_signal) > 30 and time.time() - last_chart_update_time > chart_update_interval:
            curr_resp, curr_time = get_resp(frame, start_time)
            resp.append(curr_resp)
            timestamps.append(curr_time)

            rppg, heart_rate, rppg_peaks, _ = get_rppg(r_signal, g_signal, b_signal)
            chart_img = create_realtime_chart(resp, rppg, rppg_peaks, heart_rate)
            chart_img = cv2.resize(chart_img, (window_width, webcam_area_height))
            chart_img = cv2.cvtColor(chart_img, cv2.COLOR_RGB2BGR)
            chart_img = cv2.cvtColor(chart_img, cv2.COLOR_BGR2RGB)
            chart_surface = pygame.surfarray.make_surface(np.rot90(chart_img))
            last_chart_update_time = time.time()

        # Gambar ke layar pygame
        screen.fill((0, 0, 0))
        screen.blit(frame_surface, (0, 0))
        if chart_surface:
            screen.blit(chart_surface, (0, webcam_area_height))

        pygame.display.flip()
        
        # Tampilkan tulisan instruksi
        text_surface = font.render("Tekan 'q' untuk keluar", True, (255, 255, 255))
        screen.blit(text_surface, (10, 10))

        pygame.display.update()

        # Event handling
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
