import matplotlib.pyplot as plt
import numpy as np
import cv2
import collections # Untuk deque

class Visualizer:
    """
    Kelas untuk menampilkan visualisasi sinyal rPPG, respirasi, dan mood.
    """
    def __init__(self, max_points=300):
        """
        Menginisialisasi Visualizer.

        Args:
            max_points (int): Jumlah maksimum titik data yang akan ditampilkan
                              pada grafik sinyal.
        """
        self.max_points = max_points
        self.rppg_data = collections.deque(np.zeros(max_points), maxlen=max_points)
        self.resp_data = collections.deque(np.zeros(max_points), maxlen=max_points)
        
        self.combined_window_name = "Mood Meter - Combined View"
        
        # Define target dimensions for the combined window
        # Ini adalah ukuran jendela OpenCV yang akan menampilkan semua
        self.target_width = 1000 
        self.target_height = 800 

        # Calculate dimensions for webcam and plot areas within the combined window
        # Webcam akan berada di bagian atas, plot di bagian bawah
        self.webcam_display_height = int(self.target_height * 0.6) # 60% untuk webcam
        self.plot_display_height = self.target_height - self.webcam_display_height # 40% untuk plot

        # Setup Matplotlib figure dan axes
        plt.ioff() # Matplotlib non-interactive mode for rendering to image
        # Sesuaikan figsize dan dpi agar plot terlihat baik saat di-render
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, 
                                                     figsize=(self.target_width / 100, self.plot_display_height / 100), 
                                                     dpi=100) 
        
        self.line_rppg, = self.ax1.plot(list(self.rppg_data), label='Sinyal rPPG', color='red')
        self.line_resp, = self.ax2.plot(list(self.resp_data), label='Sinyal Respirasi', color='blue')

        self.ax1.set_title('Sinyal rPPG')
        self.ax1.set_ylabel('Amplitudo')
        self.ax1.legend()
        self.ax1.set_ylim([-0.2, 0.2]) # Contoh batas Y untuk rPPG

        self.ax2.set_title('Sinyal Respirasi')
        self.ax2.set_xlabel('Waktu (sampel)')
        self.ax2.set_ylabel('Amplitudo')
        self.ax2.legend()
        self.ax2.set_ylim([200, 400]) # Contoh batas Y untuk respirasi (disesuaikan dengan output optical flow)

        self.fig.tight_layout()
        # plt.show(block=False) # Tidak perlu show Matplotlib secara terpisah lagi

    def update_signals(self, rppg_signal, resp_signal):
        """
        Memperbarui data sinyal, grafik, dan merender grafik ke gambar.

        Args:
            rppg_signal (np.array): Sinyal rPPG terbaru.
            resp_signal (np.array): Sinyal respirasi terbaru.

        Returns:
            np.array: Gambar Matplotlib yang dirender (BGR format).
        """
        if rppg_signal is not None and len(rppg_signal) > 0:
            self.rppg_data.append(rppg_signal[-1])
            self.line_rppg.set_ydata(list(self.rppg_data))
            min_rppg = min(self.rppg_data)
            max_rppg = max(self.rppg_data)
            if max_rppg - min_rppg < 0.01:
                self.ax1.set_ylim([min_rppg - 0.05, max_rppg + 0.05])
            else:
                self.ax1.set_ylim([min_rppg - (max_rppg - min_rppg) * 0.1, max_rppg + (max_rppg - min_rppg) * 0.1])
        
        if resp_signal is not None and len(resp_signal) > 0:
            self.resp_data.append(resp_signal[-1])
            self.line_resp.set_ydata(list(self.resp_data))
            min_resp = min(self.resp_data)
            max_resp = max(self.resp_data)
            if max_resp - min_resp < 10:
                self.ax2.set_ylim([min_resp - 20, max_resp + 20])
            else:
                self.ax2.set_ylim([min_resp - (max_resp - min_resp) * 0.1, max_resp + (max_resp - min_resp) * 0.1])

        # Render matplotlib figure to image
        self.fig.canvas.draw()
        # Get RGBA image from matplotlib renderer
        fig_image = np.array(self.fig.canvas.renderer.buffer_rgba())
        # Convert RGBA to BGR for OpenCV
        fig_image = cv2.cvtColor(fig_image, cv2.COLOR_RGBA2BGR)
        
        return fig_image

    def display_frame_and_mood(self, frame, face_bbox, mood_text, mood_confidence, rppg_features, resp_features, matplotlib_image):
        """
        Menampilkan frame video dengan bounding box wajah, informasi mood, fitur,
        dan grafik Matplotlib dalam satu jendela.

        Args:
            frame (np.array): Frame video yang telah diproses (dengan overlay).
            face_bbox (tuple): Koordinat bounding box wajah (x, y, w, h).
            mood_text (str): Teks mood yang terdeteksi.
            mood_confidence (float): Persentase keyakinan dari deteksi mood.
            rppg_features (dict): Fitur rPPG.
            resp_features (dict): Fitur respirasi.
            matplotlib_image (np.array): Gambar Matplotlib yang dirender.
        """
        # Resize webcam frame to fit its allocated height and maintain aspect ratio
        webcam_h, webcam_w, _ = frame.shape
        aspect_ratio = webcam_w / webcam_h
        resized_webcam_w = int(self.webcam_display_height * aspect_ratio)
        
        # Pastikan lebar tidak melebihi target_width
        if resized_webcam_w > self.target_width:
            resized_webcam_w = self.target_width
            self.webcam_display_height = int(self.target_width / aspect_ratio)

        resized_webcam_frame = cv2.resize(frame, (resized_webcam_w, self.webcam_display_height))

        # Create a blank canvas for the combined display
        combined_canvas = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)

        # Place resized webcam frame at the top center
        start_x_webcam = (self.target_width - resized_webcam_w) // 2
        combined_canvas[0:self.webcam_display_height, start_x_webcam:start_x_webcam+resized_webcam_w] = resized_webcam_frame

        # Resize matplotlib image to fit its allocated height and target width
        resized_matplotlib_image = cv2.resize(matplotlib_image, (self.target_width, self.plot_display_height))
        
        # Place resized matplotlib image at the bottom
        combined_canvas[self.webcam_display_height:self.target_height, 0:self.target_width] = resized_matplotlib_image

        # Add text overlays (mood, HR, RR, HRV) on the webcam part of the combined canvas
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Adjust text positions for the combined canvas
        text_offset_y_features = 30 # Initial Y offset for first line of text for features
        line_height_features = 30 # Spacing between lines for features

        # Mood text on combined canvas (top part, relative to face_bbox)
        if face_bbox:
            x, y, w, h = face_bbox
            # Scale bounding box coordinates to the resized webcam frame
            scale_x_bbox = resized_webcam_w / webcam_w
            scale_y_bbox = self.webcam_display_height / webcam_h
            scaled_x, scaled_y, scaled_w, scaled_h = int(x * scale_x_bbox), int(y * scale_y_bbox), int(w * scale_x_bbox), int(h * scale_y_bbox)

            # Draw scaled bounding box on the combined canvas (offset by start_x_webcam)
            cv2.rectangle(combined_canvas, (start_x_webcam + scaled_x, scaled_y), 
                          (start_x_webcam + scaled_x + scaled_w, scaled_y + scaled_h), (0, 255, 0), 2)

            mood_info = f"{mood_text} ({mood_confidence * 100:.1f}%)"
            (text_width, text_height), baseline = cv2.getTextSize(mood_info, font, 0.6, 2)
            
            # Position text relative to scaled bounding box
            text_x_mood = start_x_webcam + scaled_x
            text_y_mood = scaled_y - 10 if scaled_y - 10 > text_height else scaled_y + scaled_h + text_height + 5
            
            # Ensure text is within bounds
            if text_x_mood < 0: text_x_mood = 0
            if text_y_mood < 0: text_y_mood = 0 
            if text_x_mood + text_width > self.target_width: text_x_mood = self.target_width - text_width # Prevent text overflowing right
            if text_y_mood + text_height > self.webcam_display_height: text_y_mood = self.webcam_display_height - text_height # Prevent text overflowing bottom

            cv2.putText(combined_canvas, mood_info, (text_x_mood, text_y_mood), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Feature texts on combined canvas (top left corner of combined canvas)
        cv2.putText(combined_canvas, f"HR: {rppg_features.get('heart_rate_bpm', 0.0):.1f} BPM", 
                    (10, text_offset_y_features), font, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(combined_canvas, f"RR: {resp_features.get('respiration_rate_bpm', 0.0):.1f} BPM", 
                    (10, text_offset_y_features + line_height_features), font, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(combined_canvas, f"HRV: {rppg_features.get('hrv_sdnn', 0.0):.3f}", 
                    (10, text_offset_y_features + 2 * line_height_features), font, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(self.combined_window_name, combined_canvas)

    def close(self):
        """
        Menutup semua jendela visualisasi.
        """
        plt.close(self.fig)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Contoh penggunaan:
    visualizer = Visualizer()

    # Simulasi data real-time
    simulated_fps = 30.0
    time_step = 1 / simulated_fps
    current_time = 0

    while True:
        # Simulasi sinyal
        rppg_val = 0.1 * np.sin(2 * np.pi * 1.2 * current_time) + 0.01 * np.random.randn()
        resp_val = 300 + 50 * np.sin(2 * np.pi * 0.2 * current_time) + 5 * np.random.randn()

        # Simulasi frame (hitam polos)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Simulasi fitur dan mood
        dummy_rppg_features = {'heart_rate_bpm': 70 + 10 * np.sin(current_time/5), 'hrv_sdnn': 0.03 + 0.01 * np.random.randn()}
        dummy_resp_features = {'respiration_rate_bpm': 15 + 3 * np.sin(current_time/7), 'respiration_amplitude': 15 + 5 * np.random.randn()}
        dummy_mood = "Netral"
        dummy_confidence = 0.8
        if dummy_rppg_features['heart_rate_bpm'] > 80:
            dummy_mood = "Agak Stres"
            dummy_confidence = 0.7

        # Simulasi bounding box wajah (contoh statis)
        dummy_face_bbox = (200, 150, 240, 200) # x, y, w, h

        # Dapatkan gambar Matplotlib yang dirender
        matplotlib_img = visualizer.update_signals(np.array([rppg_val]), np.array([resp_val]))
        
        # Tampilkan semua dalam satu jendela
        visualizer.display_frame_and_mood(dummy_frame, dummy_face_bbox, dummy_mood, dummy_confidence, dummy_rppg_features, dummy_resp_features, matplotlib_img)

        current_time += time_step
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    visualizer.close()
