import numpy as np
import scipy.signal as signal

class FeatureExtractor:
    """
    Kelas untuk mengekstrak fitur-fitur dari sinyal rPPG dan respirasi.
    """
    def __init__(self, fps):
        """
        Menginisialisasi FeatureExtractor.

        Args:
            fps (float): Frame rate dari sinyal.
        """
        self.fps = fps
        # Parameter untuk deteksi puncak
        self.rppg_peak_min_distance = int(self.fps / 4) # Minimal 15 BPM
        self.resp_peak_min_distance = int(self.fps / 0.5) # Minimal 30 BPM
        self.filter_order = 3 # Ditambahkan: Inisialisasi filter_order

    def extract_rppg_features(self, rppg_signal):
        """
        Mengekstrak fitur dari sinyal rPPG.

        Args:
            rppg_signal (np.array): Sinyal rPPG yang telah difilter.

        Returns:
            dict: Kamus berisi fitur-fitur rPPG (e.g., 'heart_rate_bpm').
        """
        features = {}
        if rppg_signal is None or len(rppg_signal) < self.fps * 2: # Butuh minimal 2 detik data
            return features

        try:
            # Deteksi puncak untuk menghitung detak jantung
            peaks, _ = signal.find_peaks(rppg_signal, distance=self.rppg_peak_min_distance)
            
            if len(peaks) > 1:
                # Menghitung Heart Rate (BPM)
                duration_seconds = len(rppg_signal) / self.fps
                heart_rate_bpm = (len(peaks) / duration_seconds) * 60
                features['heart_rate_bpm'] = heart_rate_bpm

                # Perhitungan HRV (sederhana: SDNN - Standard Deviation of NN intervals)
                # NN intervals adalah jarak antar puncak
                nn_intervals = np.diff(peaks) / self.fps # dalam detik
                if len(nn_intervals) > 1:
                    features['hrv_sdnn'] = np.std(nn_intervals)
                else:
                    features['hrv_sdnn'] = 0.0 # Tidak cukup data untuk HRV
            else:
                features['heart_rate_bpm'] = 0.0
                features['hrv_sdnn'] = 0.0
        except Exception as e:
            print(f"Error saat mengekstrak fitur rPPG: {e}")
            features['heart_rate_bpm'] = 0.0
            features['hrv_sdnn'] = 0.0
        
        return features

    def extract_respiration_features(self, resp_signal):
        """
        Mengekstrak fitur dari sinyal respirasi.

        Args:
            resp_signal (np.array): Sinyal respirasi.

        Returns:
            dict: Kamus berisi fitur-fitur respirasi (e.g., 'respiration_rate_bpm', 'respiration_amplitude').
        """
        features = {}
        if resp_signal is None or len(resp_signal) < self.fps * 5: # Butuh minimal 5 detik data
            return features

        try:
            # Filter sinyal respirasi (opsional, jika belum difilter di signal_process)
            # Contoh filter bandpass untuk respirasi (0.1 - 0.5 Hz)
            # Ini bisa disesuaikan lebih lanjut
            b_resp, a_resp = signal.butter(self.filter_order, [0.1, 0.5], btype='band', fs=self.fps)
            filtered_resp_signal = signal.filtfilt(b_resp, a_resp, resp_signal)

            # Deteksi puncak untuk menghitung laju pernapasan
            peaks, _ = signal.find_peaks(filtered_resp_signal, distance=self.resp_peak_min_distance)
            
            if len(peaks) > 1:
                # Menghitung Respiration Rate (BPM)
                duration_seconds = len(resp_signal) / self.fps
                respiration_rate_bpm = (len(peaks) / duration_seconds) * 60
                features['respiration_rate_bpm'] = respiration_rate_bpm

                # Menghitung amplitudo pernapasan (misalnya, rata-rata jarak puncak-ke-lembah)
                # Ini adalah pendekatan sederhana
                valleys, _ = signal.find_peaks(-filtered_resp_signal, distance=self.resp_peak_min_distance)
                if len(peaks) > 0 and len(valleys) > 0:
                    # Mencari pasangan puncak-lembah terdekat
                    amplitudes = []
                    for p_idx in peaks:
                        closest_valley_idx = valleys[np.argmin(np.abs(valleys - p_idx))]
                        amplitude = np.abs(filtered_resp_signal[p_idx] - filtered_resp_signal[closest_valley_idx])
                        amplitudes.append(amplitude)
                    if amplitudes:
                        features['respiration_amplitude'] = np.mean(amplitudes)
                    else:
                        features['respiration_amplitude'] = 0.0
                else:
                    features['respiration_amplitude'] = 0.0
            else:
                features['respiration_rate_bpm'] = 0.0
                features['respiration_amplitude'] = 0.0
        except Exception as e:
            print(f"Error saat mengekstrak fitur respirasi: {e}")
            features['respiration_rate_bpm'] = 0.0
            features['respiration_amplitude'] = 0.0

        return features

if __name__ == "__main__":
    # Contoh penggunaan:
    # Simulasi sinyal rPPG dan respirasi
    simulated_fps = 30.0
    time = np.arange(0, 60, 1/simulated_fps) # 60 detik data

    # Sinyal rPPG simulasi (sinusoidal dengan noise)
    sim_rppg = 0.1 * np.sin(2 * np.pi * 1.2 * time) + 0.05 * np.random.randn(len(time))
    
    # Sinyal respirasi simulasi (sinusoidal dengan noise)
    sim_resp = 10 * np.sin(2 * np.pi * 0.2 * time) + 2 * np.random.randn(len(time))

    extractor = FeatureExtractor(simulated_fps)

    rppg_features = extractor.extract_rppg_features(sim_rppg)
    resp_features = extractor.extract_respiration_features(sim_resp)

    print("Fitur rPPG:", rppg_features)
    print("Fitur Respirasi:", resp_features)
