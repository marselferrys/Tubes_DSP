class MoodClassifier:
    """
    Kelas untuk mengklasifikasikan mood berdasarkan fitur fisiologis.
    Ini adalah implementasi dasar berbasis aturan.
    """
    def __init__(self):
        # Rentang normal untuk detak jantung dan laju pernapasan
        # Nilai-nilai ini adalah contoh dan mungkin perlu disesuaikan
        self.normal_hr_min = 60
        self.normal_hr_max = 100
        self.normal_rr_min = 12
        self.normal_rr_max = 20

    def classify_mood(self, rppg_features, respiration_features):
        """
        Mengklasifikasikan mood berdasarkan fitur rPPG dan respirasi.

        Args:
            rppg_features (dict): Kamus fitur rPPG (e.g., 'heart_rate_bpm', 'hrv_sdnn').
            respiration_features (dict): Kamus fitur respirasi (e.g., 'respiration_rate_bpm', 'respiration_amplitude').

        Returns:
            tuple: (str: Mood yang terdeteksi, float: Persentase keyakinan).
        """
        heart_rate = rppg_features.get('heart_rate_bpm', 0.0)
        hrv_sdnn = rppg_features.get('hrv_sdnn', 0.0)
        respiration_rate = respiration_features.get('respiration_rate_bpm', 0.0)
        respiration_amplitude = respiration_features.get('respiration_amplitude', 0.0)

        mood = "Tidak Diketahui"
        confidence = 0.5 # Default confidence jika tidak ada aturan yang cocok

        # Aturan sederhana untuk klasifikasi mood
        # Ini adalah contoh dan perlu disempurnakan dengan data dan model ML yang sebenarnya

        if heart_rate == 0.0 or respiration_rate == 0.0:
            mood = "Menunggu Data..."
            confidence = 0.0 # Confidence rendah jika data belum lengkap
        elif heart_rate > self.normal_hr_max and respiration_rate > self.normal_rr_max:
            mood = "Stres/Cemas"
            confidence = 0.9
        elif heart_rate < self.normal_hr_min and respiration_rate < self.normal_rr_min:
            mood = "Sangat Tenang"
            confidence = 0.8
        elif self.normal_hr_min <= heart_rate <= self.normal_hr_max and \
             self.normal_rr_min <= respiration_rate <= self.normal_rr_max:
            mood = "Netral/Tenang"
            confidence = 0.75
        elif heart_rate > self.normal_hr_max:
            mood = "Aktivitas Tinggi/Agitasi"
            confidence = 0.7
        elif respiration_rate > self.normal_rr_max:
            mood = "Napas Cepat/Panik"
            confidence = 0.7

        # HRV yang lebih tinggi umumnya dikaitkan dengan relaksasi
        if hrv_sdnn > 0.05 and mood == "Netral/Tenang": # Contoh ambang batas
            mood = "Tenang"
            confidence = max(confidence, 0.85) # Tingkatkan confidence
        elif hrv_sdnn < 0.02 and mood == "Netral/Tenang":
            mood = "Sedikit Stres"
            confidence = min(confidence, 0.6) # Turunkan confidence

        # Amplitudo pernapasan yang rendah bisa menandakan napas dangkal (stres)
        if respiration_amplitude < 5 and mood == "Netral/Tenang": # Contoh ambang batas
             mood = "Mungkin Stres (napas dangkal)"
             confidence = min(confidence, 0.6)

        return mood, confidence

if __name__ == "__main__":
    # Contoh penggunaan:
    classifier = MoodClassifier()

    # Skenario 1: Data normal
    rppg_feat_normal = {'heart_rate_bpm': 75.0, 'hrv_sdnn': 0.03}
    resp_feat_normal = {'respiration_rate_bpm': 16.0, 'respiration_amplitude': 15.0}
    mood1, conf1 = classifier.classify_mood(rppg_feat_normal, resp_feat_normal)
    print(f"Fitur Normal -> Mood: {mood1} ({conf1*100:.1f}%)")

    # Skenario 2: Stres
    rppg_feat_stress = {'heart_rate_bpm': 110.0, 'hrv_sdnn': 0.01}
    resp_feat_stress = {'respiration_rate_bpm': 25.0, 'respiration_amplitude': 8.0}
    mood2, conf2 = classifier.classify_mood(rppg_feat_stress, resp_feat_stress)
    print(f"Fitur Stres -> Mood: {mood2} ({conf2*100:.1f}%)")

    # Skenario 3: Sangat Tenang
    rppg_feat_calm = {'heart_rate_bpm': 55.0, 'hrv_sdnn': 0.06}
    resp_feat_calm = {'respiration_rate_bpm': 10.0, 'respiration_amplitude': 20.0}
    mood3, conf3 = classifier.classify_mood(rppg_feat_calm, resp_feat_calm)
    print(f"Fitur Tenang -> Mood: {mood3} ({conf3*100:.1f}%)")

    # Skenario 4: Tidak ada data
    rppg_feat_empty = {}
    resp_feat_empty = {}
    mood4, conf4 = classifier.classify_mood(rppg_feat_empty, resp_feat_empty)
    print(f"Fitur Kosong -> Mood: {mood4} ({conf4*100:.1f}%)")
