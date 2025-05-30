<div align="center">
   
# **Realtime rPPG and Respiration Signal**
# Tugas Besar Mata Kuliah Pengolahan Sinyal Digital (IF3024)

### Dosen Pengampu: **Martin C.T Manullang, S.T., M.T., PhD.**
---

| No  | Nama                     | NIM       | Username         |
| --- | ----------------------   | --------- | ---------------- |
| 1   | Marchel Ferry Timoteus S | 121140195 | marselferrys     |
| 2   | Arof Andestama           | 121140182 | Arof182      |

</div>

---

## **Deskripsi**

Program ini merupakan sistem yang digunakan untuk menangkap sinyal remote Photoplethysmography (rPPG) serta sinyal pernapasan secara real-time menggunakan webcam. Pengambilan sinyal rPPG dilakukan melalui deteksi wajah dengan bantuan Mediapipe, kemudian menganalisis sinyal RGB pada wilayah tertentu di wajah (Region of Interest/ROI) menggunakan algoritma POS (Plane-Orthogonal-to-Skin). Sementara itu, sinyal pernapasan diperoleh dengan mendeteksi perubahan antar frame melalui metode Optical Flow.

## **Fitur**

1. Menampilkan visualisasi hasil pembacaan sinyal rPPG dan denyut jantung per menit
2. Menampilkan visualisasi hasil pembacaan sinyal Respiration
---

## **Instuksi Instalasi**

1. Siapkan code editor pilihan
2. install python (versi 3.10.16)
3. buat folder projek dan clone repositori
   ```bash
   cd projek
   git clone https://github.com/marselferrys/Tubes_DSP.git
   ```
4. install file dependency
   ```
   pip install -r requirements.txt
   ```
5. Setelah semua file terintstall run program pada file main.py
   ```
   python main.py
   ```

## 🌐 Teknologi yang Digunakan

Berikut adalah teknologi dan alat yang digunakan dalam proyek ini:

| Logo                                                                                                                          | Nama Teknologi | Fungsi                                                                           |
| ------------------------------------------------------------------------------------------------------------------------------ | -------------- | -------------------------------------------------------------------------------- |
| <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" style="width:50px;" alt="Python Logo" width="60">            | Python         | Bahasa pemrograman utama untuk pengembangan filter.                              |
| <img src="https://upload.wikimedia.org/wikipedia/commons/9/9a/Visual_Studio_Code_1.35_icon.svg" style="width:50px;" alt="VS Code Logo" width="60"> | VS Code        | Editor teks untuk mengedit skrip secara efisien dengan dukungan ekstensi Python. |

---

## **Library yang dibutuhkan**

| No. | Library      | Kegunaan dalam Program                                                                 |
|-----|--------------|-----------------------------------------------------------------------------------------|
| 1   | `opencv`     | Digunakan untuk menangkap video dari webcam, menampilkan video frame, dan manipulasi gambar (frame) secara real-time. Fungsi penting seperti `cv2.VideoCapture`, `cv2.imshow`, `cv2.cvtColor`, dan `cv2.putText` berasal dari sini. |
| 2   | `numpy`      | Digunakan untuk pengolahan array numerik, menghitung rata-rata sinyal RGB dari ROI wajah, serta menyusun data sinyal untuk pemrosesan rPPG. |
| 3   | `mediapipe`  | Digunakan untuk mendeteksi pose tubuh (bahu kiri dan kanan) untuk respirasi, dan deteksi wajah (bounding box) untuk mendapatkan ROI wajah untuk rPPG. |
| 4   | `OpenCV`     | Sama seperti `opencv`, hanya disebut dengan nama lengkap. Ini adalah library yang sama (`cv2`), digunakan untuk akses webcam, manipulasi dan tampilan gambar/video. |
| 5   | `matplotlib` | Digunakan untuk membuat grafik sinyal rPPG dan respirasi secara real-time yang kemudian disimpan sebagai gambar (`.png`) dan ditampilkan di bawah frame video. |
| 6   | `scipy`      | Digunakan untuk pemrosesan sinyal, termasuk desain dan penerapan filter Butterworth (`scipy.signal.butter`, `scipy.signal.filtfilt`) serta deteksi puncak sinyal (`scipy.signal.find_peaks`). |

---

## **Dokumentasi Hasil**


---

## **Logbook**

## <img src="Images/Mentahan/Panah.svg" width="30px;"/> **Weekly Logbook**
| Week | Task | Person | Status |
| :---: | :---: | :---: | :---: |
| Week 1 | - Create repository <br> - understand, brainstrom, and learn how to build the final task  | Marchel  | Done |
| Week 2 | - Develop resp.py, rppg.py, and main.py <br> - debugging the code error | Marchel & Arof | Done |
| Week 3 | - Build the report and finalize the code <br> | Arof & Marchel | Done |
