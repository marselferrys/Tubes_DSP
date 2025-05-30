<img src="Realtime rPPG and Respiration Signal.png" width="100%" />

<div align="center">
   
# **Realtime rPPG & Respiration Visualizer**
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
   git clone https://github.com/marselferrys/Tubes_DSP.git
   cd Tubes_DSP
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

| No. | Library | Kegunaan dalam Program |
|:---:|:---|:---|
| 1 | **`opencv`** | Berperan sebagai **sumber data visual utama**. Fungsinya untuk mengakuisisi *frame* video secara *real-time* dari webcam (`cv2.VideoCapture`). Selain itu, digunakan untuk pemrosesan gambar awal seperti mengubah ruang warna (`cv2.cvtColor`) dan menyesuaikan ukuran *frame* (`cv2.resize`) sebelum diolah lebih lanjut atau ditampilkan. |
| 2 | **`numpy`** | Merupakan **fondasi untuk semua operasi data numerik**. Library ini digunakan untuk merepresentasikan *frame* gambar sebagai *array* multidimensi, melakukan kalkulasi matematis pada nilai piksel (seperti rata-rata RGB pada ROI), dan menyusun data deret waktu (sinyal) yang menjadi input bagi Matplotlib dan SciPy. |
| 3 | **`mediapipe`** | Berfungsi sebagai **mesin deteksi cerdas berbasis *machine learning***. Dalam program ini, MediaPipe secara spesifik digunakan untuk dua tugas: 1) Mendeteksi *landmark* wajah untuk menentukan *Region of Interest* (ROI) pada dahi untuk analisis rPPG, dan 2) Mendeteksi *landmark* bahu untuk mengukur pergerakan periodik yang diasosiasikan dengan sinyal pernapasan. |
| 4 | **`pygame`** | Bertindak sebagai **lapisan presentasi (GUI)** aplikasi. Fungsinya adalah untuk membuat jendela utama, menangani interaksi pengguna (seperti menekan tombol 'q' untuk keluar), dan yang terpenting, menggabungkan (*blit*) beberapa sumber visual yaitu *surface* dari *frame* video (OpenCV) dan *surface* dari gambar grafik (Matplotlib) menjadi satu tampilan yang koheren. |
| 5 | **`matplotlib`** | Digunakan sebagai **mesin visualisasi data**. Secara spesifik, fungsinya untuk mengubah data sinyal rPPG dan respirasi menjadi plot atau grafik. Melalui modul `backend_agg`, grafik tersebut tidak ditampilkan di jendela baru, melainkan digambar ke *buffer* memori internal yang kemudian dikonversi menjadi *array* NumPy, sehingga bisa ditampilkan di dalam jendela Pygame. |
| 6 | **`scipy`** | Merupakan *toolkit* utama untuk **pemrosesan sinyal digital (DSP)**. Library ini digunakan untuk membersihkan dan menganalisis sinyal mentah. Fungsi utamanya adalah mendesain dan menerapkan *filter* digital (seperti Butterworth *band-pass filter*) untuk mengisolasi frekuensi detak jantung, serta untuk mendeteksi puncak (*peaks*) pada sinyal rPPG yang telah difilter untuk menghitung *Heart Rate*. |

---

## **Dokumentasi Hasil**

<img src="demo.png" width="100%" />

---

## **Logbook**

## <img src="Images/Mentahan/Panah.svg" width="30px;"/> **Weekly Logbook**
| Week | Task | Person | Status |
| :---: | :---: | :---: | :---: |
| Week 1 | - Create repository <br> - understand, brainstrom, and learn how to build the final task  | Marchel  | Done |
| Week 2 | - Develop resp.py, rppg.py, and main.py <br> - debugging the code error | Marchel & Arof | Done |
| Week 3 | - Build the report and finalize the code <br> | Arof & Marchel | Done |
