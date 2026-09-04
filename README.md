# Gen-Z Social Media Behavioral Archetypes & Well-being Profiling

Studi segmentasi analitik berbasis *unsupervised machine learning* pada 1.000.000 data pengguna Generasi Z (usia 13–27 tahun). Proyek ini memetakan arketipe perilaku digital laten tanpa label terarah (*unlabeled data*), kemudian menguji secara empiris hubungannya terhadap tingkat adiksi dan skor kesejahteraan mental (*mental well-being*).

---

## Daftar Isi
- [Latar Belakang & Metodologi](#latar-belakang--metodologi)
- [Ringkasan Arketipe Persona](#ringkasan-arketipe-persona)
- [Visualisasi & Temuan Kunci](#visualisasi--temuan-kunci)
- [Tech Stack & Arsitektur Data](#tech-stack--arsitektur-data)
- [Instalasi & Panduan Menjalankan](#instalasi--panduan-menjalankan)
- [Keahlian & Nilai Pembelajaran](#keahlian--nilai-pembelajaran)
- [Kesimpulan Strategis](#kesimpulan-strategis)


---
## Struktur Folder & Penempatan Data
```bash
gen-z-behavioral-clustering/
├── data/
│   ├── raw/
│   │   └── genz_social_media_usage_1M.csv
│   └── processed/
├── docs/
│   └── images/
├── models/
├── notebooks/
│   └── 01_eda_clustering.ipynb
├── requirements.txt
└── README.md
```

---

## Latar Belakang & Metodologi

Perdebatan mengenai dampak media sosial terhadap Gen-Z sering kali hanya berpusat pada metrik durasi (*screen time*). Proyek ini membongkar perilaku tersebut ke level granular dengan mempertimbangkan:
1. **Volume & Ritme Penggunaan:** Total jam per hari vs durasi satu sesi penggunaan.
2. **Timing Penggunaan:** Rasio *screen time* tepat sebelum tidur dan penggunaan larut malam (*night usage*).
3. **Fragmentasi Perhatian:** Frekuensi buka-tutup aplikasi (*session frequency*) dan dispersi jumlah platform.

### Alur Pemodelan
- **Data Engineering:** Ingesti 1 juta baris data mentah CSV ke format kolumnar Apache Parquet via DuckDB untuk efisiensi komputasi dan memori.
- **Preprocessing:** Standardisasi skala fitur numerik perilaku menggunakan `StandardScaler`.
- **Clustering:** Evaluasi klaster optimal menggunakan *Elbow Method* dengan algoritma `MiniBatchKMeans` ($k=4$).
- **Manifold Learning:** Proyeksi topologi berdimensi tinggi ke ruang 2D via UMAP (*Uniform Manifold Approximation and Projection*) untuk inspeksi struktur visual.
- **Validasi Eksternal:** Kolom `addiction_level` dan `mental_health_score` sengaja dipisahkan dari proses clustering dan hanya digunakan sebagai metrik validasi post-hoc.

---

## Ringkasan Arketipe Persona

| Persona | Profil Perilaku Digital | Tingkat Adiksi | Rata-rata Skor Mental (1–10) | Karakteristik Utama |
|---|---|---|:---:|---|
| **The Mindful Minimalist** | Sesi terencana, *screen time* sebelum tidur sangat rendah, tanpa kebiasaan malam. | **94.4% Low** / 0% High | **~8.2** | Memiliki batasan digital yang kuat dan higienitas tidur terjaga. |
| **The Casual Browser** | Durasi moderat, durasi sesi menengah, frekuensi harian wajar. | **65.9% Medium** / 19.1% Low | **~6.1** | Pengguna pasif; mengonsumsi konten tanpa ketergantungan ritme sirkadian. |
| **The Hyper-Connected Micro-Checker** | Frekuensi buka aplikasi sangat tinggi, sesi pendek terpecah-pecah di berbagai platform. | **67.4% Medium** / 14.6% High | **~5.4** | Perhatian terfragmentasi secara konstan (*FOMO-driven checking behavior*). |
| **The Late-Night Doomscroller** | Durasi harian tinggi, aktif larut malam, *screen time* intensif sebelum tidur. | **37.6% High** / 5.1% Low | **~3.8** | Ketergantungan stimulasi malam hari dengan disrupsi ritme istirahat ekstrem. |

---

## Visualisasi & Temuan Kunci

### 1. 2D UMAP Projection of Behavioral Archetypes
![UMAP Projection](<img width="1514" height="418" alt="Screenshot 2026-09-04 093436" src="https://github.com/user-attachments/assets/f3ddd50d-cfaf-421c-bca5-ea7e5b78dae6" />)
*Keterangan Gambar: Proyeksi 2D menggunakan UMAP (50.000 sampel data). Terlihat pemisahan klaster yang terpolarisasi menjadi dua domain utama yang dipicu oleh perilaku `night_usage`, kemudian terpecah lebih lanjut berdasarkan intensitas durasi harian dan sesi konsumsi konten.*

### 2. Radar Chart Profil DNA Arketipe
![Radar Profile](docs/images/radar_profile.png)
*Keterangan Gambar: Matriks multivariat rata-rata perilaku tiap persona. 'The Late-Night Doomscroller' menempati spektrum tertinggi pada volume jam harian dan durasi sebelum tidur, berkorelasi terbalik dengan skor kesehatan mental.*

### 3. Distribusi Persona Lintas Kelompok Umur
![Age Distribution](docs/images/age_distribution.png)
*Keterangan Gambar: 100% Stacked Bar Chart distribusi persona pada kelompok Adolescents (13–17), College Age (18–22), dan Young Pros (23–27). Hasil menunjukkan sebaran proporsi yang identik lintas rentang umur Gen-Z.*

### 4. Afinitas Platform & Pemetaan Lintas Negara
![Platform & Country Distribution](docs/images/platform_affinity.png)
*Keterangan Gambar: Heatmap preferensi platform utama dan segmentasi geografis 10 negara terbesar. Tidak ada platform tunggal atau negara spesifik yang memonopoli persona berisiko tinggi.*

---

## Tech Stack & Arsitektur Data

- **Core Runtime:** Python 3.10+
- **Data Querying & OLAP Engine:** DuckDB, Polars, Apache Arrow
- **Data Transformation & Scientific Computing:** Pandas, NumPy
- **Machine Learning & Preprocessing:** Scikit-Learn (`MiniBatchKMeans`, `StandardScaler`)
- **Manifold Learning:** UMAP (`umap-learn`)
- **Visualisasi Data:** Plotly Express / Graph Objects, Seaborn, Matplotlib
- **Artefak Serialisasi:** Joblib

---

## Instalasi & Panduan Menjalankan

### Prasyarat
- Git
- Python 3.10 atau versi yang lebih baru

### 1. Clone Repositori
```bash
git clone [https://github.com/SamMorales11/gen-z-behavioral-clustering.git](https://github.com/SamMorales11/gen-z-behavioral-clustering.git)
cd gen-z-behavioral-clustering
```
### 2. Konfigurasi Virtual Environment
```bash
python -m venv venv
.\venv\Scripts\Activate
```
### 3. Instalasi Dependensi
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
### 4. Instalasi Dependensi
```bash
jupyter notebook notebooks/01_eda_clustering.ipynb
```
