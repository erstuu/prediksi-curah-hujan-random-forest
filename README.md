# 🌧️ Prediksi Curah Hujan Harian Kota Sukabumi — Random Forest (CRISP-DM)

> **Notebook:** `[2]_02_crisp_dm_prediksi_curah_hujan.ipynb`
>
> Model machine learning untuk memprediksi curah hujan harian (RR, dalam mm) di Kota Sukabumi menggunakan algoritma **Random Forest Regressor**, dikembangkan mengikuti metodologi **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*) secara end-to-end.

---

## 📋 Daftar Isi

- [Gambaran Umum](#-gambaran-umum)
- [Metodologi CRISP-DM](#-metodologi-crisp-dm)
  - [Fase 1 — Business Understanding](#fase-1--business-understanding)
  - [Fase 2 — Data Understanding](#fase-2--data-understanding)
  - [Fase 3 — Data Preparation](#fase-3--data-preparation)
  - [Fase 4 — Modeling](#fase-4--modeling)
  - [Fase 5 — Evaluation](#fase-5--evaluation)
  - [Fase 6 — Deployment](#fase-6--deployment)
- [Struktur File Proyek](#-struktur-file-proyek)
- [Dataset](#-dataset)
- [Fitur Model (Feature Set)](#-fitur-model-feature-set)
- [Arsitektur Pipeline](#-arsitektur-pipeline)
- [Hasil & Performa Model](#-hasil--performa-model)
- [Cara Penggunaan](#-cara-penggunaan)
  - [Prasyarat](#prasyarat)
  - [Instalasi](#instalasi)
  - [Menjalankan Notebook](#menjalankan-notebook)
  - [Menggunakan Model yang Tersimpan](#menggunakan-model-yang-tersimpan)
- [Contoh Prediksi](#-contoh-prediksi)
- [Keterbatasan & Catatan Penting](#-keterbatasan--catatan-penting)
- [Referensi](#-referensi)

---

## 🔍 Gambaran Umum

Proyek ini membangun model prediksi curah hujan harian untuk **Kota Sukabumi** menggunakan data meteorologi dari **BMKG** (*Badan Meteorologi, Klimatologi, dan Geofisika*) rentang tahun **2020–2025**. Algoritma yang digunakan adalah **Random Forest Regressor** dari library Scikit-Learn.

### Konteks Permasalahan

Curah hujan merupakan variabel cuaca kritis yang berdampak langsung pada sektor pertanian, mitigasi bencana (banjir, tanah longsor), dan perencanaan infrastruktur. Prediksi curah hujan yang akurat berbasis data historis memungkinkan peringatan dini dan pengambilan keputusan yang lebih baik.

### Tujuan Utama

| Aspek | Deskripsi |
|---|---|
| **Tujuan bisnis/penelitian** | Membangun model prediksi curah hujan harian (RR, mm) di Kota Sukabumi berdasarkan variabel cuaca harian |
| **Tujuan data mining** | Regresi — memprediksi nilai numerik RR dari fitur cuaca, dievaluasi dengan MAE, RMSE, dan R² |
| **Kriteria keberhasilan** | Model dengan R² setinggi mungkin pada data uji, terutama pada subset **BMKG_OBSERVASI** |
| **Algoritma** | Random Forest Regressor (Scikit-Learn) |
| **Metodologi** | CRISP-DM (6 fase lengkap) |

---

## 🔄 Metodologi CRISP-DM

Notebook ini mengimplementasikan seluruh 6 fase CRISP-DM secara sistematis dan terdokumentasi.

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   1. Business    │────▶│  2. Data         │────▶│  3. Data         │
│   Understanding  │     │  Understanding   │     │  Preparation     │
└──────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                           │
┌──────────────────┐     ┌──────────────────┐     ┌────────▼─────────┐
│  6. Deployment   │◀────│  5. Evaluation   │◀────│  4. Modeling     │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

---

### Fase 1 — Business Understanding

**Tujuan:** Mendefinisikan tujuan bisnis/penelitian dan menerjemahkannya menjadi tujuan data mining.

- **Tujuan bisnis/penelitian:** Membangun model prediksi curah hujan harian (RR, dalam mm) di Kota Sukabumi menggunakan algoritma Random Forest, berdasarkan variabel cuaca harian (suhu, kelembapan, penyinaran matahari, kecepatan angin).
- **Tujuan data mining:** Regresi — memprediksi nilai numerik RR dari fitur cuaca lain, dievaluasi dengan MAE, RMSE, dan R².
- **Kriteria keberhasilan:** Model dengan R² setinggi mungkin pada data uji, terutama pada subset `BMKG_OBSERVASI` (data pengukuran langsung) — karena itu yang mencerminkan performa di dunia nyata, bukan pada data hasil augmentasi statistik.

---

### Fase 2 — Data Understanding

**Tujuan:** Memahami data yang tersedia, mengidentifikasi masalah kualitas, dan melakukan eksplorasi awal.

#### 2.1 Import Library

Library utama yang digunakan:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

#### 2.2 Pengumpulan Data (*Collect Initial Data*)

- **Sumber data:** `dataset_bmkg_2020_2025.csv` (output dari Notebook 1: `01_transformasi_data_bmkg.ipynb`)
- **Rentang waktu:** 2020–2025 (2194 baris)
- **Kolom `SUMBER`:** Menandai asal data:
  - `BMKG_OBSERVASI` — pengukuran langsung dari stasiun BMKG (~25%)
  - `BMKG_AUGMENTASI` — estimasi statistik berbasis pola BMKG_OBSERVASI (~75%)

#### 2.3 Deskripsi dan Verifikasi Kualitas Data

Pemeriksaan meliputi:
- `df.info()` — tipe data dan non-null count
- Missing value per kolom
- Duplikasi tanggal
- Proporsi sumber data (`BMKG_OBSERVASI` vs `BMKG_AUGMENTASI`)
- Statistik deskriptif (`describe()`) untuk semua kolom numerik

#### 2.4 Deteksi Potensi Anomali

Meskipun dataset sudah melalui pembersihan di Notebook 1, verifikasi ulang dilakukan secara independen:

| Pemeriksaan | Deskripsi |
|---|---|
| RR negatif | Curah hujan tidak mungkin negatif |
| RH_AVG di luar 0–100% | Kelembapan relatif harus dalam rentang 0–100% |
| SS negatif | Penyinaran matahari tidak mungkin negatif |
| Kode error BMKG (8888/9999) | Penanda data tidak tersedia/error dari BMKG |

#### 2.5 Eksplorasi Visual

Visualisasi yang dihasilkan:
1. **Time Series Plot** — Curah hujan harian (RR) 2020–2025
2. **Scatter Plot Sumber Data** — Distribusi temporal `BMKG_OBSERVASI` (merah) vs `BMKG_AUGMENTASI` (abu-abu)
3. **Box Plot Bulanan** — Distribusi RR per bulan (pola musiman)
4. **Heatmap Korelasi** — Korelasi antar fitur cuaca numerik

**Temuan utama:**
- Dataset lengkap 2020–2025 (2194 baris)
- ±75% baris `BMKG_AUGMENTASI`, ±25% `BMKG_OBSERVASI`
- Tidak ditemukan anomali fisik (hasil pembersihan Notebook 1 konsisten)

---

### Fase 3 — Data Preparation

**Tujuan:** Menyiapkan data mentah menjadi dataset siap model melalui sub-tahapan standar CRISP-DM.

> Fase ini dikerjakan secara **independen** — tidak berasumsi input sudah "siap pakai" dari notebook sebelumnya.

#### 3.1 Select Data

Pemilihan kolom yang relevan:

| Kategori | Kolom | Peran |
|---|---|---|
| **Metadata** | `TANGGAL`, `SUMBER` | Pelacakan temporal & validasi (bukan fitur model) |
| **Fitur Mentah** | `TN`, `TX`, `TAVG`, `RH_AVG`, `SS`, `FF_X`, `FF_AVG` | Kandidat variabel prediktor |
| **Target** | `RR` | Variabel curah hujan yang diprediksi |

#### 3.2 Clean Data

Langkah pembersihan dijalankan ulang secara eksplisit:

1. **Hapus duplikat tanggal** — Mempertahankan kemunculan pertama
2. **Konversi kode error BMKG** — Mengubah `8888`/`9999` menjadi `NaN`
3. **Clip nilai ke rentang fisik valid:**
   - `RH_AVG`: 0–100%
   - `RR`: ≥ 0 mm
   - `SS`: ≥ 0 jam
4. **Imputasi missing value** — Interpolasi berbasis waktu (`method='time'`)
5. **Pembulatan** — `FF_X` dan `FF_AVG` dibulatkan ke bilangan bulat

#### 3.3 Construct Data (Feature Engineering)

Fitur bulan diubah menjadi representasi **siklikal** menggunakan transformasi sinus-kosinus:

```python
df_model['bulan_sin'] = np.sin(2 * np.pi * df_model['bulan'] / 12)
df_model['bulan_cos'] = np.cos(2 * np.pi * df_model['bulan'] / 12)
```

**Alasan:** Agar model memahami bahwa Desember (12) dan Januari (1) itu "berdekatan" secara musiman, bukan berjauhan seperti representasi angka biasa.

**Eksperimen yang dicoba tapi tidak dipakai:**
- Fitur *lag* (RR hari sebelumnya, rata-rata bergulir 3 hari)
- Hasilnya R² justru **turun** (dari ~0.75 ke ~0.63)
- **Penyebab:** ±75% data adalah `BMKG_AUGMENTASI` yang disampel independen per hari — fitur lag pada bagian ini hanya noise, bukan sinyal asli

#### 3.4 Format Data — Split Data Latih & Uji

Pembagian data menggunakan **Time-Based Split** (bukan random split):

| Set | Proporsi | Deskripsi |
|---|---|---|
| **Training** | 80% | Tanggal paling awal |
| **Testing** | 20% | Tanggal paling akhir |

**Alasan:**
- Data deret waktu → split berbasis kronologis mencegah kebocoran informasi temporal
- Lebih mencerminkan skenario nyata: memprediksi masa depan dari data historis
- Data uji secara alami didominasi `BMKG_OBSERVASI` (data asli) karena berada di ujung akhir garis waktu

---

### Fase 4 — Modeling

**Tujuan:** Membangun, melatih, dan mengoptimalkan model prediktif.

#### 4.1 Regularisasi pada Random Forest

Random Forest **tidak** memiliki regularisasi L1/L2 seperti regresi linear. Kontrol kompleksitas dilakukan melalui hyperparameter:

| Hyperparameter | Fungsi |
|---|---|
| `n_estimators` | Jumlah pohon — makin banyak makin stabil, tapi ada titik jenuh |
| `max_depth` | Batas kedalaman tiap pohon — makin dangkal, makin sederhana |
| `min_samples_leaf` | Jumlah data minimum di tiap daun — makin besar, makin general |
| `min_samples_split` | Jumlah data minimum untuk membelah node |
| `max_features` | Jumlah fitur per split — mengontrol keberagaman antar pohon |

#### 4.2 Model Baseline

Model baseline menggunakan parameter default Scikit-Learn (hanya `random_state=42` untuk reprodusibilitas).

#### 4.2b Model Pembanding — Linear Regression

Sebagai baseline konvensional, dilatih model `LinearRegression` menggunakan fitur dan pembagian data yang **identik** (apple-to-apple comparison).

#### 4.3 Hyperparameter Tuning

Metode tuning yang digunakan:

| Komponen | Pilihan | Alasan |
|---|---|---|
| **Pencarian** | `RandomizedSearchCV` (50 iterasi) | Efisien pada ruang pencarian besar |
| **Cross-Validation** | `TimeSeriesSplit(n_splits=5)` | Menjaga urutan kronologis, mencegah temporal data leakage |
| **Scoring** | R² | Metrik utama evaluasi |

**Ruang pencarian hyperparameter:**

```python
param_distributions = {
    'n_estimators': [100, 150, 200, 300, 400, 500],
    'max_depth': [None, 10, 14, 18, 22],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.5, 1.0],
}
```

> **Penting:** Seluruh proses pencarian hanya menggunakan `X_train`/`y_train`; data uji sama sekali tidak disentuh sampai model final terpilih.

#### 4.4 Model Final

Model final dilatih ulang pada seluruh data latih menggunakan konfigurasi terbaik hasil tuning:

```python
model_rf = RandomForestRegressor(random_state=42, n_jobs=-1, **best_cfg)
model_rf.fit(X_train, y_train)
```

---

### Fase 5 — Evaluation

**Tujuan:** Mengevaluasi performa model secara komprehensif dari berbagai sudut pandang.

#### 5.1 Perbandingan Model (Seluruh Data Uji)

Tiga model dibandingkan secara head-to-head:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression (Baseline Pembanding) | — | — | — |
| Random Forest (Baseline Default) | — | — | — |
| **Random Forest (Tuned Final)** | — | — | — |

> *Nilai spesifik dihasilkan saat menjalankan notebook.*

Visualisasi pendukung:
- **Scatter plot Aktual vs Prediksi** — dengan garis diagonal sempurna (y = x)

#### 5.2 Validasi pada BMKG_OBSERVASI

Evaluasi khusus pada subset data uji yang hanya berisi data pengukuran langsung BMKG:

> Ini metrik yang **paling jujur** untuk mengukur performa di dunia nyata, karena hanya memakai baris pengukuran langsung BMKG (bukan hasil augmentasi statistik).

**Catatan:** Karena split berbasis kronologis dan data `BMKG_OBSERVASI` berada di ujung akhir garis waktu (Jul 2024 – Jan 2026), data uji otomatis didominasi baris observasi asli. Ini membuat evaluasi yang dilaporkan representatif untuk data pengukuran nyata.

#### 5.3 Evaluasi Hujan Lebat dan Ekstrem (RR > 50 mm/hari)

Pengujian khusus pada kejadian curah hujan lebat/ekstrem sesuai standar klasifikasi BMKG:

- **Ambang batas:** RR > 50 mm/hari
- **Tujuan:** Memastikan model tidak hanya akurat pada kondisi normal, tetapi juga mampu mengenali lonjakan curah hujan yang berpotensi menyebabkan banjir/longsor

#### 5.4 Feature Importance

Analisis kontribusi setiap fitur terhadap prediksi model menggunakan `feature_importances_` dari Random Forest, divisualisasikan sebagai horizontal bar chart.

---

### Fase 6 — Deployment

**Tujuan:** Menyimpan model dan menyediakan antarmuka prediksi yang siap digunakan.

#### 6.1 Simpan Model Final

Model disimpan dalam dua format:

| File | Format | Ukuran |
|---|---|---|
| `model_random_forest_curah_hujan_sukabumi.pkl` | Python Pickle | ~67.7 MB |
| `model_random_forest_curah_hujan_sukabumi.joblib` | Joblib | ~67.7 MB |

#### 6.2 Fungsi Prediksi

Fungsi siap pakai untuk prediksi curah hujan:

```python
def prediksi_curah_hujan(tn, tx, tavg, rh_avg, ss, ff_x, ff_avg, bulan, model=model_rf):
    """Prediksi curah hujan harian (mm) dari variabel cuaca."""
    bulan_sin = np.sin(2 * np.pi * bulan / 12)
    bulan_cos = np.cos(2 * np.pi * bulan / 12)
    X_baru = pd.DataFrame([{
        'TN': tn, 'TX': tx, 'TAVG': tavg, 'RH_AVG': rh_avg,
        'SS': ss, 'FF_X': ff_x, 'FF_AVG': ff_avg,
        'bulan_sin': bulan_sin, 'bulan_cos': bulan_cos
    }])
    return model.predict(X_baru)[0]
```

---

## 📁 Struktur File Proyek

```
Model/
├── [2]_02_crisp_dm_prediksi_curah_hujan.ipynb   # Notebook utama (CRISP-DM)
├── model_random_forest_curah_hujan_sukabumi.pkl  # Model tersimpan (Pickle)
├── model_random_forest_curah_hujan_sukabumi.joblib # Model tersimpan (Joblib)
├── dataset_bmkg_2020_2025.csv                    # Dataset input (dari Notebook 1)
└── README.md                                     # Dokumentasi ini
```

### Ketergantungan Antar Notebook

```
01_transformasi_data_bmkg.ipynb          [2]_02_crisp_dm_prediksi_curah_hujan.ipynb
┌─────────────────────────┐          ┌──────────────────────────────────────┐
│ Raw BMKG data           │          │ CRISP-DM Pipeline                    │
│ ↓                       │          │ ↓                                    │
│ Transformasi & Cleaning │──CSV──▶  │ Data Understanding → Preparation     │
│ ↓                       │          │ → Modeling → Evaluation → Deployment │
│ dataset_bmkg_2020_2025  │          │ ↓                                    │
└─────────────────────────┘          │ model_random_forest_*.pkl/.joblib    │
                                     └──────────────────────────────────────┘
```

---

## 📊 Dataset

### Sumber

Data bersumber seluruhnya dari **BMKG** (Badan Meteorologi, Klimatologi, dan Geofisika), terdiri dari dua kategori:

| Kategori | Deskripsi | Proporsi |
|---|---|---|
| `BMKG_OBSERVASI` | Pengukuran langsung dari stasiun cuaca BMKG | ~25% |
| `BMKG_AUGMENTASI` | Estimasi statistik yang dibangun dari pola musiman `BMKG_OBSERVASI` | ~75% |

### Spesifikasi Dataset

| Atribut | Detail |
|---|---|
| **Lokasi** | Kota Sukabumi, Jawa Barat, Indonesia |
| **Rentang waktu** | 2020–2025 |
| **Jumlah baris** | 2194 |
| **Resolusi temporal** | Harian |
| **File** | `dataset_bmkg_2020_2025.csv` |

### Kolom Dataset

| Kolom | Deskripsi | Satuan | Tipe |
|---|---|---|---|
| `TANGGAL` | Tanggal pengukuran | — | datetime |
| `SUMBER` | Asal data (BMKG_OBSERVASI / BMKG_AUGMENTASI) | — | kategorikal |
| `TN` | Suhu minimum harian | °C | float |
| `TX` | Suhu maksimum harian | °C | float |
| `TAVG` | Suhu rata-rata harian | °C | float |
| `RH_AVG` | Kelembapan relatif rata-rata | % | float |
| `RR` | Curah hujan harian (**target**) | mm | float |
| `SS` | Penyinaran matahari | jam | float |
| `FF_X` | Kecepatan angin maksimum | knot | int |
| `FF_AVG` | Kecepatan angin rata-rata | knot | int |

---

## 🔢 Fitur Model (Feature Set)

Model final menggunakan **9 fitur** — 7 fitur cuaca asli + 2 fitur rekayasa siklikal:

| # | Fitur | Sumber | Deskripsi |
|---|---|---|---|
| 1 | `TN` | BMKG | Suhu minimum harian (°C) |
| 2 | `TX` | BMKG | Suhu maksimum harian (°C) |
| 3 | `TAVG` | BMKG | Suhu rata-rata harian (°C) |
| 4 | `RH_AVG` | BMKG | Kelembapan relatif rata-rata (%) |
| 5 | `SS` | BMKG | Penyinaran matahari (jam) |
| 6 | `FF_X` | BMKG | Kecepatan angin maksimum (knot) |
| 7 | `FF_AVG` | BMKG | Kecepatan angin rata-rata (knot) |
| 8 | `bulan_sin` | Rekayasa | sin(2π × bulan / 12) — komponen siklikal musiman |
| 9 | `bulan_cos` | Rekayasa | cos(2π × bulan / 12) — komponen siklikal musiman |

**Target:** `RR` — Curah hujan harian (mm)

---

## ⚙️ Arsitektur Pipeline

```
dataset_bmkg_2020_2025.csv
         │
         ▼
  ┌──────────────┐
  │  Select Data │  Pilih kolom relevan (metadata, fitur, target)
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │  Clean Data  │  Hapus duplikat, konversi error codes, clip, imputasi
  └──────┬───────┘
         ▼
  ┌──────────────────┐
  │ Feature Engineer │  Transformasi siklikal bulan (sin/cos)
  └──────┬───────────┘
         ▼
  ┌──────────────────┐
  │ Time-Based Split │  80% train (awal) / 20% test (akhir)
  └──────┬───────────┘
         ▼
  ┌──────────────────────────────────────────┐
  │            Modeling Pipeline             │
  │                                          │
  │  ┌─────────────┐  ┌──────────────────┐   │
  │  │  Baseline   │  │ Baseline LR      │   │
  │  │  RF Default │  │ (Pembanding)     │   │
  │  └─────────────┘  └──────────────────┘   │
  │                                          │
  │  ┌────────────────────────────────────┐  │
  │  │ RandomizedSearchCV                 │  │
  │  │ + TimeSeriesSplit (5 folds)        │  │
  │  │ → 50 iterasi pencarian            │  │
  │  └──────────────┬─────────────────────┘  │
  │                 ▼                        │
  │  ┌────────────────────────────────────┐  │
  │  │ Model Final (Best Hyperparams)    │  │
  │  └────────────────────────────────────┘  │
  └──────────────────┬───────────────────────┘
                     ▼
  ┌──────────────────────────────────────────┐
  │              Evaluation                  │
  │                                          │
  │  • Baseline vs Tuned (MAE, RMSE, R²)    │
  │  • Khusus BMKG_OBSERVASI                │
  │  • Khusus hujan ekstrem (>50 mm)        │
  │  • Feature Importance                    │
  └──────────────────┬───────────────────────┘
                     ▼
  ┌──────────────────────────────────────────┐
  │  model_random_forest_*.pkl / .joblib     │
  └──────────────────────────────────────────┘
```

---

## 📈 Hasil & Performa Model

### Metrik Evaluasi

| Metrik | Deskripsi |
|---|---|
| **MAE** (*Mean Absolute Error*) | Rata-rata selisih absolut prediksi vs aktual (mm) |
| **RMSE** (*Root Mean Squared Error*) | Akar rata-rata kuadrat error — sensitif terhadap error besar (mm) |
| **R²** (*Coefficient of Determination*) | Proporsi variansi target yang dijelaskan model (0–1, makin tinggi makin baik) |

### Perbandingan Model

Tiga model dibandingkan secara apple-to-apple pada data uji yang sama:

| Model | Keterangan |
|---|---|
| **Linear Regression** | Baseline konvensional |
| **Random Forest (Default)** | Parameter default Scikit-Learn |
| **Random Forest (Tuned)** | Hyperparameter hasil `RandomizedSearchCV` + `TimeSeriesSplit` |

> ⚠️ Nilai metrik spesifik dihasilkan secara dinamis saat menjalankan notebook, karena bergantung pada versi Scikit-Learn dan hasil randomized search.

### Evaluasi Multi-Sudut Pandang

1. **Seluruh data uji** — evaluasi umum
2. **Khusus BMKG_OBSERVASI** — evaluasi paling jujur (data pengukuran langsung)
3. **Khusus hujan ekstrem (RR > 50 mm)** — evaluasi pada kejadian berisiko tinggi
4. **Feature importance** — kontribusi setiap fitur terhadap prediksi

---

## 🚀 Cara Penggunaan

### Prasyarat

- **Python** ≥ 3.8
- **Jupyter Notebook** atau **JupyterLab** atau **Google Colab**

### Instalasi

Instal seluruh dependensi yang dibutuhkan:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Library dan Versi

| Library | Fungsi |
|---|---|
| `pandas` | Manipulasi dataframe |
| `numpy` | Operasi numerik & transformasi siklikal |
| `matplotlib` | Visualisasi grafik |
| `seaborn` | Visualisasi statistik (heatmap, boxplot) |
| `scikit-learn` | Model Random Forest, metrik evaluasi, tuning, splitting |
| `pickle` / `joblib` | Serialisasi model |

### Menjalankan Notebook

1. Pastikan file `dataset_bmkg_2020_2025.csv` berada di direktori yang sama dengan notebook
2. Buka notebook:

```bash
jupyter notebook "[2]_02_crisp_dm_prediksi_curah_hujan.ipynb"
```

3. Jalankan seluruh sel secara berurutan (*Run All*)

### Menggunakan Model yang Tersimpan

#### Menggunakan Pickle

```python
import pickle
import numpy as np
import pandas as pd

# Muat model
with open('model_random_forest_curah_hujan_sukabumi.pkl', 'rb') as f:
    model = pickle.load(f)

# Siapkan input
bulan = 1  # Januari
X_baru = pd.DataFrame([{
    'TN': 22.5,      # Suhu minimum (°C)
    'TX': 31.0,      # Suhu maksimum (°C)
    'TAVG': 26.5,    # Suhu rata-rata (°C)
    'RH_AVG': 85,    # Kelembapan relatif (%)
    'SS': 3.0,       # Penyinaran matahari (jam)
    'FF_X': 10,      # Kecepatan angin maks (knot)
    'FF_AVG': 2,     # Kecepatan angin rata-rata (knot)
    'bulan_sin': np.sin(2 * np.pi * bulan / 12),
    'bulan_cos': np.cos(2 * np.pi * bulan / 12)
}])

# Prediksi
prediksi = model.predict(X_baru)[0]
print(f'Prediksi curah hujan: {prediksi:.2f} mm')
```

#### Menggunakan Joblib

```python
import joblib

model = joblib.load('model_random_forest_curah_hujan_sukabumi.joblib')
# Gunakan model.predict() seperti di atas
```

---

## 💡 Contoh Prediksi

Contoh pemanggilan fungsi prediksi yang tersedia di notebook:

```python
# Prediksi untuk kondisi cuaca tertentu di bulan Januari
contoh = prediksi_curah_hujan(
    tn=22.5,      # Suhu minimum (°C)
    tx=31.0,      # Suhu maksimum (°C)
    tavg=26.5,    # Suhu rata-rata (°C)
    rh_avg=85,    # Kelembapan relatif (%)
    ss=3.0,       # Penyinaran matahari (jam)
    ff_x=10,      # Kecepatan angin maks (knot)
    ff_avg=2,     # Kecepatan angin rata-rata (knot)
    bulan=1       # Bulan (1=Januari, ..., 12=Desember)
)
print(f'Prediksi curah hujan: {contoh:.2f} mm')
```

### Deskripsi Parameter Input

| Parameter | Deskripsi | Contoh | Satuan |
|---|---|---|---|
| `tn` | Suhu minimum harian | 22.5 | °C |
| `tx` | Suhu maksimum harian | 31.0 | °C |
| `tavg` | Suhu rata-rata harian | 26.5 | °C |
| `rh_avg` | Kelembapan relatif rata-rata | 85 | % |
| `ss` | Lama penyinaran matahari | 3.0 | jam |
| `ff_x` | Kecepatan angin maksimum | 10 | knot |
| `ff_avg` | Kecepatan angin rata-rata | 2 | knot |
| `bulan` | Bulan pengukuran (1–12) | 1 | — |

### Output

- Nilai prediksi curah hujan harian dalam satuan **milimeter (mm)**

---

## ⚠️ Keterbatasan & Catatan Penting

### Keterbatasan Dataset

1. **Dominasi data augmentasi:** ±75% data (2020 – pertengahan 2024) adalah `BMKG_AUGMENTASI`, bukan pengukuran lapangan langsung
2. **Lokasi tunggal:** Model dilatih khusus untuk Kota Sukabumi — tidak dapat langsung diterapkan ke wilayah lain tanpa pelatihan ulang
3. **Resolusi harian:** Model memprediksi curah hujan per hari, bukan per jam atau sub-harian

### Rekomendasi untuk Laporan/Skripsi

- Skor R² **BMKG_OBSERVASI-only** (bagian 5.2) disarankan jadi **acuan utama kesimpulan** penelitian, bukan skor keseluruhan yang tercampur data augmentasi
- Kolom `SUMBER` di dataset final memungkinkan validasi ini diulang kapan saja (misalnya kalau data observasi BMKG bertambah di masa depan)
- Seluruh dataset bersumber dari BMKG — `BMKG_AUGMENTASI` dibangun dari pola musiman `BMKG_OBSERVASI`, bukan dikarang bebas

### Catatan Teknis

- **Fitur lag tidak dipakai:** Eksperimen menambahkan fitur lag (RR hari sebelumnya, moving average 3 hari) menurunkan R² dari ~0.75 ke ~0.63 karena dominasi data augmentasi
- **Random state:** `RANDOM_STATE = 42` digunakan secara konsisten untuk reprodusibilitas
- **Time-based split:** Mencegah temporal data leakage pada data deret waktu
- **TimeSeriesSplit pada CV:** Menjaga urutan kronologis saat cross-validation dalam proses tuning

---

## 📚 Referensi

- **CRISP-DM:** Chapman, P., et al. (2000). *CRISP-DM 1.0: Step-by-step data mining guide*. SPSS Inc.
- **Random Forest:** Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
- **Scikit-Learn:** Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825–2830.
- **BMKG:** Badan Meteorologi, Klimatologi, dan Geofisika — [https://dataonline.bmkg.go.id](https://dataonline.bmkg.go.id)

---

## 📝 Lisensi

Proyek ini dikembangkan untuk keperluan akademik (skripsi/tugas akhir). Data cuaca bersumber dari BMKG dan tunduk pada ketentuan penggunaan data BMKG.

---

<div align="center">

**Dikembangkan sebagai bagian dari penelitian prediksi curah hujan Kota Sukabumi**

*Menggunakan metodologi CRISP-DM • Algoritma Random Forest • Data BMKG 2020–2025*

</div>
