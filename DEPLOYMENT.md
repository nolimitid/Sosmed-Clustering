# Panduan Deployment — Sosmed Clustering

Dokumen ini mencakup seluruh langkah untuk menjalankan pipeline clustering media
sosial di server Linux, termasuk setup awal, upload file, menjalankan server,
dan menggunakan UI web.

---

## Daftar Isi

1. [Prasyarat Server](#1-prasyarat-server)
2. [Struktur File di Server](#2-struktur-file-di-server)
3. [Menjalankan di Lokal (Testing)](#3-menjalankan-di-lokal-testing)
4. [Upload File dari Lokal ke Server](#4-upload-file-dari-lokal-ke-server)
5. [Setup Pertama Kali di Server](#5-setup-pertama-kali-di-server)
6. [Menjalankan Server](#6-menjalankan-server)
7. [Menggunakan UI Web](#7-menggunakan-ui-web)
8. [Menjalankan Pipeline Job](#8-menjalankan-pipeline-job)
9. [Menjalankan lewat CLI (Terminal)](#9-menjalankan-lewat-cli-terminal)
10. [Memantau & Mengunduh Hasil](#10-memantau--mengunduh-hasil)
11. [Update Kode](#11-update-kode)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Prasyarat Server

| Kebutuhan | Keterangan |
|---|---|
| OS | Linux (Ubuntu 20.04+ / Debian) |
| Python | 3.12 |
| RAM | Minimal 8 GB; 16 GB+ untuk data jutaan baris |
| Storage | Minimal 20 GB kosong (model + data + hasil) |
| GPU | Opsional tapi sangat disarankan untuk embedding (NVIDIA CUDA) |
| Port | 8880 harus bisa diakses (atau lewat SSH tunnel) |

---

## 2. Struktur File di Server

Semua file aplikasi ditempatkan di `~/app/cluster/`:

```
~/app/cluster/
├── api.py                  ← server FastAPI (+ endpoint UI)
├── cluster.py              ← pipeline clustering satu batch
├── es_fetch.py             ← fetch data dari Elasticsearch
├── merge.py                ← merge centroid semua chunk → global topics
├── pipeline.py             ← orchestrator end-to-end
├── preprocess.py           ← fungsi cleaning teks
├── language_id.py          ← deteksi bahasa
├── slang_lexicon.csv       ← kamus normalisasi slang
├── stopwords_id.txt        ← daftar stopword bahasa Indonesia
├── requirements.txt        ← daftar dependensi Python
├── run.sh                  ← skrip untuk menjalankan server
└── static/
    └── index.html          ← UI web
```

Data output pipeline disimpan di:
```
~/app/cluster/data/<nama_project>/
├── fetch.done
├── chunks/                 ← parquet hasil fetch dari ES
├── cluster_chunk_00000/    ← hasil cluster per chunk
│   ├── cluster.done
│   ├── centroids.npy
│   ├── assignments.csv
│   └── topics_summary.csv
└── merged/                 ← OUTPUT UTAMA
    ├── merge.done
    ├── global_topics_summary.csv
    ├── topic_graph.html
    └── topic_graph_edges.csv
```

---

 ## 3. Menjalankan di Lokal (Testing)

Untuk testing di Windows sebelum deploy ke server. Koneksi ke Elasticsearch
tetap dibutuhkan (VPN / akses jaringan ke `dev.elastic.dashboard.nolimit.id`).

### 3.1 Prasyarat lokal

| Kebutuhan | Keterangan |
|---|---|
| Python | 3.12 (cek dengan `py -3.12 --version`) |
| Git | Untuk clone repo (opsional) |
| RAM | Minimal 8 GB |
| Akses ES | VPN atau jaringan yang bisa reach `dev.elastic.dashboard.nolimit.id` |

### 3.2 Setup virtual environment (sekali saja)

Buka **PowerShell** atau **Command Prompt** di folder project:

```powershell
cd "D:\NoLimit Indonesia - DS\Sosmed-Clustering"

# Cek versi Python — harus menampilkan 3.12.x
py -3.12 --version

# Buat venv dengan Python 3.12
py -3.12 -m venv venv

# Aktifkan venv
venv\Scripts\activate

# Install dependensi
pip install --upgrade pip
pip install -r requirements.txt
```

> **Windows + hdbscan/umap**: bila `pip install` gagal di `hdbscan` atau
> `umap-learn`, install dulu Microsoft C++ Build Tools dari
> https://visualstudio.microsoft.com/visual-cpp-build-tools/ — atau gunakan
> conda:
> ```powershell
> conda create -n cluster-env python=3.12
> conda activate cluster-env
> conda install -c conda-forge hdbscan umap-learn
> pip install -r requirements.txt
> ```

### 3.3 Pre-download model embedding (sekali saja)

```powershell
venv\Scripts\activate
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('LazarusNLP/all-indo-e5-small-v4'); print('OK')"
```

Download sekitar 471 MB — tunggu hingga selesai sebelum menjalankan pipeline.

### 3.4 Jalankan server API + UI lokal

```powershell
venv\Scripts\activate
cd "D:\NoLimit Indonesia - DS\Sosmed-Clustering"
uvicorn api:app --host 127.0.0.1 --port 8880 --reload
```

Buka browser: **`http://localhost:8880/ui`**

Flag `--reload` membuat server otomatis restart bila ada file `.py` yang
berubah — berguna saat development.

### 3.5 Test fetch + pipeline dari ES

```powershell
venv\Scripts\activate
cd "D:\NoLimit Indonesia - DS\Sosmed-Clustering"

# Test fetch 10k data dulu untuk cek koneksi ES
python es_fetch.py ^
    --project-id <uuid-project> ^
    --output-dir data\test_lokal\chunks ^
    --max-docs 10000 ^
    --chunk-size 10000

# Jalankan pipeline 100k (via CLI)
python pipeline.py ^
    --project-id <uuid-project> ^
    --output-dir data\test_lokal ^
    --max-docs 100000 ^
    --chunk-size 100000 ^
    --target-min 20 --target-max 50 ^
    --assign-outliers
```

> **Catatan `^`**: tanda `^` adalah line-continuation di CMD/PowerShell Windows.
> Di PowerShell modern bisa pakai backtick `` ` `` sebagai gantinya.

Atau cukup buka `http://localhost:8880/ui` dan submit form dari browser.

### 3.6 Test cluster satu file JSON (tanpa ES)

Bila hanya ingin test pipeline cluster tanpa fetch dari ES:

```powershell
venv\Scripts\activate
python cluster.py ^
    --input "C:\Users\asus\Downloads\report_raw_posts_e4d55ffd.json" ^
    --text-col content ^
    --output-dir data\test_json ^
    --sample 20000 ^
    --target-min 20 --target-max 50 ^
    --assign-outliers
```

Hasil ada di `data\test_json\assignments.csv` dan `data\test_json\topics_summary.csv`.

### 3.7 Perbedaan lokal vs server

| Aspek | Lokal (Windows) | Server (Linux) |
|---|---|---|
| Path separator | `\` (backslash) | `/` (slash) |
| Venv activate | `venv\Scripts\activate` | `source ~/pyvenvs/cluster-venv/bin/activate` |
| GPU | Biasanya tidak ada (CPU mode) | Ada CUDA — embedding jauh lebih cepat |
| Port server | `127.0.0.1:8880` | `0.0.0.0:8880` |
| Kecepatan embedding | ~100–300 teks/detik (CPU) | ~1000–3000+ teks/detik (GPU) |
| Cocok untuk | Testing, debugging, develop | Data skala besar (ratusan ribu–jutaan) |

---

## 4. Upload File dari Lokal ke Server

### Menggunakan `scp` (dari terminal lokal)

```bash
# Upload semua file Python + aset ke server
scp cluster.py merge.py pipeline.py es_fetch.py \
    api.py preprocess.py language_id.py \
    slang_lexicon.csv stopwords_id.txt requirements.txt \
    user@<ip-server>:~/app/cluster/

# Upload folder static (UI)
scp -r static/ user@<ip-server>:~/app/cluster/
```

### Menggunakan `rsync` (lebih efisien untuk update ulang)

```bash
rsync -avz --exclude '__pycache__' --exclude '*.pyc' \
    --exclude 'api_jobs/' --exclude 'data/' --exclude 'results*/' \
    ./ user@<ip-server>:~/app/cluster/
```

### Menggunakan Termius (GUI)

1. Buka tab **SFTP** di Termius
2. Navigasi ke `~/app/cluster/` di sisi server
3. Drag & drop semua file `.py`, `.csv`, `.txt`, folder `static/`

---

## 5. Setup Pertama Kali di Server

Jalankan perintah berikut **satu kali** di server via SSH:

```bash
# 1. Buat folder aplikasi
mkdir -p ~/app/cluster/static
mkdir -p ~/app/cluster/data

# 2. Buat virtual environment (jika belum ada)
mkdir -p ~/pyvenvs
python3.12 -m venv ~/pyvenvs/cluster-venv

# 3. Aktifkan venv dan install dependensi
source ~/pyvenvs/cluster-venv/bin/activate
cd ~/app/cluster
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Opsional tapi disarankan) Pre-download model embedding
#    agar saat pipeline pertama jalan tidak perlu download di tengah proses
python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('LazarusNLP/all-indo-e5-small-v4')
print('Model berhasil diunduh.')
"
```

> **Catatan GPU**: bila server punya GPU NVIDIA, pastikan CUDA driver sudah
> terpasang. PyTorch akan otomatis memakai GPU saat embedding.

---

## 6. Menjalankan Server

### `run.sh` — tidak perlu diubah

UI web sudah terintegrasi dalam server FastAPI yang sama. `run.sh` tetap:

```bash
#!/bin/bash
source ~/pyvenvs/cluster-venv/bin/activate
cd ~/app/cluster
uvicorn api:app --host 0.0.0.0 --port 8880
```

### Cara menjalankan

```bash
# Jalankan langsung (koneksi SSH harus tetap terbuka)
bash ~/app/cluster/run.sh

# Jalankan di background dengan screen (DISARANKAN)
screen -S clustering-server
bash ~/app/cluster/run.sh
# Tekan Ctrl+A lalu D untuk detach
# Kembali dengan: screen -r clustering-server

# Atau dengan nohup
nohup bash ~/app/cluster/run.sh > ~/app/cluster/server.log 2>&1 &
echo "Server PID: $!"
```

### Cek apakah server sudah berjalan

```bash
curl http://localhost:8880/health
# respons: {"status":"ok","jobs_total":0,"running":0,"queued":0,...}
```

---

## 7. Menggunakan UI Web

Buka di browser: **`http://<ip-server>:8880/ui`**

> Bila server tidak bisa diakses langsung (firewall), gunakan SSH tunnel:
> ```bash
> # Di terminal lokal (Windows: PowerShell / Git Bash)
> ssh -L 8880:localhost:8880 user@<ip-server>
> # Lalu buka: http://localhost:8880/ui
> ```

### Tampilan UI

```
┌─────────────────────────────────────────────┐
│ 📊 Sosmed Clustering        Token: [____]   │
├─────────────────────────────────────────────┤
│ ⚡ Buat Pipeline Job Baru                    │
│  Project ID: [______________________________]│
│  Target Data: [100000]  Topik: [20] — [50]  │
│  ▶ Pengaturan Lanjutan                       │
│                          [⚡ Mulai Pipeline] │
├─────────────────────────────────────────────┤
│ 📋 Daftar Job                    [↻ Refresh]│
│  Job ID   | Info        | Status  | Aksi    │
│  ab12cd34 | Pipeline... | Berjalan| [Log][×]│
│  ef56gh78 | Pipeline... | Selesai | [Hasil] │
└─────────────────────────────────────────────┘
```

| Kolom UI | Keterangan |
|---|---|
| **Project ID** | UUID project di Elasticsearch (wajib) |
| **Target Data** | 0 = semua data project; angka lain = batas fetch |
| **Jumlah Topik** | Rentang min–maks topik yang dihasilkan (default 20–50) |
| **Pengaturan Lanjutan** | Model, chunk size, filter bahasa, device, dll. |
| **Token** | Isi jika server memakai `API_TOKEN` env variable |

---

## 8. Menjalankan Pipeline Job

### Lewat UI (cara utama)

1. Buka `http://<ip-server>:8880/ui`
2. Isi **Project ID** (UUID project ES)
3. Set **Target Data** — contoh `100000` untuk test, `0` untuk semua data
4. Set rentang topik (default 20–50 sudah cukup untuk mulai)
5. Klik **⚡ Mulai Pipeline**
6. Job muncul di tabel bawah dengan status **Antri** → **Berjalan**
7. Klik **📄 Log** untuk melihat progress secara live
8. Setelah **Selesai**, klik **📥 Hasil** untuk download output

### Alur pipeline yang berjalan di background

```
Tahap 1: es_fetch.py   — fetch data dari ES ke chunks/
Tahap 2: cluster.py    — cluster tiap chunk (paralel bila multi-worker)
Tahap 3: merge.py      — merge semua centroid → global 20–50 topik
```

Setiap tahap menyimpan **checkpoint** (`.done` file). Bila pipeline gagal di
tengah jalan, jalankan ulang dari UI — tahap yang sudah selesai tidak akan
diproses ulang.

---

## 9. Menjalankan lewat CLI (Terminal)

Untuk kasus advanced atau debugging, semua komponen bisa dijalankan manual:

```bash
source ~/pyvenvs/cluster-venv/bin/activate
cd ~/app/cluster

# Pipeline lengkap (fetch + cluster + merge)
python pipeline.py \
    --project-id e4d55ffd-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
    --output-dir data/project1 \
    --max-docs 100000 \
    --chunk-size 100000 \
    --target-min 20 --target-max 50 \
    --assign-outliers

# Hanya fetch data
python es_fetch.py \
    --project-id <uuid> \
    --output-dir data/project1/chunks \
    --max-docs 100000

# Hanya cluster satu file parquet
python cluster.py \
    --input data/project1/chunks/chunk_00000.parquet \
    --text-col content \
    --output-dir data/project1/cluster_chunk_00000 \
    --target-min 20 --target-max 50 \
    --assign-outliers

# Hanya merge hasil cluster
python merge.py \
    --base-dir data/project1 \
    --output-dir data/project1/merged \
    --target-min 20 --target-max 50

# Lanjutkan pipeline setelah crash (skip tahap yang sudah selesai)
python pipeline.py --project-id <uuid> --output-dir data/project1 --skip-fetch
python pipeline.py --project-id <uuid> --output-dir data/project1 \
    --skip-fetch --skip-cluster
```

---

## 10. Memantau & Mengunduh Hasil

### Lewat UI

- **📄 Log** — log live dari job yang sedang berjalan
- **📥 Hasil** — muncul saat job selesai; berisi daftar semua output file
  - `merged/global_topics_summary.csv` — topik global + keyword ← **output utama**
  - `merged/topic_graph.html` — visualisasi hubungan antar topik (buka di browser)
  - `merged/topic_graph_edges.csv` — edge graph topik
  - `cluster_chunk_NNNNN/topics_summary.csv` — topik per chunk (data lebih detail)

### Lewat curl / CLI

```bash
JOB=ab12cd34ef56

# Status + metrik
curl http://localhost:8880/jobs/$JOB

# Log (200 baris terakhir)
curl "http://localhost:8880/jobs/$JOB/log?tail=200"

# Daftar file output
curl http://localhost:8880/jobs/$JOB/files

# Download hasil utama
curl -OJ http://localhost:8880/jobs/$JOB/files/merged/global_topics_summary.csv
curl -OJ http://localhost:8880/jobs/$JOB/files/merged/topic_graph.html
```

---

## 11. Update Kode

Bila ada perubahan kode (misal update `cluster.py`, `api.py`, `static/index.html`):

```bash
# 1. Upload file yang berubah ke server
scp cluster.py api.py user@<ip-server>:~/app/cluster/
scp static/index.html user@<ip-server>:~/app/cluster/static/

# 2. Restart server
ssh user@<ip-server>
pkill -f "uvicorn api:app"
screen -r clustering-server   # atau buat screen baru
bash ~/app/cluster/run.sh
```

> **UI tidak butuh restart server** bila hanya `static/index.html` yang berubah —
> FastAPI membaca file HTML saat setiap request masuk. Cukup hard-refresh browser
> (Ctrl+Shift+R).
>
> Bila `api.py` atau file Python lain berubah, server **harus di-restart**.

---

## 12. Troubleshooting

### Server tidak bisa diakses dari browser

```bash
# Cek apakah server berjalan
ps aux | grep uvicorn

# Cek port
ss -tlnp | grep 8880

# Cek firewall (Ubuntu/Debian)
sudo ufw status
sudo ufw allow 8880  # bila perlu
```

### Error "UI not found" di browser

File `static/index.html` belum ada di server:
```bash
ls ~/app/cluster/static/
# Bila kosong, upload ulang:
scp static/index.html user@<ip-server>:~/app/cluster/static/
```

### Pipeline gagal di tahap fetch (ES connection error)

```bash
# Test koneksi ES manual
curl -u dev:nopassword -k https://dev.elastic.dashboard.nolimit.id/_cat/indices
```

### Model embedding tidak ter-download / stuck

```bash
source ~/pyvenvs/cluster-venv/bin/activate
python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('LazarusNLP/all-indo-e5-small-v4')
"
# Bila stuck, cek koneksi internet server:
curl -I https://huggingface.co
```

### Package tidak ditemukan saat pip install

```bash
# Pastikan venv aktif
source ~/pyvenvs/cluster-venv/bin/activate
which python   # harus menunjuk ke ~/pyvenvs/cluster-venv/bin/python

# Install ulang
pip install -r ~/app/cluster/requirements.txt
```

### Melihat log server (bukan log job)

```bash
# Bila pakai screen
screen -r clustering-server

# Bila pakai nohup
tail -f ~/app/cluster/server.log
```

### Membersihkan job lama

```bash
# Via UI: klik 🗑 pada job yang ingin dihapus
# Via CLI:
curl -X DELETE http://localhost:8880/jobs/<job_id>

# Hapus semua data output pipeline (HATI-HATI, tidak bisa dikembalikan)
rm -rf ~/app/cluster/data/<nama_project>/
```
