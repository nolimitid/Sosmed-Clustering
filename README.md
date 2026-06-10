# Klasterisasi Teks Media Sosial Berbahasa Indonesia

Mengelompokkan postingan/komentar berbahasa Indonesia yang pendek dan informal
("tidak baku") dari Twitter/X, Facebook, Instagram, dll. ke dalam rentang target
20–50 topik.

Pipeline: **bersihkan → normalkan slang → saring sampah → deduplikasi →
sentence embedding → UMAP → HDBSCAN → kata kunci c-TF-IDF (BERTopic) →
gabungkan ke rentang target → ekspor.**

## Mulai cepat

```bash
pip install -r requirements.txt

python cluster.py \
    --input posts.csv \
    --text-col text \
    --output-dir results \
    --target-min 20 --target-max 50
```

GPU sangat disarankan di atas ~50rb dokumen (Google Colab T4 sudah cukup).
Embedding di-cache di `results/cache/`, jadi menjalankan ulang dengan parameter
klasterisasi yang berbeda menjadi cepat.

### Flag yang berguna

| Flag | Fungsinya |
|---|---|
| `--model X` | Model embedding. Default: `LazarusNLP/all-indo-e5-small-v4` (dilatih untuk bahasa Indonesia, 384-dim — tradeoff tepat untuk jutaan baris). Alternatif lebih besar untuk dibandingkan: `paraphrase-multilingual-mpnet-base-v2` (768-dim, ~3-4x lebih lambat, lebih baik untuk code-mixing) |
| `--fit-sample N` | Latih UMAP/HDBSCAN pada maksimal N teks unik (default 300rb); sisanya ditetapkan ke sentroid topik terdekat. `0` = latih pada semua (**jangan** lakukan ini pada jutaan baris tanpa cuML) |
| `--sample 20000` | Run percontohan pada sampel acak baris input sebelum memproses seluruh korpus |
| `--assign-outliers` | Paksa setiap dokumen masuk ke topik terdekat (lihat "Outlier" di bawah) |
| `--min-cluster-size N` | Mengganti heuristik otomatis (`clip(fit_n/1000, 15, 500)`) |
| `--keep-emoji` | Pertahankan emoji pada teks yang diberikan ke embedder |
| `--no-slang-norm` | Nonaktifkan normalisasi slang kamus-alay |

## Keluaran (`results/`)

- `assignments.csv` — setiap baris asli beserta `clean_text`, `topic`, `topic_label`.
  `topic = -1` → outlier HDBSCAN; `topic = -2` → disaring sebagai non-informatif.
- `topics_summary.csv` — per topik: ukuran, 10 kata kunci c-TF-IDF teratas, 3 dokumen perwakilan.
- `metrics.json` — % outlier, silhouette, porsi klaster terbesar, parameter yang dipakai.
- `topic_map.html`, `topic_barchart.html` — inspeksi interaktif.
- `bertopic_model/` — model tersimpan; muat ulang dengan `BERTopic.load()` untuk
  melabeli data baru lewat `topic_model.transform(new_docs, new_embeddings)`.

> Catatan: nama field di `metrics.json` dan label `topic_label` khusus
> (`FILTERED_NON_INFORMATIVE`, `OUTLIER_NO_TOPIC`) sengaja dibiarkan dalam bahasa
> Inggris karena merupakan pengenal (identifier) yang dapat diproses program.
> Ubah di `cluster.py` bila Anda ingin keluarannya berbahasa Indonesia.

## Penskalaan ke jutaan baris (bagaimana pipeline ini menangani 6 juta)

Volume mengubah rekayasa, bukan statistiknya. Pilihan desainnya:

- **Embedding per-chunk yang dapat dilanjutkan.** Teks di-encode dalam checkpoint
  per 100rb yang disimpan ke `results/cache/`; jika Colab terputus di tengah run,
  menjalankan ulang akan melewati chunk yang sudah selesai. Embedding akhir berupa
  memmap float16, tidak pernah dimuat seluruhnya ke RAM (6 juta x 384 float16 ≈
  4,6 GB di disk; 768-dim akan dua kali lipat).
- **Latih pada sampel, tetapkan sisanya.** UMAP + HDBSCAN pada jutaan titik dengan
  Python murni butuh berjam-jam hingga kehabisan memori. *Struktur* topik yang
  diperkirakan dari sampel acak 300rb-1 juta pada korpus jutaan baris pada dasarnya
  sama dengan dari seluruh korpus — yang ditambahkan baris sisanya adalah volume
  per topik, bukan topik baru (tema yang terlalu langka untuk muncul pada sampel
  300rb juga terlalu langka untuk lolos `min_cluster_size`). Dokumen sisanya
  ditetapkan berdasarkan kemiripan kosinus ke sentroid topik di ruang embedding
  asli, secara per-chunk.
- **Aturan outlier yang konsisten.** Dokumen di luar sampel yang kemiripan
  terbaiknya di bawah persentil ke-5 dari kemiripan dokumen di dalam sampel
  terhadap sentroidnya sendiri diberi label `-1`, sehingga "outlier" bermakna
  sama di dalam maupun di luar sampel yang dilatih.
- **Jika Anda harus melatih pada semuanya**: BERTopic mendukung UMAP/HDBSCAN GPU
  lewat RAPIDS cuML (`from cuml.manifold import UMAP; from cuml.cluster import HDBSCAN`),
  yang sanggup menangani jutaan titik. Membutuhkan lingkungan CUDA dengan RAPIDS
  terpasang; masukkan model cuML tersebut ke `build_topic_model` sendiri.

Perkiraan kasar pada T4 untuk ~6 juta baris (verifikasi dulu dengan benchmark 10rb
milik Anda — angkanya bergantung pada panjang teks dan ukuran batch): deduplikasi
biasanya membuang sebagian besar korpus media sosial lebih dulu; embedding sisanya
dengan model 384-dim default berada di kisaran 1-3 jam; pelatihan sampel 300rb
hitungan menit; penetapan sentroid untuk sisanya hitungan menit.

Urutan yang disarankan:
```bash
# 1. percontohan ujung-ke-ujung pada 200rb baris; periksa topics_summary.csv
python cluster.py --input posts.parquet --text-col text --sample 200000 --output-dir pilot
# 2. run penuh (cache embedding membuat run ulang dengan parameter baru jadi murah)
python cluster.py --input posts.parquet --text-col text --output-dir full
# 3. cek stabilitas dengan seed berbeda (memakai ulang cache embedding)
python cluster.py --input posts.parquet --text-col text --output-dir full_s1 --seed 1
```
Gunakan parquet, bukan CSV, untuk input 6 juta baris — parsing CSV pandas pada
ukuran itu lambat dan boros memori.

## Alasan pilihan-pilihan ini

**Sentence embedding, bukan TF-IDF.** Postingan 5–15 token menghasilkan vektor
TF-IDF yang hampir seluruhnya nol; kemiripan kosinus antara dua postingan
informal pendek biasanya 0 meski membahas hal yang sama ("bensin naik lagi anjir"
vs "harga BBM makin gila"). Embedding transformer memetakan keduanya ke posisi
yang berdekatan.

**Pembersihan ringan saja.** Model embedding adalah komponen yang "memahami"
bahasa informal, dan ia dilatih pada teks alami. Stemming (Sastrawi) dan
penghapusan stopword agresif sebelum embedding merusak konteks dan menurunkan
kualitas — hal itu peninggalan era TF-IDF. Kita hanya membuang artefak platform
(URL, mention, penanda RT, '#') dan menormalkan pola yang benar-benar merusak
tokenizer: perpanjangan karakter (`mantaaap` → `mantap`), tawa
(`wkwkwkwk`/`awokawok` → satu token), dan ~4,3rb pemetaan slang→baku dari
Colloquial Indonesian Lexicon (Salsabila dkk., 2018). Stopword dipakai hanya
untuk (a) filter sampah dan (b) ekstraksi kata kunci c-TF-IDF — tidak pernah pada
teks yang di-embed.

**HDBSCAN dulu, baru gabungkan — bukan KMeans dengan K ditetapkan di awal.**
KMeans akan dengan senang hati memotong korpus menjadi tepat 35 bagian terlepas
dari ada-tidaknya 35 tema, dan pada data media sosial beberapa bagian itu akan
menjadi gabungan yang tidak koheren. HDBSCAN menemukan struktur klaster alami;
jika menemukan lebih dari `target-max`, BERTopik menggabungkan topik termirip
hingga ke target; jika menemukan kurang dari `target-min`, skrip mengulang dengan
`min_cluster_size` lebih kecil lalu *memperingatkan Anda* alih-alih mengada-ada
klaster. Jika peringatan itu terus muncul, itu temuan tentang data Anda, bukan bug.

**Deduplikasi sebelum klasterisasi.** Retweet, kampanye salin-tempel, dan spam
membuat string yang sama bisa muncul ribuan kali. Klasterisasi pada duplikat
membuang komputasi dan membiarkan satu postingan viral mendominasi geometri
klaster. Kita mengklaster teks unik yang sudah dinormalkan dan mempropagasikan
label kembali ke semua baris.

## Outlier — baca ini sebelum melaporkan hasil

HDBSCAN melabeli titik yang tidak masuk ke wilayah padat manapun sebagai `-1`.
Pada korpus media sosial ini lazimnya **20–50% dokumen**, dan itu *jujur*:
sebagian besar media sosial memang tidak membahas tema yang berulang. Anda punya
tiga opsi, urut dari yang paling jujur secara metodologis:

1. Laporkan outlier sebagai kategori "tanpa topik jelas" tersendiri (default).
2. `--assign-outliers` — setiap dokumen dapat topik, tetapi kemurnian per topik
   menurun. Cocok jika kegunaan hilir Anda menuntut cakupan penuh; sebutkan ini
   di laporan.
3. Turunkan `min_cluster_size`/`min_samples` — klaster lebih banyak dan lebih
   kecil serta outlier lebih sedikit, tetapi lebih banyak topik nyaris-duplikat
   untuk digabung.

## Protokol evaluasi (jangan dilewati)

Skor silhouette pada klaster embedding teks pendek adalah bukti yang lemah — ia
memberi nilai pada keterpisahan geometris, bukan koherensi semantik. Gunakan
hanya untuk membandingkan antar-run. Evaluasi yang benar-benar penting untuk
capstone:

1. **Uji kewajaran kata kunci**: untuk tiap topik di `topics_summary.csv`, bisakah
   Anda menamai topik itu dari kata kuncinya dalam <5 detik? Hitung berapa yang
   tidak bisa.
2. **Uji penyusup (intruder test)**: ambil 5 dokumen dari satu topik + 1 dari
   topik lain; jika manusia tidak bisa menebak penyusupnya, topik itu tidak
   koheren. Lakukan untuk ~20 topik.
3. **Stabilitas**: jalankan dengan `--seed 1`, `--seed 2`, `--seed 3`. Topik yang
   bertahan lintas seed itu nyata; topik yang muncul sekali adalah artefak.
   (UMAP dan HDBSCAN sama-sama stokastik — hasil satu run bukan temuan.)
4. **Cakupan**: berapa fraksi dokumen yang berada di topik yang bisa Anda namai?
   Laporkan angkanya.

## Keterbatasan / risiko yang diketahui

- **Kebocoran gaya platform**: mencampur Twitter + FB + IG dapat menghasilkan
  klaster yang tersusun berdasarkan *register platform*, bukan topik. Mitigasi:
  jalankan sekali per platform lebih dulu lalu bandingkan; atau tambahkan kolom
  `platform` dan periksa distribusi platform tiap topik di `assignments.csv`.
- **Kesalahan kamus slang**: penggantian level token bersifat buta-konteks
  (~4,3rb entri, pemetaan mayoritas yang menang). Ia memperbaiki jauh lebih
  banyak daripada yang dirusak pada domain ini, tetapi periksa `clean_text` lebih
  awal. Nonaktifkan dengan `--no-slang-norm` lalu bandingkan jika mencurigakan.
- **Code-mixing**: postingan campuran Indonesia-Inggris umum dijumpai. Model
  multibahasa default menanganinya; model murni-Indonesia mungkin tidak.
  Bandingkan.
- **Label topik adalah kata kunci, bukan nama.** c-TF-IDF memberi
  `0_bbm_harga_naik_subsidi`; untuk laporan, tulis label yang layak secara manual
  (atau beri prompt ke LLM dengan kata kunci + dokumen perwakilan — tetapi
  verifikasi dengan membaca dokumen sebenarnya).
- **Legalitas pengumpulan data**: scraping FB/IG melanggar ToS mereka; akses X
  API berbayar. Dokumentasikan sumber data dan ketentuannya di capstone —
  penguji menanyakannya.

## Matriks eksperimen yang disarankan untuk capstone

| Sumbu | Variasi |
|---|---|
| Embedding | multilingual-mpnet vs `LazarusNLP/all-indo-e5-small-v4` (default) vs `indolem/indobertweet-base-uncased` (mean pooling) |
| Normalisasi slang | aktif vs nonaktif (apakah kamus-alay benar-benar membantu embedding?) |
| Baseline | TF-IDF + KMeans (perkirakan ia kalah — kontras itu *adalah* sebuah hasil) |
| Outlier | dilaporkan vs ditetapkan |

Ini memberi Anda bab "alasan di balik pilihan model dan analisis" yang dapat
dipertahankan, alih-alih satu pipeline tanpa justifikasi.