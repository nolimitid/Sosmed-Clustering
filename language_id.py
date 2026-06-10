"""
Deteksi bahasa untuk menyaring teks media sosial non-Indonesia.

Mengapa modul ini ada
---------------------
Korpus media sosial jarang seragam: selain bahasa Indonesia, ada postingan dan
komentar berbahasa Inggris, Hindi, Spanyol, Arab, dan lain-lain. Bahasa asing
itu mengotori klasterisasi -- klaster bisa terbentuk berdasarkan *bahasa* alih-
alih *topik*. Pipeline memakai modul ini untuk mendeteksi bahasa tiap dokumen
lalu *membuang* baris yang diyakini bukan berbahasa Indonesia (lihat cluster.py,
flag --keep-langs).

Backend (urut prioritas)
------------------------
1. fastText lid.176 (Joulin dkk., 2017) -- 176 bahasa, sangat cepat dan
   mendukung prediksi batch, jadi cocok untuk jutaan baris. Model `lid.176.ftz`
   (~917 KB) diunduh otomatis sekali ke ~/.cache bila belum ada.
   Pasang: `pip install fasttext-wheel` (roda prabangun, tanpa kompilator C).
2. langdetect (port Google language-detection) -- Python murni, mudah dipasang,
   tetapi jauh lebih lambat dan kurang andal pada teks pendek.
   Pasang: `pip install langdetect`.
3. Tidak ada backend -> semua baris ditandai 'und' (undetermined). Karena baris
   berkeyakinan rendah TIDAK dibuang (lihat cluster.py), tanpa backend tidak ada
   baris yang tersaring -- pipeline tetap berjalan, disertai peringatan.

Keterbatasan (baca sebelum mempercayai keluarannya)
---------------------------------------------------
Deteksi bahasa pada teks pendek + tidak baku + campur kode (code-mixing) memang
rapuh. Bahasa Indonesia kerap tertukar dengan Melayu ('ms') dan bahasa daerah
('jv', 'su', 'min') karena sangat mirip -- itulah sebabnya keep-set bawaan
menyertakan mereka, agar teks Indonesia asli tidak ikut terbuang. Skor
kepercayaan (confidence) ikut dikembalikan supaya hanya baris yang *diyakini*
berbahasa asing yang dibuang.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

# Model resmi fastText untuk identifikasi bahasa (versi terkuantisasi, kecil).
LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
DEFAULT_MODEL_PATH = Path.home() / ".cache" / "fasttext" / "lid.176.ftz"

# Kode ISO 639 -> nama bahasa, untuk laporan yang mudah dibaca. Kode di luar peta
# ini tetap ditampilkan apa adanya (mis. 'ceb'), jadi tidak ada yang tersembunyi.
LANG_NAMES = {
    "id": "Indonesian", "ms": "Malay", "jv": "Javanese", "su": "Sundanese",
    "min": "Minangkabau", "ban": "Balinese", "ace": "Acehnese",
    "en": "English", "hi": "Hindi", "es": "Spanish", "ar": "Arabic",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "th": "Thai",
    "vi": "Vietnamese", "tl": "Tagalog", "ceb": "Cebuano",
    "fr": "French", "de": "German", "pt": "Portuguese", "ru": "Russian",
    "nl": "Dutch", "it": "Italian", "tr": "Turkish", "fa": "Persian",
    "ur": "Urdu", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
    "mr": "Marathi", "pa": "Punjabi", "gu": "Gujarati", "kn": "Kannada",
    "ml": "Malayalam", "pl": "Polish", "uk": "Ukrainian", "ro": "Romanian",
    "el": "Greek", "he": "Hebrew", "sw": "Swahili",
    "und": "Undetermined",
}


def lang_name(code: str) -> str:
    """Nama yang mudah dibaca untuk kode bahasa; jatuh ke kode itu sendiri."""
    base = (code or "und").split("-")[0].lower()
    return LANG_NAMES.get(base, code)


# ---------------------------------------------------------------------------
# Pemuatan backend
# ---------------------------------------------------------------------------

def _load_fasttext(model_path: str | Path):
    import fasttext  # mengangkat ImportError bila tidak terpasang

    # Bungkam peringatan deprekasi fastText yang berisik di stderr.
    try:
        fasttext.FastText.eprint = lambda *a, **k: None
    except Exception:
        pass

    model_path = Path(model_path)
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[lang] mengunduh model fastText lid.176 -> {model_path} ...")
        tmp = model_path.with_suffix(".tmp")
        urlretrieve(LID_URL, tmp)
        tmp.rename(model_path)
    return fasttext.load_model(str(model_path))


def _load_langdetect():
    import langdetect  # mengangkat ImportError bila tidak terpasang
    from langdetect import DetectorFactory

    DetectorFactory.seed = 0  # buat hasil langdetect deterministik
    return langdetect


# ---------------------------------------------------------------------------
# Detektor
# ---------------------------------------------------------------------------

class LanguageDetector:
    """
    Detektor bahasa dengan backend yang dapat berganti otomatis. Pakai
    `LanguageDetector.load(...)` untuk memilih backend terbaik yang tersedia,
    lalu `detect_batch(texts)` untuk mendeteksi banyak teks sekaligus.
    """

    def __init__(self, backend: str, model=None):
        self.backend = backend
        self._model = model

    @property
    def name(self) -> str:
        return self.backend

    @property
    def available(self) -> bool:
        return self.backend != "none"

    @classmethod
    def load(cls, model_path: str | Path | None = None,
             prefer: str = "fasttext") -> "LanguageDetector":
        order = ["fasttext", "langdetect"]
        if prefer == "langdetect":
            order = ["langdetect", "fasttext"]

        for backend in order:
            try:
                if backend == "fasttext":
                    return cls("fasttext",
                               _load_fasttext(model_path or DEFAULT_MODEL_PATH))
                if backend == "langdetect":
                    return cls("langdetect", _load_langdetect())
            except Exception as exc:
                print(f"[lang] backend '{backend}' tidak tersedia: {exc}")

        print("[lang] PERINGATAN: tidak ada backend deteksi bahasa terpasang; "
              "tidak ada baris yang disaring berdasarkan bahasa.\n"
              "       Pasang salah satu agar penyaringan aktif:\n"
              "         pip install fasttext-wheel   # cepat, disarankan untuk jutaan baris\n"
              "         pip install langdetect       # Python murni, lebih lambat")
        return cls("none", None)

    def detect_batch(self, texts: list[str],
                     batch_size: int = 50_000) -> tuple[list[str], list[float]]:
        """
        Mengembalikan (kode_bahasa, kepercayaan) untuk tiap teks. Teks kosong
        atau backend tak tersedia -> ('und', 0.0).
        """
        n = len(texts)
        codes = ["und"] * n
        confs = [0.0] * n
        if not self.available or self._model is None:
            return codes, confs
        if self.backend == "fasttext":
            self._detect_fasttext(texts, codes, confs, batch_size)
        else:
            self._detect_langdetect(texts, codes, confs)
        return codes, confs

    # -- backend khusus ------------------------------------------------------

    def _detect_fasttext(self, texts, codes, confs, batch_size):
        model = self._model
        n = len(texts)
        for lo in range(0, n, batch_size):
            chunk = texts[lo:lo + batch_size]
            # fastText.predict tidak boleh memuat newline; teks kosong dilewati
            # (tetap 'und'). Kumpulkan hanya yang non-kosong untuk satu panggilan.
            prepared, idx = [], []
            for j, t in enumerate(chunk):
                s = t.replace("\n", " ").strip() if isinstance(t, str) else ""
                if s:
                    prepared.append(s)
                    idx.append(j)
            if prepared:
                labels, probs = model.predict(prepared, k=1)
                for k, j in enumerate(idx):
                    lab = labels[k][0] if labels[k] else "__label__und"
                    codes[lo + j] = lab.replace("__label__", "")
                    confs[lo + j] = float(probs[k][0]) if len(probs[k]) else 0.0
            print(f"[lang] fastText {min(lo + batch_size, n)}/{n}")

    def _detect_langdetect(self, texts, codes, confs):
        from langdetect import detect_langs
        from langdetect.lang_detect_exception import LangDetectException

        n = len(texts)
        for i, t in enumerate(texts):
            s = t.strip() if isinstance(t, str) else ""
            if not s:
                continue
            try:
                res = detect_langs(s)
                if res:
                    codes[i] = res[0].lang
                    confs[i] = float(res[0].prob)
            except LangDetectException:
                pass  # teks tanpa fitur bahasa -> biarkan 'und'
            if (i + 1) % 50_000 == 0:
                print(f"[lang] langdetect {i + 1}/{n}")
