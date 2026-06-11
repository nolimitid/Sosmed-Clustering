"""
Server FastAPI untuk menjalankan pipeline klasterisasi (cluster.py) dari jarak
jauh, mis. lewat Termius/SSH di ponsel.

Setiap permintaan klasterisasi dijalankan sebagai job latar belakang (subprocess
`cluster.py`), jadi koneksi SSH/HTTP tidak perlu tetap terbuka selama proses yang
bisa memakan waktu berjam-jam. Memori model ML juga dibebaskan saat tiap job
selesai karena berjalan di proses terpisah.

Menjalankan:
    pip install -r requirements.txt
    uvicorn api:app --host 0.0.0.0 --port 8000     # atau:  python3 api.py

Buka dokumentasi interaktif di  http://<host>:8000/docs

Alur tipikal (contoh curl dari Termius):
    # 1. mulai job (unggah berkas)
    curl -F text_col=text -F file=@posts.csv http://localhost:8000/jobs
    # -> {"id": "ab12cd34ef56", "status": "queued", ...}

    # atau pakai berkas yang sudah ada di server (cocok untuk parquet jutaan baris)
    curl -F text_col=text -F input_path=/data/posts.parquet -F sample=200000 \
         http://localhost:8000/jobs

    # 2. pantau status & metrik
    curl http://localhost:8000/jobs/ab12cd34ef56
    # 3. lihat log proses (200 baris terakhir)
    curl "http://localhost:8000/jobs/ab12cd34ef56/log?tail=200"
    # 4. daftar & unduh keluaran
    curl http://localhost:8000/jobs/ab12cd34ef56/files
    curl -OJ http://localhost:8000/jobs/ab12cd34ef56/files/assignments.csv

Keamanan: server ini tidak punya autentikasi bawaan. Bila mengikat ke 0.0.0.0,
batasi lewat firewall / SSH tunnel, atau set variabel lingkungan API_TOKEN untuk
mewajibkan header  `X-API-Token: <token>`  (atau `Authorization: Bearer <token>`)
pada semua endpoint job. input_path bisa membaca berkas mana pun yang dapat
diakses proses server, jadi jangan paparkan tanpa proteksi.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

HERE = Path(__file__).resolve().parent
CLUSTER_SCRIPT  = HERE / "cluster.py"
PIPELINE_SCRIPT = HERE / "pipeline.py"
STATIC_DIR      = HERE / "static"
STATIC_DIR.mkdir(exist_ok=True)
JOBS_ROOT = Path(os.environ.get("API_JOBS_DIR", HERE / "api_jobs"))
JOBS_ROOT.mkdir(parents=True, exist_ok=True)

# Job klasterisasi berat memakai banyak RAM/GPU; secara bawaan jalankan satu per
# satu. Naikkan lewat API_MAX_WORKERS bila perangkat keras Anda memadai.
MAX_WORKERS = int(os.environ.get("API_MAX_WORKERS", "1"))
API_TOKEN = os.environ.get("API_TOKEN")

ALLOWED_INPUT_SUFFIXES = {".csv", ".tsv", ".parquet", ".jsonl", ".json"}

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="cluster-job")
_lock = threading.Lock()
JOBS: dict[str, "Job"] = {}


# ---------------------------------------------------------------------------
# Parameter pipeline -> pemetaan ke flag CLI cluster.py
# ---------------------------------------------------------------------------
# (nama_field, flag_cli, kind)  kind: "value" | "flag" (store_true) | "optional"
PARAM_SPEC = [
    ("model", "--model", "value"),
    ("e5_prefix", "--e5-prefix", "flag"),
    ("fit_sample", "--fit-sample", "value"),
    ("target_min", "--target-min", "value"),
    ("target_max", "--target-max", "value"),
    ("min_cluster_size", "--min-cluster-size", "value"),
    ("min_content_tokens", "--min-content-tokens", "value"),
    ("assign_outliers", "--assign-outliers", "flag"),
    ("keep_emoji", "--keep-emoji", "flag"),
    ("no_slang_norm", "--no-slang-norm", "flag"),
    ("no_lang_filter", "--no-lang-filter", "flag"),
    ("keep_langs", "--keep-langs", "value"),
    ("lang_backend", "--lang-backend", "value"),
    ("lang_model", "--lang-model", "optional"),
    ("lang_min_conf", "--lang-min-conf", "value"),
    ("sample", "--sample", "value"),
    ("graph_threshold", "--graph-threshold", "value"),
    ("batch_size", "--batch-size", "value"),
    ("device", "--device", "optional"),
    ("seed", "--seed", "value"),
]


def build_pipeline_command(config: dict, output_dir: Path) -> list[str]:
    """Susun perintah `python pipeline.py ...` dari konfigurasi job pipeline."""
    cmd = [
        sys.executable, "-u", str(PIPELINE_SCRIPT),
        "--project-id", config["project_id"],
        "--output-dir", str(output_dir),
        "--chunk-size", str(config.get("chunk_size", 100_000)),
        "--target-min", str(config.get("target_min", 20)),
        "--target-max", str(config.get("target_max", 50)),
        "--model",      config.get("model", "LazarusNLP/all-indo-e5-small-v4"),
        "--batch-size", str(config.get("batch_size", 128)),
        "--keep-langs", config.get("keep_langs", "id,ms,jv,su,min,ban,ace"),
        "--graph-threshold", str(config.get("graph_threshold", 0.25)),
        "--seed",        str(config.get("seed", 42)),
    ]
    if config.get("max_docs", 0) > 0:
        cmd += ["--max-docs", str(config["max_docs"])]
    if config.get("device"):
        cmd += ["--device", config["device"]]
    if config.get("no_lang_filter"):
        cmd.append("--no-lang-filter")
    if config.get("assign_outliers"):
        cmd.append("--assign-outliers")
    return cmd


def build_command(config: dict, input_path: Path, output_dir: Path) -> list[str]:
    """Susun perintah `python cluster.py ...` dari konfigurasi job."""
    cmd = [
        sys.executable, "-u", str(CLUSTER_SCRIPT),
        "--input", str(input_path),
        "--text-col", config["text_col"],
        "--output-dir", str(output_dir),
    ]
    for field, flag, kind in PARAM_SPEC:
        val = config.get(field)
        if kind == "flag":
            if val:
                cmd.append(flag)
        elif kind == "optional":
            if val:
                cmd += [flag, str(val)]
        else:  # value
            cmd += [flag, str(val)]
    return cmd


# ---------------------------------------------------------------------------
# Model Job
# ---------------------------------------------------------------------------

class Job:
    def __init__(self, job_id: str, config: dict, input_path: Path,
                 output_dir: Path, log_path: Path):
        self.id = job_id
        self.config = config
        self.input_path = str(input_path)
        self.output_dir = str(output_dir)
        self.log_path = str(log_path)
        self.status = "queued"          # queued | running | done | failed | cancelled | interrupted
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.returncode: int | None = None
        self.pid: int | None = None
        self.error: str | None = None
        self.metrics: dict | None = None
        self._proc: subprocess.Popen | None = None

    def snapshot(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_snapshot(cls, data: dict) -> "Job":
        job = cls.__new__(cls)
        job.__dict__.update(data)
        job._proc = None
        return job


def _save_status(job: Job) -> None:
    try:
        Path(job.output_dir).parent.mkdir(parents=True, exist_ok=True)
        status_path = Path(job.output_dir).parent / "status.json"
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(job.snapshot(), f, indent=2, ensure_ascii=False)
    except Exception as exc:  # persistensi status hanya pelengkap; jangan menggagalkan job
        print(f"[api] gagal menulis status.json untuk {job.id}: {exc}", file=sys.stderr)


def _read_metrics(output_dir: Path) -> dict | None:
    path = output_dir / "metrics.json"
    if path.is_file():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _run_job(job_id: str) -> None:
    job = JOBS[job_id]
    if job.status == "cancelled":      # dibatalkan selagi masih antre
        return
    job.status = "running"
    job.started_at = time.time()
    _save_status(job)

    if "cmd_args" in job.config:
        cmd = job.config["cmd_args"]
    else:
        cmd = build_command(job.config, Path(job.input_path), Path(job.output_dir))
    script_name = Path(cmd[2]).name if len(cmd) > 2 else "script"
    try:
        with open(job.log_path, "a", buffering=1, encoding="utf-8") as logf:
            logf.write("$ " + " ".join(cmd) + "\n\n")
            proc = subprocess.Popen(
                cmd, cwd=str(HERE),
                stdout=logf, stderr=subprocess.STDOUT, text=True,
            )
            job._proc = proc
            job.pid = proc.pid
            returncode = proc.wait()
        job.returncode = returncode
        if job.status == "cancelled":
            pass
        elif returncode == 0:
            job.status = "done"
            job.metrics = _read_metrics(Path(job.output_dir))
        else:
            job.status = "failed"
            job.error = (f"{script_name} keluar dengan kode {returncode} — "
                         f"lihat log untuk detail")
    except Exception as exc:
        job.status = "failed"
        job.error = repr(exc)
    finally:
        job.finished_at = time.time()
        job._proc = None
        _save_status(job)


def _load_existing_jobs() -> None:
    """Pulihkan metadata job dari run sebelumnya saat server dimulai."""
    for status_path in sorted(JOBS_ROOT.glob("*/status.json")):
        try:
            with open(status_path, encoding="utf-8") as f:
                data = json.load(f)
            job = Job.from_snapshot(data)
            # proses dari sesi server sebelumnya sudah mati
            if job.status in ("queued", "running"):
                job.status = "interrupted"
            JOBS[job.id] = job
        except Exception as exc:
            print(f"[api] lewati {status_path}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Autentikasi token opsional
# ---------------------------------------------------------------------------

def require_token(
    x_api_token: str | None = Header(None),
    authorization: str | None = Header(None),
) -> None:
    if not API_TOKEN:
        return
    supplied = x_api_token
    if supplied is None and authorization and authorization.lower().startswith("bearer "):
        supplied = authorization.split(" ", 1)[1]
    if supplied != API_TOKEN:
        raise HTTPException(status_code=401, detail="token tidak valid")


AUTH = [Depends(require_token)]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_existing_jobs()          # pulihkan metadata job dari sesi sebelumnya
    yield
    executor.shutdown(wait=False)


app = FastAPI(
    title="Sosmed Clustering API",
    description="Jalankan pipeline klasterisasi cluster.py sebagai job latar belakang.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict:
    return {
        "service": "sosmed-clustering",
        "docs": "/docs",
        "endpoints": {
            "create_job": "POST /jobs",
            "list_jobs": "GET /jobs",
            "job_status": "GET /jobs/{job_id}",
            "job_log": "GET /jobs/{job_id}/log?tail=N",
            "list_files": "GET /jobs/{job_id}/files",
            "download_file": "GET /jobs/{job_id}/files/{name}",
            "cancel_job": "POST /jobs/{job_id}/cancel",
            "delete_job": "DELETE /jobs/{job_id}",
        },
        "auth_required": bool(API_TOKEN),
    }


@app.get("/health")
def health() -> dict:
    running = sum(1 for j in JOBS.values() if j.status == "running")
    queued = sum(1 for j in JOBS.values() if j.status == "queued")
    return {"status": "ok", "jobs_total": len(JOBS),
            "running": running, "queued": queued, "max_workers": MAX_WORKERS}


@app.post("/jobs", dependencies=AUTH)
async def create_job(
    text_col: str = Form(..., description="nama kolom yang memuat teks"),
    file: UploadFile | None = File(None, description="berkas csv/tsv/parquet/jsonl/json"),
    input_path: str | None = Form(None, description="path berkas yang sudah ada di server"),
    output_dir: str | None = Form(None, description="abaikan untuk memakai folder job otomatis"),
    model: str = Form("LazarusNLP/all-indo-e5-small-v4"),
    e5_prefix: bool = Form(False),
    fit_sample: int = Form(300_000),
    target_min: int = Form(20),
    target_max: int = Form(50),
    min_cluster_size: int = Form(0),
    min_content_tokens: int = Form(2),
    assign_outliers: bool = Form(False),
    keep_emoji: bool = Form(False),
    no_slang_norm: bool = Form(False),
    no_lang_filter: bool = Form(False),
    keep_langs: str = Form("id,ms,jv,su,min,ban,ace"),
    lang_backend: str = Form("fasttext"),
    lang_model: str | None = Form(None),
    lang_min_conf: float = Form(0.5),
    sample: int = Form(0),
    graph_threshold: float = Form(0.25),
    batch_size: int = Form(128),
    device: str | None = Form(None),
    seed: int = Form(42),
) -> dict:
    if (file is None) == (input_path is None):
        raise HTTPException(
            status_code=400,
            detail="sertakan tepat satu dari 'file' (unggahan) atau 'input_path' (di server)",
        )

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_ROOT / job_id
    input_dir = job_dir / "input"
    out_dir = Path(output_dir) if output_dir else (job_dir / "output")
    input_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if file is not None:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_INPUT_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"format '{suffix}' tidak didukung; gunakan salah satu dari "
                       f"{sorted(ALLOWED_INPUT_SUFFIXES)}",
            )
        dest = input_dir / f"data{suffix}"
        with open(dest, "wb") as out:
            shutil.copyfileobj(file.file, out)
        resolved_input = dest
    else:
        resolved_input = Path(input_path).expanduser()
        if not resolved_input.is_file():
            raise HTTPException(status_code=400,
                                detail=f"input_path tidak ditemukan di server: {resolved_input}")
        if resolved_input.suffix.lower() not in ALLOWED_INPUT_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"format '{resolved_input.suffix}' tidak didukung",
            )

    config = {
        "text_col": text_col, "model": model, "e5_prefix": e5_prefix,
        "fit_sample": fit_sample, "target_min": target_min, "target_max": target_max,
        "min_cluster_size": min_cluster_size, "min_content_tokens": min_content_tokens,
        "assign_outliers": assign_outliers, "keep_emoji": keep_emoji,
        "no_slang_norm": no_slang_norm, "no_lang_filter": no_lang_filter,
        "keep_langs": keep_langs, "lang_backend": lang_backend, "lang_model": lang_model,
        "lang_min_conf": lang_min_conf, "sample": sample,
        "graph_threshold": graph_threshold, "batch_size": batch_size,
        "device": device, "seed": seed,
    }

    log_path = job_dir / "job.log"
    log_path.touch()
    job = Job(job_id, config, resolved_input, out_dir, log_path)
    with _lock:
        JOBS[job_id] = job
    _save_status(job)
    executor.submit(_run_job, job_id)
    return job.snapshot()


@app.get("/jobs", dependencies=AUTH)
def list_jobs() -> dict:
    jobs = sorted(JOBS.values(), key=lambda j: j.created_at, reverse=True)
    return {"jobs": [j.snapshot() for j in jobs]}


def _get_job(job_id: str) -> Job:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job tidak ditemukan: {job_id}")
    return job


def _download_name(job: Job, rel_path: Path) -> str:
    """Nama berkas unduhan: `<nama-asli>_<project_id>.<ext>`.

    Suffiks dengan project_id (untuk job pipeline) atau id job sehingga berkas yang
    diunduh unik antar project di folder Unduhan. Subdirektori (mis.
    `cluster_chunk_00001/`) diabaikan agar nama tetap ringkas — contoh:
    `topics_summary_<project_id>.csv`.
    """
    pid = job.config.get("project_id") or job.id
    pid = re.sub(r"[^A-Za-z0-9._-]+", "_", str(pid)).strip("_") or job.id
    return f"{rel_path.stem}_{pid}{rel_path.suffix}"


@app.get("/jobs/{job_id}", dependencies=AUTH)
def get_job(job_id: str) -> dict:
    return _get_job(job_id).snapshot()


@app.get("/jobs/{job_id}/log", response_class=PlainTextResponse, dependencies=AUTH)
def get_log(job_id: str, tail: int = 0) -> str:
    job = _get_job(job_id)
    path = Path(job.log_path)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if tail and tail > 0:
        text = "\n".join(text.splitlines()[-tail:])
    return text


@app.get("/jobs/{job_id}/files", dependencies=AUTH)
def list_files(job_id: str) -> dict:
    job = _get_job(job_id)
    base = Path(job.output_dir)
    files = []
    if base.exists():
        for p in sorted(base.rglob("*")):
            if p.is_dir():
                continue
            rel = p.relative_to(base)
            if "cache" in rel.parts:        # lewati cache embedding (bisa beberapa GB)
                continue
            files.append({
                "name": str(rel),
                "bytes": p.stat().st_size,
                "download_name": _download_name(job, rel),
            })
    return {"job_id": job_id, "output_dir": job.output_dir, "files": files}


@app.get("/jobs/{job_id}/files/{filename:path}", dependencies=AUTH)
def download_file(job_id: str, filename: str) -> FileResponse:
    job = _get_job(job_id)
    base = Path(job.output_dir).resolve()
    target = (base / filename).resolve()
    if base != target and base not in target.parents:   # cegah path traversal
        raise HTTPException(status_code=400, detail="path berkas tidak valid")
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"berkas tidak ditemukan: {filename}")
    return FileResponse(target, filename=_download_name(job, target.relative_to(base)))


@app.post("/jobs/{job_id}/cancel", dependencies=AUTH)
def cancel_job(job_id: str) -> dict:
    job = _get_job(job_id)
    if job.status not in ("queued", "running"):
        raise HTTPException(status_code=409,
                            detail=f"job sudah '{job.status}', tidak bisa dibatalkan")
    proc = job._proc
    job.status = "cancelled"
    if proc is not None and proc.poll() is None:
        proc.terminate()
    _save_status(job)
    return job.snapshot()


@app.delete("/jobs/{job_id}", dependencies=AUTH)
def delete_job(job_id: str) -> dict:
    job = _get_job(job_id)
    if job.status in ("queued", "running"):
        raise HTTPException(status_code=409,
                            detail="batalkan job dulu sebelum menghapus")
    with _lock:
        JOBS.pop(job_id, None)
    job_dir = JOBS_ROOT / job_id
    if job_dir.exists() and JOBS_ROOT in job_dir.resolve().parents:
        shutil.rmtree(job_dir, ignore_errors=True)
    return {"deleted": job_id}


# ---------------------------------------------------------------------------
# Pipeline jobs (es_fetch → cluster per chunk → merge)
# ---------------------------------------------------------------------------

@app.post("/pipeline-jobs", dependencies=AUTH)
async def create_pipeline_job(
    project_id: str = Form(..., description="UUID project di Elasticsearch"),
    output_dir: str | None = Form(None, description="path output di server; kosong = otomatis"),
    max_docs: int = Form(0, description="batas total dokumen dari ES (0 = semua)"),
    chunk_size: int = Form(100_000, description="dokumen per file parquet saat fetch"),
    target_min: int = Form(20),
    target_max: int = Form(50),
    model: str = Form("LazarusNLP/all-indo-e5-small-v4"),
    no_lang_filter: bool = Form(False),
    keep_langs: str = Form("id,ms,jv,su,min,ban,ace"),
    assign_outliers: bool = Form(True),
    graph_threshold: float = Form(0.25),
    batch_size: int = Form(128),
    device: str | None = Form(None),
    seed: int = Form(42),
) -> dict:
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_ROOT / job_id
    out_dir = Path(output_dir) if output_dir else (job_dir / "output")
    out_dir.mkdir(parents=True, exist_ok=True)
    job_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "job_type": "pipeline",
        "project_id": project_id,
        "max_docs": max_docs,
        "chunk_size": chunk_size,
        "target_min": target_min,
        "target_max": target_max,
        "model": model,
        "no_lang_filter": no_lang_filter,
        "keep_langs": keep_langs,
        "assign_outliers": assign_outliers,
        "graph_threshold": graph_threshold,
        "batch_size": batch_size,
        "device": device,
        "seed": seed,
    }
    config["cmd_args"] = build_pipeline_command(config, out_dir)

    log_path = job_dir / "job.log"
    log_path.touch()
    job = Job(job_id, config, out_dir, out_dir, log_path)
    with _lock:
        JOBS[job_id] = job
    _save_status(job)
    executor.submit(_run_job, job_id)
    return job.snapshot()


# ---------------------------------------------------------------------------
# UI web
# ---------------------------------------------------------------------------

@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def ui() -> HTMLResponse:
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h2 style='font-family:sans-serif;padding:40px'>UI belum diunggah.<br>"
        "Salin <code>static/index.html</code> ke folder aplikasi di server.</h2>",
        status_code=200,
    )


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
