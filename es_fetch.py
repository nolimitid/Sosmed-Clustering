"""
Fetch data dari Elasticsearch ke parquet chunks untuk pipeline clustering.

Menggunakan search_after (bukan scroll) agar dapat di-resume bila crash:
  - Tiap chunk disimpan sebagai chunk_NNNNN.parquet + chunk_NNNNN.done
  - Bila dilanjutkan, chunk yang sudah ada (.done) dilewati
  - Cursor disimpan di cursor.json — restart lanjut dari posisi terakhir

Usage:
  python es_fetch.py --project-id <uuid> --output-dir data/my_project
  python es_fetch.py --project-id <uuid> --output-dir data/my_project \\
      --indices v5_monthly_social_media_post_made_2026-05_indsight \\
      --chunk-size 100000
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth

warnings.filterwarnings("ignore", message="Unverified HTTPS")

ES_URL   = "https://dev.elastic.dashboard.nolimit.id"
ES_AUTH  = HTTPBasicAuth("dev", "nopassword")
ES_VERIFY = False

DEFAULT_INDICES = [
    "v5_monthly_social_media_post_made_2026-05_indsight",
    "v5_monthly_social_media_talk_2026-05_indsight",
    "v5_monthly_social_media_post_made_2026-06_indsight",
    "v5_monthly_social_media_talk_2026-06_indsight",
]

# Field yang diambil — cukup untuk clustering, tidak semua field
FETCH_FIELDS = [
    "content", "project_id", "object_id", "original_id",
    "specific_resource_type", "specific_type",
    "timestamp", "from_username", "from_id",
    "final_sentiment", "engagement", "reach",
    "link", "lang",
]


def es_get(path: str, **kwargs) -> dict:
    r = requests.get(f"{ES_URL}/{path}", auth=ES_AUTH, verify=ES_VERIFY, **kwargs)
    r.raise_for_status()
    return r.json()


def es_post(path: str, body: dict, **kwargs) -> dict:
    r = requests.post(f"{ES_URL}/{path}", auth=ES_AUTH, verify=ES_VERIFY,
                      json=body, **kwargs)
    r.raise_for_status()
    return r.json()


def count_docs(index: str, project_id: str) -> int:
    body = {"query": {"term": {"project_id": project_id}}}
    return es_post(f"{index}/_count", body)["count"]


def fetch_chunk(
    index: str,
    project_id: str,
    page_size: int,
    search_after: list | None,
) -> tuple[list[dict], list | None]:
    """
    Ambil satu batch dokumen menggunakan search_after.
    Returns (hits, next_search_after) — next_search_after=None bila sudah habis.
    """
    body: dict = {
        "size": page_size,
        "_source": FETCH_FIELDS,
        "query": {"term": {"project_id": project_id}},
        "sort": [
            {"timestamp": "asc"},
            {"original_id": "asc"},
        ],
    }
    if search_after:
        body["search_after"] = search_after

    resp = es_post(f"{index}/_search", body, timeout=120)
    hits = resp["hits"]["hits"]
    if not hits:
        return [], None
    last_sort = hits[-1]["sort"]
    return [h["_source"] for h in hits], last_sort


def load_cursor(cursor_path: Path) -> dict:
    if cursor_path.exists():
        return json.loads(cursor_path.read_text())
    return {"chunk_idx": 0, "search_after": None, "total_fetched": 0}


def save_cursor(cursor_path: Path, state: dict) -> None:
    cursor_path.write_text(json.dumps(state, indent=2))


def run_fetch(
    project_id: str,
    indices: list[str],
    output_dir: Path,
    chunk_size: int,
    page_size: int,
    max_docs: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cursor_path = output_dir / "cursor.json"
    state = load_cursor(cursor_path)

    chunk_idx    = state["chunk_idx"]
    search_after = state["search_after"]
    total_fetched = state["total_fetched"]

    # hitung total doc untuk progress bar
    print(f"[fetch] menghitung total dokumen project {project_id} ...")
    total_docs = 0
    for idx in indices:
        try:
            n = count_docs(idx, project_id)
            print(f"  {idx}: {n:,}")
            total_docs += n
        except Exception as e:
            print(f"  {idx}: error - {e}")
    print(f"[fetch] total: {total_docs:,} docs | chunk_size={chunk_size:,}")

    if max_docs > 0:
        total_docs = min(total_docs, max_docs)
        print(f"[fetch] batas max_docs={max_docs:,} — akan berhenti setelah {total_docs:,} dokumen")
    if total_fetched >= total_docs and total_docs > 0:
        print("[fetch] semua data sudah diambil sebelumnya.")
        return

    buffer: list[dict] = []
    t0 = time.time()

    for index in indices:
        print(f"\n[fetch] memproses index: {index}")
        while True:
            try:
                hits, next_after = fetch_chunk(index, project_id, page_size, search_after)
            except Exception as e:
                print(f"[fetch] error saat fetch, retry 10s: {e}")
                time.sleep(10)
                continue

            if not hits or (max_docs > 0 and total_fetched >= max_docs):
                search_after = None  # reset untuk index berikutnya
                break

            if max_docs > 0:
                remaining = max_docs - total_fetched
                hits = hits[:remaining]
            buffer.extend(hits)
            search_after = next_after
            total_fetched += len(hits)

            # simpan chunk bila buffer penuh
            while len(buffer) >= chunk_size:
                _flush_chunk(buffer[:chunk_size], chunk_idx, output_dir)
                buffer = buffer[chunk_size:]
                chunk_idx += 1
                save_cursor(cursor_path, {
                    "chunk_idx": chunk_idx,
                    "search_after": search_after,
                    "total_fetched": total_fetched,
                })

            elapsed = time.time() - t0
            rate = total_fetched / max(elapsed, 1)
            remain = max(total_docs - total_fetched, 0)
            eta = remain / max(rate, 1) / 60
            print(f"[fetch] {total_fetched:,}/{total_docs:,} docs "
                  f"| {rate:.0f} doc/s | ETA ~{eta:.1f} menit")

    # flush sisa buffer
    if buffer:
        _flush_chunk(buffer, chunk_idx, output_dir)
        chunk_idx += 1

    save_cursor(cursor_path, {
        "chunk_idx": chunk_idx,
        "search_after": None,
        "total_fetched": total_fetched,
    })
    print(f"\n[fetch] selesai — {total_fetched:,} docs dalam {chunk_idx} chunk "
          f"di {output_dir}/")


def _flush_chunk(rows: list[dict], idx: int, out_dir: Path) -> None:
    chunk_file = out_dir / f"chunk_{idx:05d}.parquet"
    done_file  = out_dir / f"chunk_{idx:05d}.done"
    if done_file.exists():
        print(f"[fetch] chunk {idx:05d} sudah ada, dilewati.")
        return
    df = pd.DataFrame(rows)
    df.to_parquet(chunk_file, index=False)
    done_file.write_text("ok")
    print(f"[fetch] chunk {idx:05d} disimpan ({len(rows):,} rows) -> {chunk_file.name}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-id", required=True, help="UUID project ES")
    p.add_argument("--output-dir", required=True, help="direktori output chunks parquet")
    p.add_argument("--indices", nargs="+", default=DEFAULT_INDICES,
                   help="nama index ES (default: semua 4 index)")
    p.add_argument("--chunk-size", type=int, default=100_000,
                   help="jumlah dokumen per file parquet (default: 100000)")
    p.add_argument("--page-size", type=int, default=5_000,
                   help="ukuran batch per request ES (default: 5000)")
    p.add_argument("--max-docs", type=int, default=0,
                   help="batas maksimal dokumen yang diambil dari ES (0 = semua; "
                        "berguna untuk testing, misal --max-docs 100000)")
    args = p.parse_args()

    run_fetch(
        project_id=args.project_id,
        indices=args.indices,
        output_dir=Path(args.output_dir),
        chunk_size=args.chunk_size,
        page_size=args.page_size,
        max_docs=args.max_docs,
    )


if __name__ == "__main__":
    main()
