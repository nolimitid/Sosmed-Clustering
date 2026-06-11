"""
Orchestrator end-to-end: ES fetch → cluster per chunk → merge global topics.

Checkpoint logic:
  - fetch.done       : skip ES fetch bila sudah selesai
  - cluster_chunk_NNNNN/cluster.done : skip clustering chunk bila sudah selesai
  - merged/merge.done: skip merge bila sudah selesai

Semua chunk yang sudah di-cluster dengan sukses tidak akan diproses ulang
walaupun pipeline dijalankan ulang setelah crash atau interrupt.

Usage:
  python pipeline.py --project-id <uuid> --output-dir data/my_project

  # Lanjutkan setelah crash (fetch sudah selesai, mulai dari cluster):
  python pipeline.py --project-id <uuid> --output-dir data/my_project --skip-fetch

  # Hanya jalankan merge dari chunk yang sudah di-cluster:
  python pipeline.py --project-id <uuid> --output-dir data/my_project \\
      --skip-fetch --skip-cluster
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def run_cmd(cmd: list[str], desc: str) -> int:
    """Jalankan perintah, stream output ke terminal, kembalikan exit code."""
    print(f"\n{'=' * 60}")
    print(f"[pipeline] {desc}")
    print(f"{'=' * 60}")
    print(f"  cmd: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n[pipeline] ERROR: '{desc}' gagal (exit {result.returncode})")
    return result.returncode


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    # ---- Identitas proyek ----
    p.add_argument("--project-id", required=True,
                   help="UUID project di Elasticsearch")
    p.add_argument("--output-dir", required=True,
                   help="direktori dasar output; subfolder chunks/, cluster_*, "
                        "dan merged/ akan dibuat di sini")

    # ---- ES fetch ----
    p.add_argument("--indices", nargs="+", default=None,
                   help="nama index ES (default: 4 index monthly dari es_fetch.py)")
    p.add_argument("--chunk-size", type=int, default=100_000,
                   help="jumlah dokumen per file parquet (default: 100000)")
    p.add_argument("--page-size", type=int, default=5_000,
                   help="ukuran batch per request ES (default: 5000)")
    p.add_argument("--max-docs", type=int, default=0,
                   help="batas maksimal total dokumen yang diambil dari ES "
                        "(0 = semua data project; berguna untuk testing, "
                        "misal --max-docs 100000)")

    # ---- Cluster per chunk ----
    p.add_argument("--model", default="LazarusNLP/all-indo-e5-small-v4")
    p.add_argument("--target-min", type=int, default=20)
    p.add_argument("--target-max", type=int, default=50)
    p.add_argument("--fit-sample", type=int, default=300_000,
                   help="maks teks unik untuk fit BERTopic per chunk (0 = semua)")
    p.add_argument("--min-cluster-size", type=int, default=0,
                   help="HDBSCAN min_cluster_size (0 = otomatis)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default=None, help="cuda / cpu")
    p.add_argument("--no-lang-filter", action="store_true")
    p.add_argument("--keep-langs", default="id,ms,jv,su,min,ban,ace")
    p.add_argument("--lang-backend", choices=["fasttext", "langdetect"],
                   default="fasttext")
    p.add_argument("--assign-outliers", action="store_true",
                   help="paksa semua teks ke topik terdekat (disarankan untuk pipeline)")
    p.add_argument("--graph-threshold", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)

    # ---- Merge ----
    p.add_argument("--merge-graph-threshold", type=float, default=0.30,
                   help="ambang centroid similarity untuk edge graph saat merge "
                        "(default: 0.30; sedikit lebih tinggi dari --graph-threshold "
                        "karena bekerja pada centroid-of-centroids)")

    # ---- Skip flags ----
    p.add_argument("--skip-fetch", action="store_true",
                   help="lewati tahap fetch ES (gunakan chunks yang sudah ada)")
    p.add_argument("--skip-cluster", action="store_true",
                   help="lewati tahap clustering per chunk")
    p.add_argument("--skip-merge", action="store_true",
                   help="lewati tahap merge global")

    args = p.parse_args()

    out_dir = Path(args.output_dir)
    chunks_dir = out_dir / "chunks"
    py = sys.executable   # pakai interpreter yang sama dengan pipeline.py

    # =========================================================================
    # TAHAP 1: Fetch dari Elasticsearch
    # =========================================================================
    if not args.skip_fetch:
        fetch_done = out_dir / "fetch.done"
        if fetch_done.exists():
            print(f"[pipeline] fetch sudah selesai sebelumnya (fetch.done ada), dilewati")
        else:
            cmd = [
                py, "es_fetch.py",
                "--project-id", args.project_id,
                "--output-dir", str(chunks_dir),
                "--chunk-size", str(args.chunk_size),
                "--page-size", str(args.page_size),
            ]
            if args.max_docs > 0:
                cmd += ["--max-docs", str(args.max_docs)]
            if args.indices:
                cmd += ["--indices"] + args.indices

            rc = run_cmd(cmd, f"ES fetch project {args.project_id}")
            if rc != 0:
                sys.exit(1)
            fetch_done.write_text("ok")
            print(f"[pipeline] fetch selesai -> fetch.done")
    else:
        print("[pipeline] --skip-fetch: melewati tahap fetch")

    # =========================================================================
    # TAHAP 2: Cluster per chunk
    # =========================================================================
    if not args.skip_cluster:
        chunk_files = sorted(chunks_dir.glob("chunk_?????.parquet"))
        if not chunk_files:
            print(f"[pipeline] PERINGATAN: tidak ada chunk parquet di {chunks_dir}. "
                  f"Jalankan tanpa --skip-fetch atau pastikan path sudah benar.")
            if not args.skip_merge:
                sys.exit(1)
        else:
            print(f"\n[pipeline] {len(chunk_files)} chunk ditemukan untuk di-cluster")
            failed: list[str] = []
            for chunk_path in chunk_files:
                chunk_stem = chunk_path.stem        # "chunk_00000"
                cluster_dir = out_dir / f"cluster_{chunk_stem}"
                done_file = cluster_dir / "cluster.done"

                if done_file.exists():
                    print(f"[pipeline] {chunk_stem}: sudah di-cluster, dilewati")
                    continue

                cmd = [
                    py, "cluster.py",
                    "--input", str(chunk_path),
                    "--text-col", "content",
                    "--output-dir", str(cluster_dir),
                    "--model", args.model,
                    "--target-min", str(args.target_min),
                    "--target-max", str(args.target_max),
                    "--fit-sample", str(args.fit_sample),
                    "--min-cluster-size", str(args.min_cluster_size),
                    "--batch-size", str(args.batch_size),
                    "--graph-threshold", str(args.graph_threshold),
                    "--seed", str(args.seed),
                    "--done-file", str(done_file),
                ]
                if args.device:
                    cmd += ["--device", args.device]
                if args.assign_outliers:
                    cmd.append("--assign-outliers")
                if args.no_lang_filter:
                    cmd.append("--no-lang-filter")
                else:
                    cmd += ["--keep-langs", args.keep_langs,
                            "--lang-backend", args.lang_backend]

                rc = run_cmd(cmd, f"cluster {chunk_stem}")
                if rc != 0:
                    print(f"[pipeline] PERINGATAN: {chunk_stem} gagal, lanjut ke chunk berikutnya")
                    failed.append(chunk_stem)

            done_count = sum(
                1 for f in chunk_files
                if (out_dir / f"cluster_{f.stem}" / "cluster.done").exists()
            )
            print(f"\n[pipeline] clustering selesai: "
                  f"{done_count}/{len(chunk_files)} chunk berhasil")
            if failed:
                print(f"[pipeline] chunk yang gagal ({len(failed)}): {failed}")
    else:
        print("[pipeline] --skip-cluster: melewati tahap cluster")

    # =========================================================================
    # TAHAP 3: Merge global topics
    # =========================================================================
    if not args.skip_merge:
        merge_dir = out_dir / "merged"
        merge_done = merge_dir / "merge.done"

        if merge_done.exists():
            print(f"[pipeline] merge sudah selesai sebelumnya (merge.done ada), dilewati")
        else:
            cmd = [
                py, "merge.py",
                "--base-dir", str(out_dir),
                "--output-dir", str(merge_dir),
                "--target-min", str(args.target_min),
                "--target-max", str(args.target_max),
                "--graph-threshold", str(args.merge_graph_threshold),
                "--seed", str(args.seed),
            ]
            rc = run_cmd(cmd, "merge global topics")
            if rc != 0:
                sys.exit(1)
    else:
        print("[pipeline] --skip-merge: melewati tahap merge")

    print(f"\n{'=' * 60}")
    print(f"[pipeline] SELESAI")
    print(f"  output dir   : {out_dir}/")
    print(f"  chunks        : {chunks_dir}/")
    print(f"  global topics : {out_dir / 'merged' / 'global_topics_summary.csv'}")
    print(f"  topic graph   : {out_dir / 'merged' / 'topic_graph.html'}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
