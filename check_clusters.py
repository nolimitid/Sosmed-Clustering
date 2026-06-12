"""
Periksa hasil clustering: hitung jumlah cluster di setiap CSV keluaran pipeline
dan cetak 10 cluster terbesar.

Mengenali tiga bentuk CSV yang dihasilkan pipeline:
  - merged/global_topics_summary.csv   (kolom: global_topic, count, keywords)
  - cluster_chunk_*/topics_summary.csv (kolom: Topic, Count, Keywords)
  - cluster_chunk_*/assignments.csv    (kolom: topic — satu baris per dokumen)

Cluster outlier (-1) dan baris tersaring non-informatif (-2) TIDAK dihitung
sebagai cluster, tetapi jumlahnya tetap dilaporkan.

Penggunaan:
  python check_clusters.py --output-dir data/project1
  python check_clusters.py --output-dir data/project1 --top 10 --per-file
  python check_clusters.py --csv data/project1/merged/global_topics_summary.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Urutan prioritas saat mencari kolom id cluster pada sebuah CSV.
CLUSTER_COLS = ["global_topic", "Topic", "topic"]
COUNT_COLS = ["count", "Count"]
KEYWORD_COLS = ["keywords", "Keywords", "Name"]
OUTLIER_IDS = {-1, -2}                       # -1 = outlier HDBSCAN, -2 = non-informatif
SKIP_NAMES = {"dropped_languages.csv", "topic_graph_edges.csv"}


def _first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def analyze_csv(path: Path) -> dict | None:
    """Kembalikan ringkasan cluster dari satu CSV, atau None bila bukan CSV cluster."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[check] gagal membaca {path}: {exc}", file=sys.stderr)
        return None

    ccol = _first_col(df, CLUSTER_COLS)
    if ccol is None:
        return None  # bukan CSV berisi cluster

    ids = pd.to_numeric(df[ccol], errors="coerce")

    count_col = _first_col(df, COUNT_COLS)
    if count_col is not None and df[ccol].is_unique:
        # CSV ringkasan: satu baris per cluster, ukuran ada di kolom count
        sizes = pd.Series(
            pd.to_numeric(df[count_col], errors="coerce").fillna(0).values,
            index=ids.values,
        ).groupby(level=0).sum()
    else:
        # CSV assignments: satu baris per dokumen, ukuran = frekuensi id cluster
        sizes = ids.value_counts()

    sizes = sizes[sizes.index.notna()]
    sizes.index = sizes.index.astype(int)

    n_outlier = int(sizes.get(-1, 0))
    n_filtered = int(sizes.get(-2, 0))
    valid = sizes[~sizes.index.isin(OUTLIER_IDS)].sort_values(ascending=False)

    # Peta id cluster -> kata kunci untuk ditampilkan (bila tersedia)
    kw_col = _first_col(df, KEYWORD_COLS)
    keywords: dict[int, str] = {}
    if kw_col is not None and df[ccol].is_unique:
        for cid, kw in zip(ids, df[kw_col]):
            if pd.notna(cid):
                keywords[int(cid)] = str(kw) if pd.notna(kw) else ""

    return {
        "path": path,
        "cluster_col": ccol,
        "n_clusters": int(len(valid)),
        "n_docs_clustered": int(valid.sum()),
        "n_outlier": n_outlier,
        "n_filtered": n_filtered,
        "sizes": valid,
        "keywords": keywords,
    }


def print_top(report: dict, top: int) -> None:
    sizes = report["sizes"].head(top)
    kw = report["keywords"]
    rel = report["path"]
    total = report["n_clusters"]
    note = "" if total >= top else f" (hanya {total} cluster)"
    print(f"\n=== {top} cluster terbesar{note} — {rel} ===")
    if sizes.empty:
        print("  (tidak ada cluster valid)")
        return
    has_kw = any(kw.get(int(cid)) for cid in sizes.index)
    header = f"  {'rank':>4}  {'cluster':>8}  {'count':>12}"
    if has_kw:
        header += "  keywords"
    print(header)
    for rank, (cid, cnt) in enumerate(sizes.items(), start=1):
        line = f"  {rank:>4}  {int(cid):>8}  {int(cnt):>12,}"
        if has_kw:
            line += f"  {kw.get(int(cid), '')[:70]}"
        print(line)


def find_cluster_csvs(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    out = []
    for p in sorted(root.rglob("*.csv")):
        if p.name in SKIP_NAMES or "cache" in p.relative_to(root).parts:
            continue
        out.append(p)
    return out


def pick_headline(reports: list[dict], root: Path) -> dict | None:
    """Pilih CSV 'hasil utama' untuk top-10: utamakan global_topics_summary."""
    for r in reports:
        if r["path"].name == "global_topics_summary.csv":
            return r
    # bila tidak ada hasil merge, pakai topics_summary terbesar
    summaries = [r for r in reports if r["path"].name.endswith("topics_summary.csv")]
    pool = summaries or reports
    return max(pool, key=lambda r: r["n_clusters"], default=None)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--output-dir", help="folder keluaran pipeline (berisi merged/ & cluster_chunk_*/)")
    g.add_argument("--csv", help="periksa satu berkas CSV saja")
    p.add_argument("--top", type=int, nargs="+", default=[10, 50],
                   help="level cluster terbesar yang dicetak; boleh beberapa "
                        "(default: 10 50 -> cetak top 10 lalu top 50)")
    p.add_argument("--per-file", action="store_true",
                   help="cetak top cluster untuk SETIAP CSV, bukan hanya hasil utama")
    args = p.parse_args()

    root = Path(args.csv if args.csv else args.output_dir)
    if not root.exists():
        sys.exit(f"path tidak ditemukan: {root}")

    csvs = find_cluster_csvs(root)
    reports = [r for r in (analyze_csv(c) for c in csvs) if r is not None]
    if not reports:
        sys.exit(f"tidak ada CSV berisi cluster di {root} "
                 f"(dicari kolom: {CLUSTER_COLS})")

    base = root if root.is_dir() else root.parent
    print(f"=== Jumlah cluster per CSV ({len(reports)} berkas) ===")
    print(f"  {'clusters':>8}  {'docs':>12}  {'outlier':>9}  {'filtered':>9}  file")
    for r in reports:
        try:
            rel = r["path"].relative_to(base)
        except ValueError:
            rel = r["path"].name
        print(f"  {r['n_clusters']:>8}  {r['n_docs_clustered']:>12,}  "
              f"{r['n_outlier']:>9,}  {r['n_filtered']:>9,}  {rel}")

    tops = list(dict.fromkeys(args.top))   # dedup, pertahankan urutan (mis. 10 lalu 50)
    if args.per_file or len(reports) == 1:
        for r in reports:
            for t in tops:
                print_top(r, t)
    else:
        headline = pick_headline(reports, base)
        if headline is not None:
            for t in tops:
                print_top(headline, t)
            print("\n(gunakan --per-file untuk top cluster tiap CSV)")


if __name__ == "__main__":
    main()
