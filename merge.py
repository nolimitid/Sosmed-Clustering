"""
Merge per-chunk clustering results into global topics.

Scans --base-dir for subdirectories that contain cluster.done + centroids.npy,
loads all centroids, clusters them with UMAP+HDBSCAN to produce global topics,
then writes a merged topic summary and per-chunk assignment updates.

Usage:
  python merge.py --base-dir data/my_project \\
      --output-dir data/my_project/merged --target-min 20 --target-max 50

Output files (in --output-dir):
  global_topics_summary.csv   — global topics: keywords, count, meta_topic
  global_topic_mapping.json   — {chunk_dir: {local_topic_id: global_topic_id}}
  topic_graph_edges.csv       — centroid-similarity edges between global topics
  topic_graph.html            — interactive network visualization
  merge.done                  — written when merge completes successfully
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from cluster import detect_meta_topics, visualize_topic_graph


# ---------------------------------------------------------------------------
# Discovery & loading
# ---------------------------------------------------------------------------

def discover_chunk_dirs(base_dir: Path) -> list[Path]:
    """Return sorted list of chunk result dirs with cluster.done + centroids.npy."""
    dirs = sorted(
        d for d in base_dir.iterdir()
        if d.is_dir()
        and (d / "cluster.done").exists()
        and (d / "centroids.npy").exists()
        and (d / "centroid_topic_ids.json").exists()
    )
    return dirs


def load_chunk_info(chunk_dir: Path) -> dict:
    centroids = np.load(chunk_dir / "centroids.npy")             # (n_local, dim)
    topic_ids: list[int] = json.loads(
        (chunk_dir / "centroid_topic_ids.json").read_text()
    )
    summary: pd.DataFrame | None = None
    summary_path = chunk_dir / "topics_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
    return {"dir": chunk_dir, "centroids": centroids,
            "topic_ids": topic_ids, "summary": summary}


# ---------------------------------------------------------------------------
# Keyword aggregation
# ---------------------------------------------------------------------------

def aggregate_keywords(local_topics: list[dict], top_n: int = 10) -> str:
    """Union keywords across local topics weighted by doc count."""
    counter: Counter = Counter()
    for entry in local_topics:
        kws = entry.get("keywords", "")
        if isinstance(kws, str) and kws:
            for kw in kws.split(", "):
                kw = kw.strip()
                if kw:
                    counter[kw] += max(1, int(entry.get("count", 1)))
    return ", ".join(w for w, _ in counter.most_common(top_n))


# ---------------------------------------------------------------------------
# Graph (centroid-similarity based — no raw docs at merge level)
# ---------------------------------------------------------------------------

def build_centroid_graph(
    centroids: np.ndarray,       # (n_global, dim) — L2-normalized
    topic_ids: list[int],
    threshold: float = 0.30,
) -> dict[tuple[int, int], dict]:
    """Build topic graph using pairwise centroid similarity at merge level."""
    sim = centroids @ centroids.T
    np.fill_diagonal(sim, 0.0)
    edges: dict[tuple[int, int], dict] = {}
    n = len(topic_ids)
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim[i, j])
            if s >= threshold:
                edges[(topic_ids[i], topic_ids[j])] = {
                    "cooccurrence_count": 0,
                    "cooccurrence_rate": 0.0,
                    "centroid_similarity": round(s, 4),
                }
    print(f"[merge-graph] {n} topik global, {len(edges)} edge "
          f"(centroid similarity >= {threshold})")
    return edges


# ---------------------------------------------------------------------------
# Core merge logic
# ---------------------------------------------------------------------------

def run_merge(
    base_dir: Path,
    output_dir: Path,
    target_min: int,
    target_max: int,
    min_cluster_size: int,
    seed: int,
    graph_threshold: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_dirs = discover_chunk_dirs(base_dir)
    if not chunk_dirs:
        sys.exit(f"[merge] tidak ada chunk result dir yang selesai di {base_dir}\n"
                 f"  (cari folder berisi cluster.done + centroids.npy)")
    print(f"[merge] {len(chunk_dirs)} chunk result dir ditemukan")

    all_chunks = [load_chunk_info(d) for d in chunk_dirs]

    # Stack semua local centroids: shape (total_local_topics, dim)
    all_centroids = np.vstack([c["centroids"] for c in all_chunks]).astype(np.float32)
    n_total = len(all_centroids)
    print(f"[merge] total centroid lokal: {n_total} dari {len(chunk_dirs)} chunk")

    # Mapping: row_idx -> (chunk_idx, local_topic_id)
    origins: list[tuple[int, int]] = []
    for ci, chunk in enumerate(all_chunks):
        for tid in chunk["topic_ids"]:
            origins.append((ci, tid))

    too_few = n_total < max(2, target_min)
    if too_few:
        print(f"[merge] PERINGATAN: hanya {n_total} centroid lokal < target_min={target_min}. "
              f"Lanjut dengan {n_total} topik global (tiap centroid lokal = 1 global topic).")

    if too_few:
        # Skip UMAP/HDBSCAN — tiap centroid lokal langsung jadi 1 global topic
        raw_labels = np.arange(n_total, dtype=np.int32)
    else:
        # ---- UMAP + HDBSCAN pada semua centroid lokal ----
        from hdbscan import HDBSCAN
        from umap import UMAP

        mcs = min_cluster_size
        if mcs == 0:
            # heuristik: targetkan ~target_max global topics dari n_total local centroids
            mcs = max(2, n_total // (target_max * 4))
            mcs = min(mcs, max(2, n_total // max(target_min, 1)))

        n_neighbors = min(15, n_total - 1)
        n_components = min(5, n_total - 2)
        print(f"[merge] UMAP(n_neighbors={n_neighbors}, n_components={n_components}) + "
              f"HDBSCAN(min_cluster_size={mcs}) pada {n_total} centroid")

        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=0.0,
            metric="cosine",
            random_state=seed,
        )
        reduced = umap_model.fit_transform(all_centroids)

        hdbscan_model = HDBSCAN(
            min_cluster_size=mcs,
            min_samples=max(1, mcs // 4),
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        raw_labels = hdbscan_model.fit_predict(reduced)
        n_natural = len(set(raw_labels.tolist()) - {-1})
        print(f"[merge] klaster global alami: {n_natural}")

        # Kurangi bila > target_max
        if n_natural > target_max:
            print(f"[merge] mengurangi {n_natural} -> {target_max} topik global "
                  f"(AgglomerativeClustering)")
            from sklearn.cluster import AgglomerativeClustering
            valid_mask = raw_labels != -1
            agg = AgglomerativeClustering(n_clusters=target_max, metric="euclidean",
                                          linkage="ward")
            new_valid = agg.fit_predict(reduced[valid_mask])
            raw_labels[valid_mask] = new_valid
            raw_labels[~valid_mask] = -1

        if n_natural < target_min:
            print(f"[merge] PERINGATAN: {n_natural} topik global < target_min={target_min}. "
                  f"Data atau jumlah chunk mungkin belum cukup banyak.")

    # Assign outlier centroid lokal ke global topic terdekat (nearest centroid)
    valid_global_ids = sorted(set(raw_labels.tolist()) - {-1})
    if not valid_global_ids:
        sys.exit("[merge] tidak ada global topic yang berhasil dibuat")

    g_centroids = np.stack([
        all_centroids[raw_labels == g].mean(axis=0) for g in valid_global_ids
    ]).astype(np.float32)
    g_centroids /= np.linalg.norm(g_centroids, axis=1, keepdims=True) + 1e-12

    sims = all_centroids @ g_centroids.T                  # (n_total, n_global)
    best_idx = sims.argmax(axis=1)
    global_labels = np.array([valid_global_ids[b] for b in best_idx], dtype=np.int32)

    n_global = len(valid_global_ids)
    print(f"[merge] {n_global} topik global final")

    # ---- Bangun mapping chunk → local_topic → global_topic ----
    topic_mapping: dict[str, dict[str, int]] = {}
    for row_idx, (ci, local_tid) in enumerate(origins):
        key = str(all_chunks[ci]["dir"])
        if key not in topic_mapping:
            topic_mapping[key] = {}
        topic_mapping[key][str(local_tid)] = int(global_labels[row_idx])

    (output_dir / "global_topic_mapping.json").write_text(
        json.dumps(topic_mapping, indent=2)
    )
    print("[merge] topic mapping disimpan -> global_topic_mapping.json")

    # ---- Agregasi info per global topic ----
    global_info: dict[int, dict] = {
        g: {"count": 0, "local_topics": []} for g in valid_global_ids
    }
    for row_idx, (ci, local_tid) in enumerate(origins):
        g = int(global_labels[row_idx])
        chunk = all_chunks[ci]
        entry: dict = {"topic_id": local_tid, "keywords": "", "count": 0}
        if chunk["summary"] is not None:
            row = chunk["summary"][chunk["summary"]["Topic"] == local_tid]
            if not row.empty:
                entry["keywords"] = str(row.iloc[0].get("Keywords", ""))
                entry["count"] = int(row.iloc[0].get("Count", 0))
        global_info[g]["count"] += entry["count"]
        global_info[g]["local_topics"].append(entry)

    # ---- Graph analysis pada global centroids ----
    graph_edges = build_centroid_graph(g_centroids, valid_global_ids,
                                       threshold=graph_threshold)
    meta_map = detect_meta_topics(valid_global_ids, graph_edges)

    # ---- Global topic summary ----
    rows = []
    for g in sorted(valid_global_ids):
        info = global_info[g]
        rows.append({
            "global_topic": g,
            "count": info["count"],
            "n_local_topics": len(info["local_topics"]),
            "keywords": aggregate_keywords(info["local_topics"]),
            "meta_topic": meta_map.get(g, -1),
        })
    global_summary = pd.DataFrame(rows)
    global_summary.to_csv(output_dir / "global_topics_summary.csv", index=False)
    print(f"[merge] {len(rows)} topik global -> global_topics_summary.csv")

    # Simpan g_centroids untuk super-merge lintas project
    np.save(output_dir / "g_centroids.npy", g_centroids)
    (output_dir / "g_centroid_ids.json").write_text(json.dumps(valid_global_ids))

    if graph_edges:
        pd.DataFrame(
            [{"topic_a": ti, "topic_b": tj, **data}
             for (ti, tj), data in graph_edges.items()]
        ).to_csv(output_dir / "topic_graph_edges.csv", index=False)

    try:
        fake_info = global_summary.rename(
            columns={"global_topic": "Topic", "keywords": "Name"}
        )
        visualize_topic_graph(
            valid_global_ids, graph_edges, meta_map,
            fake_info, output_dir / "topic_graph.html",
        )
    except Exception as exc:
        print(f"[merge] visualisasi graph dilewati: {exc}")

    # ---- Update assignments per chunk ----
    print("[merge] memperbarui assignments per chunk dengan global_topic ...")
    for chunk in all_chunks:
        _update_chunk_assignments(chunk, topic_mapping)

    (output_dir / "merge.done").write_text("ok")
    print(f"[merge] selesai -> {output_dir}/")


def _update_chunk_assignments(chunk: dict, topic_mapping: dict) -> None:
    """Tambahkan kolom global_topic ke assignments chunk dan simpan sebagai parquet."""
    chunk_dir: Path = chunk["dir"]
    assignments_path = chunk_dir / "assignments.csv"
    if not assignments_path.exists():
        return
    mapping = topic_mapping.get(str(chunk_dir), {})
    if not mapping:
        return
    df = pd.read_csv(assignments_path)
    if "topic" not in df.columns:
        return
    df["global_topic"] = df["topic"].map(lambda t: mapping.get(str(t), -1))
    out = chunk_dir / "assignments_global.parquet"
    _write_parquet(df, out)
    print(f"[merge] {chunk_dir.name}: {len(df):,} baris -> assignments_global.parquet")


def _as_string(v):
    """Ubah satu nilai menjadi string (decode bytes), pertahankan yang kosong."""
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return pd.NA
    return str(v)


def _write_parquet(df: pd.DataFrame, out: Path) -> None:
    """Tulis parquet dengan aman terhadap kolom object bertipe campuran.

    Kolom seperti `from_id` dari sumber Elasticsearch bisa berisi campuran int +
    bytes/str dalam satu kolom object, sehingga pyarrow gagal menebak satu tipe
    Arrow ("Expected bytes, got a 'int' object"). Bila itu terjadi, paksa kolom
    object menjadi string (id paling aman disimpan sebagai teks) lalu tulis ulang.
    """
    try:
        df.to_parquet(out, index=False)
        return
    except Exception as exc:
        obj_cols = list(df.select_dtypes(include=["object"]).columns)
        print(f"[merge] to_parquet gagal ({exc}); mengonversi kolom object "
              f"ke string lalu coba lagi: {obj_cols}")
        for col in obj_cols:
            df[col] = df[col].map(_as_string).astype("string")
        df.to_parquet(out, index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-dir", required=True,
                   help="direktori yang berisi subfolder cluster_* hasil cluster.py")
    p.add_argument("--output-dir", required=True,
                   help="direktori output untuk hasil merge")
    p.add_argument("--target-min", type=int, default=20)
    p.add_argument("--target-max", type=int, default=50)
    p.add_argument("--min-cluster-size", type=int, default=0,
                   help="min_cluster_size HDBSCAN saat cluster centroid (0 = otomatis)")
    p.add_argument("--graph-threshold", type=float, default=0.30,
                   help="ambang centroid similarity untuk edge graph (default: 0.30)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--done-file", default=None,
                   help="tulis 'ok' ke path ini setelah merge selesai")
    args = p.parse_args()

    run_merge(
        base_dir=Path(args.base_dir),
        output_dir=Path(args.output_dir),
        target_min=args.target_min,
        target_max=args.target_max,
        min_cluster_size=args.min_cluster_size,
        seed=args.seed,
        graph_threshold=args.graph_threshold,
    )

    if args.done_file:
        Path(args.done_file).write_text("ok")


if __name__ == "__main__":
    main()
