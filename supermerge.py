"""
Super-merge: gabungkan hasil clustering dari semua project, ambil top-N cluster terbesar.

Scan api_jobs/*/output/merged/ yang sudah selesai (ada merge.done + g_centroids.npy),
lalu cluster ulang semua global-topic centroid lintas project menggunakan UMAP+HDBSCAN,
kemudian ambil top-N super cluster berdasarkan total jumlah dokumen.

Output (--output-dir):
  supermerge_top{N}_summary.csv  -- rank, super_topic, count, n_projects, keywords
  supermerge.done
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def find_completed_projects(jobs_dir: Path) -> list[dict]:
    projects = []
    for job_dir in sorted(jobs_dir.iterdir()):
        if not job_dir.is_dir():
            continue
        merged_dir = job_dir / "output" / "merged"
        if not (merged_dir / "merge.done").exists():
            continue
        g_centroids_path = merged_dir / "g_centroids.npy"
        g_ids_path       = merged_dir / "g_centroid_ids.json"
        summary_path     = merged_dir / "global_topics_summary.csv"
        if not (g_centroids_path.exists() and g_ids_path.exists() and summary_path.exists()):
            continue
        project_id = job_dir.name
        status_path = job_dir / "status.json"
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
                project_id = status.get("config", {}).get("project_id", project_id)
            except Exception:
                pass
        projects.append({
            "job_id":            job_dir.name,
            "project_id":        project_id,
            "g_centroids_path":  g_centroids_path,
            "g_ids_path":        g_ids_path,
            "summary_path":      summary_path,
        })
    return projects


def supermerge(
    jobs_dir: Path,
    output_dir: Path,
    top_n: int = 10,
    min_cluster_size: int = 0,
    seed: int = 42,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    projects = find_completed_projects(jobs_dir)
    if not projects:
        sys.exit(f"[supermerge] tidak ada project selesai (merge.done + g_centroids.npy) di {jobs_dir}")

    print(f"[supermerge] {len(projects)} project selesai ditemukan")

    all_centroids_list: list[np.ndarray] = []
    origins: list[tuple[int, int]] = []   # (project_idx, global_topic_id)
    all_summaries: list[pd.DataFrame] = []

    for pi, proj in enumerate(projects):
        centroids = np.load(proj["g_centroids_path"]).astype(np.float32)
        topic_ids = json.loads(proj["g_ids_path"].read_text(encoding="utf-8"))
        summary   = pd.read_csv(proj["summary_path"])
        all_centroids_list.append(centroids)
        for tid in topic_ids:
            origins.append((pi, tid))
        all_summaries.append(summary)
        print(f"  {proj['project_id'][:36]}: {len(topic_ids)} global topics")

    all_centroids = np.vstack(all_centroids_list).astype(np.float32)
    n_total = len(all_centroids)
    print(f"[supermerge] total centroid: {n_total} dari {len(projects)} project")

    if n_total < 2:
        sys.exit("[supermerge] terlalu sedikit centroid untuk di-cluster")

    # ---- UMAP + HDBSCAN ----
    from hdbscan import HDBSCAN
    from umap import UMAP

    mcs = min_cluster_size
    if mcs == 0:
        mcs = max(2, n_total // (top_n * 6))

    n_neighbors  = min(15, n_total - 1)
    n_components = min(5, n_total - 2)
    print(f"[supermerge] UMAP(n_neighbors={n_neighbors}, n_components={n_components}) + "
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

    valid_super = sorted(set(raw_labels.tolist()) - {-1})
    if not valid_super:
        print("[supermerge] HDBSCAN tidak menghasilkan cluster — tiap centroid jadi super topic sendiri")
        raw_labels  = np.arange(n_total, dtype=np.int32)
        valid_super = list(range(n_total))

    n_super = len(valid_super)
    print(f"[supermerge] {n_super} super cluster ditemukan")

    # Assign outlier centroid ke super topic terdekat
    sc_centroids = np.stack([
        all_centroids[raw_labels == s].mean(axis=0) for s in valid_super
    ]).astype(np.float32)
    sc_centroids /= np.linalg.norm(sc_centroids, axis=1, keepdims=True) + 1e-12

    sims       = all_centroids @ sc_centroids.T
    best_idx   = sims.argmax(axis=1)
    super_labels = np.array([valid_super[b] for b in best_idx], dtype=np.int32)

    # ---- Agregasi per super cluster ----
    super_info: dict[int, dict] = {
        s: {"count": 0, "kw_parts": [], "projects": set()}
        for s in valid_super
    }
    for row_idx, (pi, global_tid) in enumerate(origins):
        s       = int(super_labels[row_idx])
        summary = all_summaries[pi]
        proj    = projects[pi]
        row     = summary[summary["global_topic"] == global_tid]
        if not row.empty:
            super_info[s]["count"] += int(row.iloc[0].get("count", 0))
            kw = str(row.iloc[0].get("keywords", ""))
            if kw:
                super_info[s]["kw_parts"].append(kw)
        super_info[s]["projects"].add(proj["project_id"])

    rows = []
    for s in sorted(valid_super):
        info     = super_info[s]
        all_kw   = " ".join(info["kw_parts"])
        top_kw   = " ".join(w for w, _ in Counter(all_kw.split()).most_common(10))
        rows.append({
            "super_topic": s,
            "count":       info["count"],
            "n_projects":  len(info["projects"]),
            "keywords":    top_kw,
        })

    df = (
        pd.DataFrame(rows)
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    df.insert(0, "rank", df.index + 1)

    top_df  = df.head(top_n)
    out_csv = output_dir / f"supermerge_top{top_n}_summary.csv"
    top_df.to_csv(out_csv, index=False)
    print(f"[supermerge] top {top_n} super cluster -> {out_csv.name}")
    for _, r in top_df.iterrows():
        print(f"  #{int(r['rank'])}: {int(r['count']):,} docs | "
              f"{int(r['n_projects'])} project | {str(r['keywords'])[:60]}")

    (output_dir / "supermerge.done").write_text("ok")
    print(f"[supermerge] selesai -> {output_dir}/")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs-dir",          required=True, type=Path,
                   help="folder api_jobs/ yang berisi semua hasil pipeline per project")
    p.add_argument("--output-dir",        required=True, type=Path,
                   help="folder output supermerge")
    p.add_argument("--top-n",             type=int, default=10,
                   help="ambil N cluster terbesar (default: 10)")
    p.add_argument("--min-cluster-size",  type=int, default=0,
                   help="min_cluster_size HDBSCAN (0 = auto)")
    p.add_argument("--seed",              type=int, default=42)
    args = p.parse_args()
    supermerge(
        jobs_dir=args.jobs_dir,
        output_dir=args.output_dir,
        top_n=args.top_n,
        min_cluster_size=args.min_cluster_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
