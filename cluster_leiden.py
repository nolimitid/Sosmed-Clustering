"""
Pipeline klasterisasi Leiden untuk teks media sosial berbahasa Indonesia.

Menggantikan BERTopic (UMAP + HDBSCAN + c-TF-IDF) dengan pendekatan graph-based:
  embed -> k-NN graph -> Leiden community detection -> keyword TF-IDF per cluster

Keunggulan vs cluster.py (BERTopic):
  - Tidak butuh UMAP (tahap terperlambat) -> ~2x lebih cepat
  - Bekerja pada embedding langsung tanpa dimensionality reduction
  - Leiden menemukan jumlah komunitas secara alami, bisa di-tune via resolution

Output identik dengan cluster.py sehingga kompatibel dengan merge.py dan pipeline.py.

Usage:
  python cluster_leiden.py --input posts.parquet --text-col content --output-dir results/
  python cluster_leiden.py --input posts.parquet --text-col content \\
      --output-dir results/ --target-min 20 --target-max 50 --k 15
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Impor utilitas bersama dari cluster.py agar tidak duplikasi kode
from cluster import (
    ASSIGN_CHUNK,
    EMBED_CHUNK,
    assign_by_centroid,
    build_topic_graph,
    compute_centroids,
    compute_silhouette,
    detect_meta_topics,
    embed_texts,
    get_top_docs_by_centroid,
    load_dataframe,
    visualize_topic_graph,
)
from language_id import LanguageDetector, lang_name
from preprocess import clean_text, is_informative, load_slang_lexicon, load_stopwords

HERE = Path(__file__).resolve().parent
DEFAULT_LEXICON   = HERE / "slang_lexicon.csv"
DEFAULT_STOPWORDS = HERE / "stopwords_id.txt"


# ---------------------------------------------------------------------------
# k-NN Graph
# ---------------------------------------------------------------------------

def build_knn_graph(
    embeddings: np.ndarray,
    k: int = 15,
    sim_threshold: float = 0.0,
) -> "ig.Graph":
    """
    Bangun k-nearest-neighbor graph dari embeddings (L2-normalized).
    Coba FAISS dulu (cepat), fallback ke sklearn NearestNeighbors.
    Edge hanya dibuat bila cosine similarity >= sim_threshold.
    """
    import igraph as ig

    n, dim = embeddings.shape
    emb_f32 = np.asarray(embeddings, dtype=np.float32)

    # -- k-NN search --
    try:
        import faiss
        index = faiss.IndexFlatIP(dim)   # inner product = cosine sim (vector sudah ternormalisasi)
        index.add(emb_f32)
        sims, idxs = index.search(emb_f32, k + 1)   # +1: hasil pertama = diri sendiri
        print(f"[knn] menggunakan FAISS: k={k}, {n} node")
    except ImportError:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1)
        nn.fit(emb_f32)
        dists, idxs = nn.kneighbors(emb_f32)
        sims = 1.0 - dists   # ubah distance ke similarity
        print(f"[knn] menggunakan sklearn NearestNeighbors: k={k}, {n} node")

    # -- Bangun edge list (undirected, no self-loop, no duplicate) --
    seen:    set[tuple[int, int]] = set()
    edges:   list[tuple[int, int]] = []
    weights: list[float] = []

    for i in range(n):
        for pos in range(1, k + 1):        # mulai dari 1: skip self
            j = int(idxs[i, pos])
            s = float(sims[i, pos])
            if s < sim_threshold:
                continue
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                edges.append(key)
                weights.append(s)

    g = ig.Graph(n=n, edges=edges)
    g.es["weight"] = weights
    print(f"[knn] {len(edges):,} edge dibuat (sim >= {sim_threshold})")
    return g


# ---------------------------------------------------------------------------
# Leiden clustering dengan tuning resolution
# ---------------------------------------------------------------------------

def leiden_cluster(
    g: "ig.Graph",
    target_min: int,
    target_max: int,
    seed: int,
    max_retries: int = 6,
) -> np.ndarray:
    """
    Jalankan Leiden (RBConfigurationVertexPartition) dengan penyesuaian
    resolution_parameter otomatis agar jumlah komunitas masuk rentang
    [target_min, target_max].

    Kembalikan array label (int32), -1 untuk node terisolasi (tanpa edge).
    """
    import leidenalg

    n          = g.vcount()
    has_weight = g.ecount() > 0
    resolution = 1.0

    labels = np.full(n, -1, dtype=np.int32)

    for attempt in range(1, max_retries + 1):
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight" if has_weight else None,
            resolution_parameter=resolution,
            seed=seed,
        )
        n_clusters = len(partition)
        print(f"[leiden] percobaan {attempt}: resolution={resolution:.4f} "
              f"-> {n_clusters} komunitas")

        if target_min <= n_clusters <= target_max:
            break

        if attempt < max_retries:
            if n_clusters < target_min:
                resolution *= 2.0
            else:
                resolution *= 0.55

    if not (target_min <= n_clusters <= target_max):
        print(f"[leiden] PERINGATAN: {n_clusters} komunitas di luar "
              f"target [{target_min}, {target_max}]")

    for mid, community in enumerate(partition):
        for node in community:
            labels[node] = mid

    # Node terisolasi (tanpa tetangga) tetap -1
    n_outlier = int((labels == -1).sum())
    if n_outlier:
        print(f"[leiden] {n_outlier} node terisolasi (label -1)")

    return labels


# ---------------------------------------------------------------------------
# Keyword extraction — c-TF-IDF sederhana tanpa BERTopic
# ---------------------------------------------------------------------------

def extract_keywords(
    texts: list[str],
    labels: np.ndarray,
    stopwords: list[str],
    top_n: int = 10,
) -> dict[int, str]:
    """
    Ekstrak kata kunci per cluster menggunakan c-TF-IDF:
      - TF: frekuensi kata dalam cluster
      - IDF: log(n_cluster / df_cluster) + 1
    Mengembalikan dict {topic_id: "kw1, kw2, ..."}.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    topic_ids = sorted(set(labels.tolist()) - {-1})
    if not topic_ids:
        return {}

    topic_docs = []
    for tid in topic_ids:
        cluster_texts = [texts[i] for i, l in enumerate(labels) if l == tid]
        topic_docs.append(" ".join(cluster_texts))

    try:
        vec = CountVectorizer(
            stop_words=stopwords,
            ngram_range=(1, 2),
            min_df=2,
            max_features=20_000,
        )
        X    = vec.fit_transform(topic_docs).toarray().astype(np.float32)
        vocab = vec.get_feature_names_out()

        # TF per topic (normalized)
        tf = X / (X.sum(axis=1, keepdims=True) + 1e-9)
        # IDF across topics
        df  = (X > 0).sum(axis=0).astype(np.float32)
        idf = np.log(len(topic_ids) / (df + 1.0)) + 1.0
        # c-TF-IDF score
        score = tf * idf   # (n_topics, vocab)

        result: dict[int, str] = {}
        for i, tid in enumerate(topic_ids):
            top_idx = score[i].argsort()[::-1][:top_n]
            kws = [vocab[j] for j in top_idx if score[i, j] > 0]
            result[tid] = ", ".join(kws)
        return result

    except Exception as exc:
        print(f"[keywords] ekstraksi gagal: {exc}")
        return {tid: "" for tid in topic_ids}


# ---------------------------------------------------------------------------
# Export (format identik dengan cluster.py)
# ---------------------------------------------------------------------------

def export_results(
    out_dir: Path,
    df: pd.DataFrame,
    keywords_map: dict[int, str],
    metrics: dict,
    centroid_docs_map: dict[int, list[str]] | None = None,
    meta_map: dict[int, int] | None = None,
    graph_edges: dict | None = None,
    centroids: np.ndarray | None = None,
    centroid_topic_ids: list[int] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if centroids is not None and centroid_topic_ids is not None:
        np.save(out_dir / "centroids.npy", centroids)
        (out_dir / "centroid_topic_ids.json").write_text(
            json.dumps(centroid_topic_ids)
        )

    df.to_csv(out_dir / "assignments.csv", index=False)

    # topics_summary — format sama dengan cluster.py
    topic_ids = sorted(set(df["topic"].tolist()) - {-1, -2})
    counts    = df[df["topic"] >= 0]["topic"].value_counts()
    rows = []
    for tid in topic_ids:
        kw  = keywords_map.get(tid, "")
        name = kw.split(", ")[0] if kw else f"topic_{tid}"
        row: dict = {
            "Topic":    tid,
            "Count":    int(counts.get(tid, 0)),
            "Name":     name,
            "Keywords": kw,
        }
        if centroid_docs_map:
            row["Top5CentroidDocs"] = " ||| ".join(
                centroid_docs_map.get(tid, [])
            )
        if meta_map:
            row["meta_topic"] = meta_map.get(tid, -1)
        rows.append(row)

    info = pd.DataFrame(rows).sort_values("Count", ascending=False)
    info.to_csv(out_dir / "topics_summary.csv", index=False)

    if graph_edges:
        pd.DataFrame(
            [{"topic_a": ti, "topic_b": tj, **data}
             for (ti, tj), data in graph_edges.items()]
        ).to_csv(out_dir / "topic_graph_edges.csv", index=False)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if meta_map is not None and graph_edges is not None:
        try:
            visualize_topic_graph(
                [t for t in meta_map if t >= 0],
                graph_edges, meta_map, info,
                out_dir / "topic_graph.html",
            )
        except Exception as exc:
            print(f"[export] visualisasi graph dilewati: {exc}")

    print(f"[export] hasil ditulis ke {out_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input",      required=True, help="berkas csv/tsv/parquet/jsonl")
    p.add_argument("--text-col",   required=True, help="kolom teks")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--model",      default="LazarusNLP/all-indo-e5-small-v4")
    p.add_argument("--e5-prefix",  action="store_true")
    p.add_argument("--k",          type=int, default=15,
                   help="jumlah tetangga terdekat untuk k-NN graph (default: 15)")
    p.add_argument("--sim-threshold", type=float, default=0.0,
                   help="ambang cosine similarity untuk edge k-NN graph (default: 0.0)")
    p.add_argument("--target-min", type=int, default=20)
    p.add_argument("--target-max", type=int, default=50)
    p.add_argument("--fit-sample", type=int, default=300_000,
                   help="maks teks unik untuk build graph & Leiden (0 = semua)")
    p.add_argument("--min-content-tokens", type=int, default=2)
    p.add_argument("--assign-outliers", action="store_true")
    p.add_argument("--keep-emoji",  action="store_true")
    p.add_argument("--no-slang-norm", action="store_true")
    p.add_argument("--no-lang-filter", action="store_true")
    p.add_argument("--keep-langs",  default="id,ms,jv,su,min,ban,ace")
    p.add_argument("--lang-backend", choices=["fasttext", "langdetect"],
                   default="fasttext")
    p.add_argument("--lang-model",  default=None)
    p.add_argument("--lang-min-conf", type=float, default=0.5)
    p.add_argument("--sample",      type=int, default=0)
    p.add_argument("--graph-threshold", type=float, default=0.25,
                   help="ambang similarity untuk topic graph / meta-topic (default: 0.25)")
    p.add_argument("--batch-size",  type=int, default=128)
    p.add_argument("--device",      default=None)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--done-file",   default=None)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # 1. Muat ---------------------------------------------------------------
    df = load_dataframe(Path(args.input))
    if args.text_col not in df.columns:
        sys.exit(f"Kolom '{args.text_col}' tidak ditemukan. Tersedia: {list(df.columns)}")
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=args.seed)
    df = df.reset_index(drop=True)
    n_raw = len(df)
    print(f"[load] {n_raw} baris")

    # 2. Bersihkan ----------------------------------------------------------
    lexicon   = {} if args.no_slang_norm else load_slang_lexicon(DEFAULT_LEXICON)
    stopwords = load_stopwords(DEFAULT_STOPWORDS)
    print(f"[clean] entri kamus slang: {len(lexicon)}")
    df["clean_text"] = df[args.text_col].map(
        lambda t: clean_text(t, lexicon=lexicon, keep_emoji=args.keep_emoji)
    )

    # 2b. Filter bahasa -----------------------------------------------------
    detector = None
    keep_langs: set[str] = set()
    n_dropped_lang = 0
    if not args.no_lang_filter:
        keep_langs = {c.strip().lower() for c in args.keep_langs.split(",") if c.strip()}
        detector   = LanguageDetector.load(model_path=args.lang_model,
                                           prefer=args.lang_backend)
        print(f"[lang] backend: {detector.name}; dipertahankan: {sorted(keep_langs)}")
        uniq_clean = df["clean_text"].drop_duplicates().tolist()
        codes, confs = detector.detect_batch(uniq_clean)
        code_map = dict(zip(uniq_clean, codes))
        conf_map = dict(zip(uniq_clean, confs))
        df["lang"]      = df["clean_text"].map(code_map).fillna("und")
        df["lang_conf"] = df["clean_text"].map(conf_map).fillna(0.0).round(3)
        foreign = (~df["lang"].isin(keep_langs)) & (df["lang_conf"] >= args.lang_min_conf)
        n_dropped_lang  = int(foreign.sum())
        if n_dropped_lang and detector.available:
            print(f"[lang] membuang {n_dropped_lang} baris non-Indonesia "
                  f"({100 * n_dropped_lang / max(n_raw, 1):.1f}%)")
        df = df[~foreign].reset_index(drop=True)
        print(f"[lang] {len(df)}/{n_raw} baris dipertahankan")

    # 2c. Filter non-informatif ---------------------------------------------
    df["informative"] = df["clean_text"].map(
        lambda t: is_informative(t, stopwords, args.min_content_tokens)
    )
    n_junk = int((~df["informative"]).sum())
    print(f"[clean] non-informatif: {n_junk} ({100 * n_junk / max(len(df), 1):.1f}%)")

    # 3. Deduplikasi --------------------------------------------------------
    work = df[df["informative"]]
    unique_texts = work["clean_text"].drop_duplicates().tolist()
    n_unique     = len(unique_texts)
    print(f"[dedup] {len(work)} informatif -> {n_unique} teks unik "
          f"({100 * (1 - n_unique / max(len(work), 1)):.1f}% duplikat)")
    if n_unique < 50:
        sys.exit(f"Hanya {n_unique} teks unik; terlalu sedikit untuk clustering.")

    # 4. Embed --------------------------------------------------------------
    embeddings = embed_texts(
        unique_texts, args.model, args.batch_size, args.device,
        cache_dir=out_dir / "cache", e5_prefix=args.e5_prefix,
    )

    # 5. Pilih sampel untuk build graph -------------------------------------
    fit_n = n_unique if args.fit_sample == 0 else min(args.fit_sample, n_unique)
    if fit_n < n_unique:
        fit_idx = np.sort(rng.choice(n_unique, fit_n, replace=False))
        print(f"[leiden] build graph pada sampel {fit_n}/{n_unique} teks unik")
    else:
        fit_idx = np.arange(n_unique)

    fit_emb   = np.asarray(embeddings[fit_idx], dtype=np.float32)
    fit_texts = [unique_texts[i] for i in fit_idx]

    # 6. k-NN graph + Leiden -----------------------------------------------
    t0    = time.time()
    graph = build_knn_graph(fit_emb, k=args.k, sim_threshold=args.sim_threshold)
    labels_fit = leiden_cluster(
        graph, args.target_min, args.target_max, args.seed,
    )
    n_clusters = len(set(labels_fit.tolist()) - {-1})
    print(f"[leiden] {n_clusters} cluster ditemukan dalam "
          f"{time.time() - t0:.1f} detik")

    # 7. Tetapkan semua teks unik via sentroid ------------------------------
    all_labels, threshold = assign_by_centroid(
        embeddings, fit_idx, labels_fit, args.assign_outliers,
    )

    # 8. Keyword extraction -------------------------------------------------
    fit_label_full = all_labels[fit_idx]   # label fit texts setelah reassign
    keywords_map   = extract_keywords(fit_texts, fit_label_full,
                                      sorted(stopwords))
    print(f"[keywords] {len(keywords_map)} cluster punya keywords")

    # 9. Centroid docs & graph meta-topic ----------------------------------
    centroid_docs_map = get_top_docs_by_centroid(unique_texts, embeddings, all_labels)
    chunk_centroids, chunk_topic_ids = compute_centroids(embeddings, all_labels)
    topic_ids_graph, graph_edges = build_topic_graph(
        embeddings, all_labels, threshold=args.graph_threshold,
    )
    meta_map = detect_meta_topics(topic_ids_graph, graph_edges)

    # 10. Propagasi label ke baris asli ------------------------------------
    label_map = dict(zip(unique_texts, all_labels.tolist()))
    df["topic"] = df["clean_text"].map(label_map)
    df["topic"] = df["topic"].where(df["informative"], other=-2)
    df["topic"] = df["topic"].fillna(-2).astype(int)
    df["topic_label"] = df["topic"].map(
        lambda t: keywords_map.get(t, "").split(", ")[0] if t >= 0 else
                  ("OUTLIER_NO_TOPIC" if t == -1 else "FILTERED_NON_INFORMATIVE")
    )
    df["meta_topic"] = df["topic"].map(meta_map)
    df.loc[df["topic"] < 0, "meta_topic"] = df.loc[df["topic"] < 0, "topic"]
    df["meta_topic"] = df["meta_topic"].fillna(-1).astype(int)

    # 11. Metrik + ekspor --------------------------------------------------
    assigned = all_labels[all_labels >= 0]
    sizes    = pd.Series(assigned).value_counts()
    metrics  = {
        "rows_total":                    n_raw,
        "lang_filter_backend":           detector.name if detector else "disabled",
        "lang_kept":                     sorted(keep_langs) if keep_langs else None,
        "rows_dropped_non_indonesian":   int(n_dropped_lang),
        "rows_after_language_filter":    int(len(df)),
        "rows_filtered_non_informative": n_junk,
        "unique_texts":                  n_unique,
        "fit_sample_size":               int(fit_n),
        "k_neighbors":                   args.k,
        "leiden_resolution_used":        "auto",
        "final_clusters":                int(len(sizes)),
        "outlier_pct_of_unique":         round(100 * float(np.mean(all_labels == -1)), 2),
        "largest_cluster_share_pct":     round(
            100 * float(sizes.max()) / max(len(assigned), 1), 2
        ) if len(sizes) else None,
        "model":                         args.model,
        "seed":                          args.seed,
        "method":                        "leiden",
    }
    print(json.dumps(metrics, indent=2))

    export_results(
        out_dir, df.drop(columns=["informative"]), keywords_map, metrics,
        centroid_docs_map, meta_map, graph_edges,
        centroids=chunk_centroids, centroid_topic_ids=chunk_topic_ids,
    )

    if args.done_file:
        Path(args.done_file).write_text("ok")
        print(f"[done] checkpoint -> {args.done_file}")


if __name__ == "__main__":
    main()
