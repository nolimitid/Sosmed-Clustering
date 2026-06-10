"""
Pipeline klasterisasi untuk teks media sosial berbahasa Indonesia informal,
dirancang agar mampu menangani jutaan baris.

Pipeline:
  muat -> bersihkan/normalkan -> saring sampah -> deduplikasi
       -> embed (per-chunk, dapat dilanjutkan, memmap float16)
       -> latih BERTopic (UMAP + HDBSCAN + c-TF-IDF) pada sampel representatif
       -> tetapkan dokumen sisanya ke sentroid topik terdekat
       -> gabungkan ke rentang topik target -> ekspor penetapan, ringkasan, metrik

Penggunaan:
  python cluster.py --input posts.csv --text-col text --output-dir results
  python cluster.py --input posts.parquet --text-col text \
      --model LazarusNLP/all-indo-e5-small-v4 \
      --fit-sample 300000 --target-min 20 --target-max 50

Format input: .csv, .tsv, .parquet, .jsonl (satu objek JSON per baris).
GPU sangat disarankan di atas ~50rb dokumen (mis. Google Colab T4).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import clean_text, is_informative, load_slang_lexicon, load_stopwords

HERE = Path(__file__).resolve().parent
DEFAULT_LEXICON = HERE / "data" / "slang_lexicon.csv"
DEFAULT_STOPWORDS = HERE / "data" / "stopwords_id.txt"

EMBED_CHUNK = 100_000          # jumlah teks yang di-encode per checkpoint
ASSIGN_CHUNK = 200_000         # jumlah teks per batch sentroid-terdekat


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".jsonl", ".json"):
        return pd.read_json(path, lines=True)
    raise ValueError(f"Format input tidak didukung: {suffix}")


def corpus_fingerprint(texts: list[str], model_name: str, e5_prefix: bool) -> str:
    h = hashlib.sha1()
    h.update(f"{model_name}|{e5_prefix}|{len(texts)}".encode())
    for t in texts:
        h.update(t.encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def embed_texts(
    texts: list[str],
    model_name: str,
    batch_size: int,
    device: str | None,
    cache_dir: Path,
    e5_prefix: bool,
) -> np.ndarray:
    """
    Meng-encode teks per chunk dengan checkpoint di disk, menghasilkan memmap
    float16. Aman untuk dihentikan dan dilanjutkan (mis. Colab terputus): chunk
    yang sudah selesai dilewati pada run berikutnya. Mengembalikan array memmap
    yang hanya-baca (read-only).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = corpus_fingerprint(texts, model_name, e5_prefix)
    final = cache_dir / f"emb_{key}.npy"
    if final.exists():
        print(f"[embed] memakai embedding dari cache: {final}")
        return np.load(final, mmap_mode="r")

    parts_dir = cache_dir / f"emb_{key}_parts"
    parts_dir.mkdir(exist_ok=True)
    n = len(texts)
    n_chunks = (n + EMBED_CHUNK - 1) // EMBED_CHUNK

    from sentence_transformers import SentenceTransformer

    print(f"[embed] memuat model {model_name} ...")
    model = SentenceTransformer(model_name, device=device)

    t0 = time.time()
    for ci in range(n_chunks):
        part = parts_dir / f"part_{ci:05d}.npy"
        if part.exists():
            continue
        lo, hi = ci * EMBED_CHUNK, min((ci + 1) * EMBED_CHUNK, n)
        chunk = texts[lo:hi]
        if e5_prefix:
            chunk = [f"query: {t}" for t in chunk]
        emb = model.encode(
            chunk,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float16)
        tmp = part.with_suffix(".tmp.npy")
        np.save(tmp, emb)
        tmp.rename(part)
        done = hi
        rate = done / max(time.time() - t0, 1e-9)
        print(f"[embed] chunk {ci + 1}/{n_chunks} selesai "
              f"({done}/{n}, ~{rate:.0f} teks/dtk, "
              f"perkiraan sisa {(n - done) / max(rate, 1e-9) / 60:.1f} menit)")

    # gabungkan part menjadi satu memmap tanpa memuat semuanya ke RAM
    dim = np.load(parts_dir / "part_00000.npy", mmap_mode="r").shape[1]
    out = np.lib.format.open_memmap(final, mode="w+", dtype=np.float16, shape=(n, dim))
    pos = 0
    for ci in range(n_chunks):
        emb = np.load(parts_dir / f"part_{ci:05d}.npy")
        out[pos:pos + len(emb)] = emb
        pos += len(emb)
    out.flush()
    del out
    shutil.rmtree(parts_dir)
    print(f"[embed] {n} embedding disimpan ke cache (dim={dim}, float16) -> {final}")
    return np.load(final, mmap_mode="r")


# ---------------------------------------------------------------------------
# Klasterisasi
# ---------------------------------------------------------------------------

def build_topic_model(min_cluster_size: int, stopwords: list[str], seed: int):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=seed,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=max(5, min_cluster_size // 4),
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        stop_words=stopwords,
        ngram_range=(1, 2),
        min_df=5,
    )
    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        verbose=True,
    )


def fit_with_target_range(
    docs: list[str],
    embeddings: np.ndarray,
    stopwords: list[str],
    target_min: int,
    target_max: int,
    min_cluster_size: int,
    seed: int,
    max_retries: int = 3,
):
    """
    Melatih BERTopic; jika jumlah klaster alami melebihi target_max, gabungkan
    hingga target_max. Jika di bawah target_min, ulangi dengan min_cluster_size
    yang lebih kecil (klaster lebih banyak dan lebih halus), maksimal sebanyak
    max_retries kali.
    """
    mcs = min_cluster_size
    for attempt in range(1, max_retries + 1):
        print(f"[cluster] percobaan {attempt}: min_cluster_size={mcs}")
        topic_model = build_topic_model(mcs, stopwords, seed)
        topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)
        n_topics = len(set(topics)) - (1 if -1 in topics else 0)
        print(f"[cluster] klaster alami yang ditemukan: {n_topics}")

        if n_topics >= target_min:
            if n_topics > target_max:
                print(f"[cluster] menggabungkan {n_topics} -> {target_max} topik")
                topic_model.reduce_topics(docs, nr_topics=target_max)
                topics = topic_model.topics_
            return topic_model, list(topics), n_topics

        if attempt < max_retries and mcs > 10:
            mcs = max(10, mcs // 2)
            print(f"[cluster] di bawah target_min={target_min}; mengulang dengan "
                  f"min_cluster_size lebih kecil={mcs}")
        else:
            print(f"[cluster] PERINGATAN: hanya {n_topics} klaster ditemukan "
                  f"(< target_min={target_min}). Data mungkin tidak memuat "
                  f"{target_min}+ tema yang terpisah. Periksa dulu sebelum "
                  f"memaksakan lebih banyak.")
            return topic_model, list(topics), n_topics
    raise RuntimeError("tidak seharusnya tercapai")


def assign_by_centroid(
    all_emb: np.ndarray,
    fit_idx: np.ndarray,
    topics_fit: np.ndarray,
    assign_outliers: bool,
) -> tuple[np.ndarray, float]:
    """
    Menghitung sentroid topik dari sampel yang dilatih, lalu menetapkan setiap
    teks unik (dokumen di luar sampel, dan outlier bila diminta) ke sentroid
    terdekat berdasarkan kemiripan kosinus.

    Aturan outlier untuk dokumen di luar sampel: dokumen yang kemiripan
    terbaiknya berada di bawah persentil ke-5 dari kemiripan dokumen di dalam
    sampel terhadap sentroidnya *sendiri* diberi label -1 (kecuali jika
    --assign-outliers). Ini menjaga definisi outlier tetap konsisten antara
    sampel yang dilatih dan sisanya.
    """
    n = len(all_emb)
    topic_ids = np.array(sorted(set(topics_fit.tolist()) - {-1}))
    if len(topic_ids) == 0:
        return np.full(n, -1, dtype=np.int32), float("nan")

    emb_fit = np.asarray(all_emb[fit_idx], dtype=np.float32)
    centroids = np.stack(
        [emb_fit[topics_fit == t].mean(axis=0) for t in topic_ids]
    )
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12

    # kemiripan tiap dokumen di dalam sampel terhadap sentroidnya sendiri
    row_of = {t: i for i, t in enumerate(topic_ids)}
    mask = topics_fit != -1
    own_sims = np.einsum(
        "ij,ij->i",
        emb_fit[mask],
        centroids[[row_of[t] for t in topics_fit[mask]]],
    )
    threshold = -1.0 if assign_outliers else float(np.quantile(own_sims, 0.05))
    print(f"[assign] ambang kemiripan sentroid: "
          f"{'dinonaktifkan (menetapkan semua)' if assign_outliers else f'{threshold:.3f}'}")

    labels = np.full(n, -1, dtype=np.int32)
    labels[fit_idx] = topics_fit
    todo = np.setdiff1d(np.arange(n), fit_idx, assume_unique=False)
    if assign_outliers:
        todo = np.union1d(todo, fit_idx[topics_fit == -1])

    for lo in range(0, len(todo), ASSIGN_CHUNK):
        idx = todo[lo:lo + ASSIGN_CHUNK]
        sims = np.asarray(all_emb[idx], dtype=np.float32) @ centroids.T
        best = sims.argmax(axis=1)
        best_sim = sims[np.arange(len(idx)), best]
        lab = topic_ids[best].astype(np.int32)
        lab[best_sim < threshold] = -1
        labels[idx] = lab
        print(f"[assign] {min(lo + ASSIGN_CHUNK, len(todo))}/{len(todo)} ditetapkan")
    return labels, threshold


# ---------------------------------------------------------------------------
# Evaluasi / ekspor
# ---------------------------------------------------------------------------

def compute_silhouette(topic_model, topics: list[int], max_sample: int = 20000) -> float | None:
    """Silhouette pada ruang UMAP 5-dimensi yang benar-benar diklaster HDBSCAN, outlier dikecualikan."""
    try:
        from sklearn.metrics import silhouette_score

        reduced = topic_model.umap_model.embedding_
        labels = np.asarray(topics)
        mask = labels != -1
        if mask.sum() < 100 or len(set(labels[mask].tolist())) < 2:
            return None
        x, y = reduced[mask], labels[mask]
        if len(y) > max_sample:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(y), max_sample, replace=False)
            x, y = x[idx], y[idx]
        return float(silhouette_score(x, y))
    except Exception as exc:  # silhouette hanya diagnostik; jangan menggagalkan run
        print(f"[eval] silhouette gagal: {exc}")
        return None


def export_results(out_dir: Path, df: pd.DataFrame, topic_model, metrics: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "assignments.csv", index=False)

    info = topic_model.get_topic_info().copy()
    keywords, examples = [], []
    for tid in info["Topic"]:
        words = topic_model.get_topic(tid) or []
        keywords.append(", ".join(w for w, _ in words[:10]))
        try:
            reps = topic_model.get_representative_docs(tid) if tid != -1 else []
        except Exception:
            reps = []
        examples.append(" ||| ".join((reps or [])[:3]))
    info["Keywords"] = keywords
    info["Examples"] = examples
    info.to_csv(out_dir / "topics_summary.csv", index=False)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    for name, fn in [
        ("topic_map.html", lambda: topic_model.visualize_topics()),
        ("topic_barchart.html", lambda: topic_model.visualize_barchart(top_n_topics=50)),
    ]:
        try:
            fn().write_html(str(out_dir / name))
        except Exception as exc:
            print(f"[export] {name} dilewati: {exc}")

    topic_model.save(str(out_dir / "bertopic_model"), serialization="safetensors",
                     save_embedding_model=False)
    print(f"[export] hasil ditulis ke {out_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="berkas csv/tsv/parquet/jsonl")
    p.add_argument("--text-col", required=True, help="kolom yang memuat teks")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--model", default="LazarusNLP/all-indo-e5-small-v4",
                   help="id model sentence-transformers (default: model kecil "
                        "yang dilatih untuk bahasa Indonesia, 384-dim, cocok "
                        "untuk jutaan baris). Alternatif lebih besar: "
                        "paraphrase-multilingual-mpnet-base-v2")
    p.add_argument("--e5-prefix", action="store_true",
                   help="tambahkan awalan 'query: ' (untuk model intfloat/multilingual-e5-*)")
    p.add_argument("--fit-sample", type=int, default=300_000,
                   help="latih model topik pada maksimal N teks unik; sisanya "
                        "ditetapkan ke sentroid topik terdekat. 0 = latih pada "
                        "semua (JANGAN lakukan ini pada jutaan baris tanpa cuML)")
    p.add_argument("--target-min", type=int, default=20)
    p.add_argument("--target-max", type=int, default=50)
    p.add_argument("--min-cluster-size", type=int, default=0,
                   help="ukuran klaster minimum HDBSCAN; 0 = heuristik otomatis")
    p.add_argument("--min-content-tokens", type=int, default=2,
                   help="jumlah minimum token non-stopword agar dokumen diklaster")
    p.add_argument("--assign-outliers", action="store_true",
                   help="paksa tetapkan setiap dokumen (termasuk outlier HDBSCAN) "
                        "ke sentroid topik terdekat")
    p.add_argument("--keep-emoji", action="store_true")
    p.add_argument("--no-slang-norm", action="store_true")
    p.add_argument("--sample", type=int, default=0,
                   help="ambil sampel acak N baris input sebelum proses lain "
                        "(0 = pakai semua). Berguna untuk run percontohan")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default=None, help="cuda / cpu (otomatis jika tidak diisi)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # 1. Muat -----------------------------------------------------------------
    df = load_dataframe(Path(args.input))
    if args.text_col not in df.columns:
        sys.exit(f"Kolom '{args.text_col}' tidak ditemukan. Tersedia: {list(df.columns)}")
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=args.seed)
    df = df.reset_index(drop=True)
    n_raw = len(df)
    print(f"[load] {n_raw} baris")

    # 2. Bersihkan ------------------------------------------------------------
    lexicon = {} if args.no_slang_norm else load_slang_lexicon(DEFAULT_LEXICON)
    stopwords = load_stopwords(DEFAULT_STOPWORDS)
    print(f"[clean] entri kamus slang: {len(lexicon)}")
    df["clean_text"] = df[args.text_col].map(
        lambda t: clean_text(t, lexicon=lexicon, keep_emoji=args.keep_emoji)
    )
    df["informative"] = df["clean_text"].map(
        lambda t: is_informative(t, stopwords, args.min_content_tokens)
    )
    n_junk = int((~df["informative"]).sum())
    print(f"[clean] disaring sebagai non-informatif: {n_junk} "
          f"({100 * n_junk / max(n_raw, 1):.1f}%)")

    # 3. Deduplikasi: klaster teks unik, lalu propagasikan label ke semua baris
    work = df[df["informative"]]
    unique_texts = work["clean_text"].drop_duplicates().tolist()
    n_unique = len(unique_texts)
    print(f"[dedup] {len(work)} baris informatif -> {n_unique} teks unik "
          f"({100 * (1 - n_unique / max(len(work), 1)):.1f}% duplikat)")
    if n_unique < 200:
        sys.exit(f"Hanya {n_unique} teks unik yang informatif; terlalu sedikit "
                 f"untuk 20-50 klaster. Kumpulkan lebih banyak data atau "
                 f"longgarkan filter.")

    # 4. Embed (per-chunk, dapat dilanjutkan) ----------------------------------
    embeddings = embed_texts(
        unique_texts, args.model, args.batch_size, args.device,
        cache_dir=out_dir / "cache", e5_prefix=args.e5_prefix,
    )

    # 5. Latih model topik pada sampel representatif ----------------------------
    fit_n = n_unique if args.fit_sample == 0 else min(args.fit_sample, n_unique)
    if fit_n < n_unique:
        fit_idx = np.sort(rng.choice(n_unique, fit_n, replace=False))
        print(f"[cluster] melatih pada sampel {fit_n}/{n_unique} teks unik")
    else:
        fit_idx = np.arange(n_unique)
    docs_fit = [unique_texts[i] for i in fit_idx]
    emb_fit = np.asarray(embeddings[fit_idx], dtype=np.float32)

    mcs = args.min_cluster_size or int(np.clip(fit_n // 1000, 15, 500))
    topic_model, topics_fit_list, natural_k = fit_with_target_range(
        docs_fit, emb_fit, sorted(stopwords),
        args.target_min, args.target_max, mcs, args.seed,
    )
    topics_fit = np.asarray(topics_fit_list, dtype=np.int32)

    # 6. Tetapkan semua teks unik lewat sentroid topik --------------------------
    all_labels, threshold = assign_by_centroid(
        embeddings, fit_idx, topics_fit, args.assign_outliers,
    )
    if args.assign_outliers:
        # perbarui representasi kata kunci dengan label sampel yang sudah diperbarui
        topics_fit = all_labels[fit_idx]
        try:
            topic_model.update_topics(
                docs_fit, topics=topics_fit.tolist(),
                vectorizer_model=topic_model.vectorizer_model,
            )
        except Exception as exc:
            print(f"[cluster] update_topics setelah penetapan outlier gagal "
                  f"(kata kunci memakai statistik sebelum penetapan): {exc}")

    # 7. Propagasikan label kembali ke baris asli -------------------------------
    label_map = dict(zip(unique_texts, all_labels.tolist()))
    info = topic_model.get_topic_info()
    name_map = dict(zip(info["Topic"], info["Name"]))
    df["topic"] = df["clean_text"].map(label_map)
    df["topic"] = df["topic"].where(df["informative"], other=-2)  # -2 = sampah
    df["topic"] = df["topic"].fillna(-2).astype(int)
    df["topic_label"] = df["topic"].map(name_map)
    df.loc[df["topic"] == -2, "topic_label"] = "FILTERED_NON_INFORMATIVE"
    df.loc[df["topic"] == -1, "topic_label"] = "OUTLIER_NO_TOPIC"

    # 8. Evaluasi + ekspor ------------------------------------------------------
    assigned = all_labels[all_labels >= 0]
    sizes = pd.Series(assigned).value_counts()
    sil = compute_silhouette(topic_model, topics_fit_list)
    metrics = {
        "rows_total": n_raw,
        "rows_filtered_non_informative": n_junk,
        "unique_texts": n_unique,
        "fit_sample_size": int(fit_n),
        "natural_clusters_before_merge": natural_k,
        "final_clusters": int(len(sizes)),
        "outlier_pct_of_unique": round(100 * float(np.mean(all_labels == -1)), 2),
        "centroid_similarity_threshold": None if args.assign_outliers
                                          else round(threshold, 4),
        "largest_cluster_share_pct": round(100 * float(sizes.max()) / max(len(assigned), 1), 2)
                                      if len(sizes) else None,
        "silhouette_umap_space_fit_sample": round(sil, 4) if sil is not None else None,
        "model": args.model,
        "min_cluster_size_used": int(mcs),
        "seed": args.seed,
    }
    print(json.dumps(metrics, indent=2))
    export_results(out_dir, df.drop(columns=["informative"]), topic_model, metrics)


if __name__ == "__main__":
    main()