"""
Pipeline klasterisasi untuk teks media sosial berbahasa Indonesia informal,
dirancang agar mampu menangani jutaan baris.

Pipeline:
  muat -> bersihkan/normalkan -> deteksi & saring bahasa (buang non-Indonesia)
       -> saring sampah -> deduplikasi
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

from language_id import LanguageDetector, lang_name
from preprocess import clean_text, is_informative, load_slang_lexicon, load_stopwords

HERE = Path(__file__).resolve().parent
DEFAULT_LEXICON = HERE / "slang_lexicon.csv"
DEFAULT_STOPWORDS = HERE / "stopwords_id.txt"

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
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        import json as _json
        with open(path, encoding="utf-8") as _f:
            raw = _json.load(_f)
        if isinstance(raw, list):
            return pd.DataFrame(raw)
        if isinstance(raw, dict):
            # cari key pertama yang berisi list of dicts (misal: "posts", "data", "results")
            for _, _val in raw.items():
                if isinstance(_val, list) and _val and isinstance(_val[0], dict):
                    return pd.DataFrame(_val)
            return pd.DataFrame([raw])
        raise ValueError(f"Struktur JSON tidak dikenali di {path}")
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
# Penamaan topik berbasis sentroid
# ---------------------------------------------------------------------------

def get_top_docs_by_centroid(
    unique_texts: list[str],
    embeddings: np.ndarray,
    labels: np.ndarray,
    top_n: int = 5,
) -> dict[int, list[str]]:
    """
    Untuk setiap topik, kembalikan top_n teks dengan kemiripan kosinus
    tertinggi ke sentroid topik di ruang embedding.
    """
    topic_ids = sorted(set(labels.tolist()) - {-1, -2})
    emb_f32 = np.asarray(embeddings, dtype=np.float32)
    result: dict[int, list[str]] = {}
    for tid in topic_ids:
        mask = labels == tid
        if not mask.any():
            continue
        centroid = emb_f32[mask].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-12
        sims = emb_f32[mask] @ centroid
        global_idx = np.where(mask)[0]
        top_local = np.argsort(sims)[::-1][:top_n]
        result[tid] = [unique_texts[global_idx[i]] for i in top_local]
    return result


# ---------------------------------------------------------------------------
# Graph analisis topik & meta-topik (Leiden community detection)
# ---------------------------------------------------------------------------

def build_topic_graph(
    embeddings: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.25,
    soft_threshold: float = 0.40,
) -> tuple[list[int], dict[tuple[int, int], dict]]:
    """
    Bangun graph topik berdasarkan CO-OCCURRENCE: berapa banyak dokumen yang
    secara semantik dekat dengan KEDUA sentroid topik A dan B sekaligus
    (similarity >= soft_threshold ke keduanya). Edge hanya dibuat bila
    cooccurrence_rate >= threshold.

    Edge menyimpan dua metrik:
      - cooccurrence_rate : fraksi dokumen yang menjembatani dua topik
      - centroid_similarity: kemiripan langsung antar sentroid (sebagai info tambahan)
    """
    topic_ids = sorted(set(labels.tolist()) - {-1, -2})
    if len(topic_ids) < 2:
        return topic_ids, {}

    emb_f32 = np.asarray(embeddings, dtype=np.float32)
    centroids = np.stack([emb_f32[labels == t].mean(axis=0) for t in topic_ids])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12

    # similarity setiap dokumen ke semua sentroid: shape (n_docs, n_topics)
    doc_sim = emb_f32 @ centroids.T          # (n_docs, n_topics)
    above = doc_sim >= soft_threshold        # bool mask
    n_docs = len(emb_f32)

    # centroid-to-centroid similarity (untuk info)
    centroid_sim = centroids @ centroids.T
    np.fill_diagonal(centroid_sim, 0)

    edges: dict[tuple[int, int], dict] = {}
    for i, ti in enumerate(topic_ids):
        for j, tj in enumerate(topic_ids):
            if i >= j:
                continue
            cooc = int(np.logical_and(above[:, i], above[:, j]).sum())
            rate = cooc / max(n_docs, 1)
            if rate >= threshold:
                edges[(ti, tj)] = {
                    "cooccurrence_count": cooc,
                    "cooccurrence_rate": round(rate, 6),
                    "centroid_similarity": round(float(centroid_sim[i, j]), 4),
                }
    print(f"[graph] {len(topic_ids)} topik, {len(edges)} edge "
          f"(co-occurrence threshold={threshold}, soft_threshold={soft_threshold})")
    return topic_ids, edges


def detect_meta_topics(
    topic_ids: list[int],
    edges: dict[tuple[int, int], float],
) -> dict[int, int]:
    """
    Jalankan Leiden algorithm pada graph topik untuk mengelompokkan topik
    ke meta-topik. Returns {topic_id: meta_topic_id} (0-indexed).
    """
    import igraph as ig
    import leidenalg

    if not edges:
        print("[graph] tidak ada edge — setiap topik menjadi meta-topiknya sendiri")
        return {t: i for i, t in enumerate(topic_ids)}

    idx_of = {t: i for i, t in enumerate(topic_ids)}
    g = ig.Graph()
    g.add_vertices(len(topic_ids))
    g.add_edges([(idx_of[ti], idx_of[tj]) for ti, tj in edges])
    g.es["weight"] = [v["cooccurrence_rate"] for v in edges.values()]

    partition = leidenalg.find_partition(
        g,
        leidenalg.ModularityVertexPartition,
        weights="weight",
        seed=42,
    )
    meta = {topic_ids[node]: mid
            for mid, community in enumerate(partition)
            for node in community}
    n_meta = len(set(meta.values()))
    print(f"[graph] Leiden menemukan {n_meta} meta-topik dari {len(topic_ids)} topik")
    return meta


def visualize_topic_graph(
    topic_ids: list[int],
    edges: dict[tuple[int, int], float],
    meta_map: dict[int, int],
    topic_info: pd.DataFrame,
    out_path: Path,
) -> None:
    """Simpan visualisasi interaktif graph hubungan antar topik (HTML)."""
    try:
        import networkx as nx
        import plotly.graph_objects as go
    except ImportError:
        print("[graph] networkx/plotly tidak tersedia, skip visualisasi graph")
        return

    G = nx.Graph()
    size_map = dict(zip(topic_info["Topic"], topic_info["Count"]))
    name_map = dict(zip(topic_info["Topic"], topic_info["Name"]))
    for tid in topic_ids:
        G.add_node(tid)
    for (ti, tj), data in edges.items():
        G.add_edge(ti, tj, **data)

    pos = nx.spring_layout(G, seed=42, weight="weight",
                           k=2.5 / max(len(topic_ids) ** 0.5, 1))

    edge_traces = []
    for ti, tj, data in G.edges(data=True):
        x0, y0 = pos[ti]
        x1, y1 = pos[tj]
        rate = data.get("cooccurrence_rate", 0.01)
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=max(0.5, rate * 200), color="rgba(150,150,150,0.4)"),
            hovertext=f"T{ti}↔T{tj} | co-occurrence: {rate:.2%} | "
                      f"centroid sim: {data.get('centroid_similarity', 0):.2f}",
            hoverinfo="text", showlegend=False,
        ))

    xs = [pos[t][0] for t in topic_ids]
    ys = [pos[t][1] for t in topic_ids]
    node_trace = go.Scatter(
        x=xs, y=ys,
        mode="markers+text",
        text=[str(t) for t in topic_ids],
        textposition="top center",
        hovertext=[
            f"<b>Topik {t}</b><br>{name_map.get(t, '')}"
            f"<br>Meta-topik: {meta_map.get(t, '?')}"
            f"<br>Jumlah post: {size_map.get(t, 0)}"
            for t in topic_ids
        ],
        hoverinfo="text",
        marker=dict(
            size=[max(12, (size_map.get(t, 10) ** 0.5) * 2.5) for t in topic_ids],
            color=[meta_map.get(t, 0) for t in topic_ids],
            colorscale="Turbo",
            showscale=True,
            colorbar=dict(title="Meta-topik"),
            line=dict(width=1.5, color="white"),
        ),
    )

    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title="Graf Hubungan Antar Topik — warna = meta-topik (Leiden)",
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=750,
            margin=dict(l=20, r=20, t=50, b=20),
        ),
    )
    fig.write_html(str(out_path))
    print(f"[graph] graf topik disimpan ke {out_path}")


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


def export_results(
    out_dir: Path,
    df: pd.DataFrame,
    topic_model,
    metrics: dict,
    centroid_docs_map: dict[int, list[str]] | None = None,
    meta_map: dict[int, int] | None = None,
    graph_edges: dict[tuple[int, int], float] | None = None,
) -> None:
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
    if centroid_docs_map:
        info["Top5CentroidDocs"] = info["Topic"].map(
            lambda tid: " ||| ".join(centroid_docs_map.get(tid, []))
        )
    if meta_map:
        info["meta_topic"] = info["Topic"].map(meta_map)
    info.to_csv(out_dir / "topics_summary.csv", index=False)

    if graph_edges:
        pd.DataFrame(
            [{"topic_a": ti, "topic_b": tj, **data}
             for (ti, tj), data in graph_edges.items()]
        ).to_csv(out_dir / "topic_graph_edges.csv", index=False)

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

    if meta_map is not None and graph_edges is not None:
        visualize_topic_graph(
            [t for t in meta_map if t >= 0],
            graph_edges, meta_map, info, out_dir / "topic_graph.html",
        )

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
    p.add_argument("--no-lang-filter", action="store_true",
                   help="lewati deteksi & penyaringan bahasa (proses semua bahasa)")
    p.add_argument("--keep-langs", default="id,ms,jv,su,min,ban,ace",
                   help="kode bahasa (ISO 639) yang dipertahankan, dipisah koma; "
                        "baris yang DIYAKINI berbahasa lain dibuang. Default "
                        "mencakup Indonesia + Melayu + bahasa daerah karena "
                        "deteksi sering mengira teks Indonesia informal sebagai "
                        "Melayu/daerah. Pakai 'id' saja untuk penyaringan ketat")
    p.add_argument("--lang-backend", choices=["fasttext", "langdetect"],
                   default="fasttext",
                   help="backend deteksi bahasa yang diutamakan (jatuh ke yang "
                        "lain bila tak terpasang)")
    p.add_argument("--lang-model", default=None,
                   help="path model fastText lid.176 (.ftz/.bin); diunduh "
                        "otomatis ke ~/.cache bila kosong")
    p.add_argument("--lang-min-conf", type=float, default=0.5,
                   help="hanya buang baris berbahasa asing bila kepercayaan "
                        "deteksi >= nilai ini; baris berkeyakinan rendah (mis. "
                        "teks sangat pendek/campur kode) dipertahankan")
    p.add_argument("--sample", type=int, default=0,
                   help="ambil sampel acak N baris input sebelum proses lain "
                        "(0 = pakai semua). Berguna untuk run percontohan")
    p.add_argument("--graph-threshold", type=float, default=0.25,
                   help="ambang kemiripan kosinus antar sentroid topik untuk membuat "
                        "edge di graph (default: 0.25). Turunkan untuk lebih banyak "
                        "koneksi, naikkan untuk hanya hubungan yang kuat.")
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

    # 2b. Deteksi & saring bahasa: buang baris yang diyakini bukan Indonesia ----
    # Deteksi dilakukan pada teks unik (lalu dipetakan ke semua baris) agar hemat
    # pada korpus jutaan baris yang banyak duplikatnya.
    detector = None
    keep_langs: set[str] = set()
    n_dropped_lang = 0
    if not args.no_lang_filter:
        keep_langs = {c.strip().lower() for c in args.keep_langs.split(",") if c.strip()}
        detector = LanguageDetector.load(model_path=args.lang_model,
                                         prefer=args.lang_backend)
        print(f"[lang] backend: {detector.name}; dipertahankan: {sorted(keep_langs)}")
        uniq_clean = df["clean_text"].drop_duplicates().tolist()
        codes, confs = detector.detect_batch(uniq_clean)
        code_map = dict(zip(uniq_clean, codes))
        conf_map = dict(zip(uniq_clean, confs))
        df["lang"] = df["clean_text"].map(code_map).fillna("und")
        df["lang_conf"] = df["clean_text"].map(conf_map).fillna(0.0).round(3)

        # Buang baris hanya jika DIYAKINI berbahasa asing: bahasa di luar keep_langs
        # DAN kepercayaan >= ambang. Baris berkeyakinan rendah (teks sangat pendek /
        # campur kode) dipertahankan agar teks Indonesia tidak salah buang.
        foreign = (~df["lang"].isin(keep_langs)) & (df["lang_conf"] >= args.lang_min_conf)
        n_dropped_lang = int(foreign.sum())
        if n_dropped_lang and detector.available:
            drop_counts = df.loc[foreign, "lang"].value_counts()
            print(f"[lang] membuang {n_dropped_lang} baris non-Indonesia "
                  f"({100 * n_dropped_lang / max(n_raw, 1):.1f}%):")
            for code, cnt in drop_counts.head(15).items():
                print(f"        {code:>5} {lang_name(code):<14} {cnt:>10}")
            pd.DataFrame({
                "lang": drop_counts.index,
                "language": [lang_name(c) for c in drop_counts.index],
                "rows_dropped": drop_counts.values,
                "pct_of_input": (100 * drop_counts.values / max(n_raw, 1)).round(2),
            }).to_csv(out_dir / "dropped_languages.csv", index=False)
            print(f"[lang] ringkasan bahasa yang dibuang -> "
                  f"{out_dir / 'dropped_languages.csv'}")
        df = df[~foreign].reset_index(drop=True)
        print(f"[lang] {len(df)}/{n_raw} baris dipertahankan setelah saring bahasa")

    # 2c. Saring sampah (non-informatif) ---------------------------------------
    df["informative"] = df["clean_text"].map(
        lambda t: is_informative(t, stopwords, args.min_content_tokens)
    )
    n_junk = int((~df["informative"]).sum())
    print(f"[clean] disaring sebagai non-informatif: {n_junk} "
          f"({100 * n_junk / max(len(df), 1):.1f}%)")

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

    centroid_docs_map = get_top_docs_by_centroid(unique_texts, embeddings, all_labels)

    # 7b. Graph analisis & meta-topik ------------------------------------------
    topic_ids_graph, graph_edges = build_topic_graph(
        embeddings, all_labels, threshold=args.graph_threshold,
    )
    meta_map = detect_meta_topics(topic_ids_graph, graph_edges)

    df["topic"] = df["clean_text"].map(label_map)
    df["topic"] = df["topic"].where(df["informative"], other=-2)  # -2 = sampah
    df["topic"] = df["topic"].fillna(-2).astype(int)
    df["topic_label"] = df["topic"].map(name_map)
    df.loc[df["topic"] == -2, "topic_label"] = "FILTERED_NON_INFORMATIVE"
    df.loc[df["topic"] == -1, "topic_label"] = "OUTLIER_NO_TOPIC"
    df["meta_topic"] = df["topic"].map(meta_map)
    df.loc[df["topic"] < 0, "meta_topic"] = df.loc[df["topic"] < 0, "topic"]
    df["meta_topic"] = df["meta_topic"].fillna(-1).astype(int)

    # 8. Evaluasi + ekspor ------------------------------------------------------
    assigned = all_labels[all_labels >= 0]
    sizes = pd.Series(assigned).value_counts()
    sil = compute_silhouette(topic_model, topics_fit_list)
    metrics = {
        "rows_total": n_raw,
        "lang_filter_backend": detector.name if detector else "disabled",
        "lang_kept": sorted(keep_langs) if keep_langs else None,
        "rows_dropped_non_indonesian": int(n_dropped_lang),
        "rows_after_language_filter": int(len(df)),
        "kept_language_distribution_pct": (
            {c: round(100 * n / max(len(df), 1), 2)
             for c, n in df["lang"].value_counts().head(10).items()}
            if "lang" in df.columns else None
        ),
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
    export_results(out_dir, df.drop(columns=["informative"]), topic_model, metrics,
                   centroid_docs_map, meta_map, graph_edges)


if __name__ == "__main__":
    main()