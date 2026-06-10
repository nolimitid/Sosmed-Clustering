"""
Pra-pemrosesan teks media sosial berbahasa Indonesia informal (bahasa gaul /
tidak baku).

Catatan desain
--------------
- Pipeline meng-embed teks dengan model transformer, jadi pembersihan harus
  RINGAN. Normalisasi berat (stemming, penghapusan stopword agresif) merusak
  konteks yang diandalkan model embedding. Kita hanya membuang artefak platform
  (URL, mention, penanda RT) dan menormalkan pola informal yang paling merusak
  (perpanjangan karakter, slang yang umum).
- Normalisasi slang memakai Colloquial Indonesian Lexicon
  (Salsabila dkk., 2018, "kamus alay"), disertakan di data/slang_lexicon.csv.
- Dua keluaran per dokumen:
    * `clean`  : dibersihkan ringan, slang dinormalkan -> diberikan ke embedder
    * `tokens` : dipakai hanya untuk filter keinformatifan
"""

from __future__ import annotations

import csv
import html
import re
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------------

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
RT_RE = re.compile(r"^\s*rt\s*:?\s*(?=@|\w)", re.IGNORECASE)
# singkatan reduplikasi: jalan2 -> jalan jalan, kelar2 -> kelar kelar
REDUP_RE = re.compile(r"\b([a-z]{3,})2\b")
HASHTAG_RE = re.compile(r"#(\w+)")
# wkwk / haha / hehe / xixi / awokwok / kwkw dan sejenisnya, panjang >= 4
LAUGH_RE = re.compile(
    r"\b(?:a*(?:wk|kw){2,}a*|(?:ha){2,}h?|(?:he){2,}h?|(?:hi){2,}h?|"
    r"(?:xi){2,}x?|(?:wa){2,}k*|w{3,})\b",
    re.IGNORECASE,
)
# 3+ pengulangan karakter yang sama -> 1 (bangettt -> banget, mantaaap -> mantap)
ELONG_RE = re.compile(r"(\w)\1{2,}")
MULTISPACE_RE = re.compile(r"\s+")
EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001faff"
    "\U00002600-\U000027bf"
    "\U0001f1e6-\U0001f1ff"
    "\U00002700-\U000027bf"
    "\ufe0f"
    "]+",
    flags=re.UNICODE,
)
NON_WORD_RE = re.compile(r"[^\w\s]")


# ---------------------------------------------------------------------------
# Kamus slang
# ---------------------------------------------------------------------------

def load_slang_lexicon(path: str | Path) -> dict[str, str]:
    """Memuat pemetaan slang -> baku (CSV dengan kolom: slang, formal)."""
    lex: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            slang = row["slang"].strip().lower()
            formal = row["formal"].strip().lower()
            if slang and formal and slang != formal:
                lex[slang] = formal
    return lex


def normalize_slang(text: str, lexicon: dict[str, str]) -> str:
    """Penggantian slang per token. Ekspansi multi-kata diperbolehkan."""
    if not lexicon:
        return text
    out = []
    for tok in text.split():
        out.append(lexicon.get(tok, tok))
    return " ".join(out)


# ---------------------------------------------------------------------------
# Pembersihan
# ---------------------------------------------------------------------------

def clean_text(
    text: str,
    lexicon: dict[str, str] | None = None,
    keep_emoji: bool = False,
) -> str:
    """Membersihkan satu dokumen secara ringan untuk embedding."""
    if not isinstance(text, str):
        return ""
    t = html.unescape(text)
    t = unicodedata.normalize("NFKC", t)
    t = t.lower()
    t = RT_RE.sub("", t)
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = HASHTAG_RE.sub(r"\1", t)          # simpan kata tagar, buang '#'
    if not keep_emoji:
        t = EMOJI_RE.sub(" ", t)
    t = LAUGH_RE.sub(" wkwk ", t)          # ringkas tawa jadi satu token
    t = ELONG_RE.sub(r"\1", t)             # bangettt -> banget
    t = NON_WORD_RE.sub(" ", t)
    t = REDUP_RE.sub(r"\1 \1", t)          # kelar2 -> kelar kelar
    t = MULTISPACE_RE.sub(" ", t).strip()
    if lexicon:
        t = normalize_slang(t, lexicon)
    return t


def load_stopwords(path: str | Path) -> set[str]:
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def is_informative(
    clean: str,
    stopwords: set[str],
    min_content_tokens: int = 2,
    min_chars: int = 5,
) -> bool:
    """
    Menyaring dokumen yang tidak membawa sinyal topik:
    kosong setelah dibersihkan, terlalu pendek, atau hanya berisi
    stopword/tawa ("wkwkwk", "mantap bang", "iya kak", ...).
    """
    if len(clean) < min_chars:
        return False
    toks = clean.split()
    content = [t for t in toks if t not in stopwords and len(t) > 1]
    return len(content) >= min_content_tokens