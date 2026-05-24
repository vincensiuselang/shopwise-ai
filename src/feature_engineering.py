# src/feature_engineering.py
# ============================================================
# Tanggung jawab: ubah data bersih → fitur yang bisa dipakai model.
#
# Dua hal utama yang dikerjakan di sini:
#   1. TEXT FEATURES
#      Gabungkan semua kolom teks (title, category, description,
#      review_text, category_source) jadi satu string per produk.
#      String ini yang akan di-TF-IDF oleh recommender_content.
#
#   2. POPULARITY SCORE
#      Hitung skor popularitas dari rating dan jumlah review.
#      Formula: popularity = W_RATING * norm_rating
#                          + W_REVIEW * norm_review_count
#      Di-normalize ke range [0, 1] supaya fair.
#
#   3. PRODUCT LOOKUP
#      Dict { product_id → row detail } untuk O(1) lookup saat predict.
# ============================================================

import pandas as pd
import numpy as np
from typing import Dict, Any

from src.config import W_RATING, W_REVIEW
from src.utils import setup_logger, save_pickle
from src.config import PRODUCT_LOOKUP_PATH

logger = setup_logger("feature_engineering")


# ── Normalization ──────────────────────────────────────────────

def normalize_score(series: pd.Series) -> pd.Series:
    """
    Min-max normalization ke range [0, 1].

    Kenapa min-max bukan standard scaler?
    Karena kita butuh nilai dalam range 0–1 untuk digabung
    dengan bobot di hybrid recommender.
    """
    min_val = series.min()
    max_val = series.max()

    if max_val == min_val:
        # Semua nilai sama → kembalikan semua 0.5
        return pd.Series(0.5, index=series.index)

    return (series - min_val) / (max_val - min_val)


# ── Text features ──────────────────────────────────────────────

def create_product_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gabungkan kolom teks jadi satu kolom 'text_features'.

    Kolom yang digabung (kalau ada):
        title + category + category_source + description + review_text

    Kenapa digabung?
    TF-IDF butuh satu string per dokumen. Makin banyak info teks
    yang digabung, makin kaya representasi produk tersebut.

    Title diberi bobot lebih tinggi dengan cara di-repeat 3x.
    Ini trik sederhana supaya judul produk lebih berpengaruh
    pada similarity tanpa perlu teknik weighting yang kompleks.
    """
    df = df.copy()

    def build_text(row) -> str:
        parts = []

        # Title di-repeat 3x → lebih berpengaruh di TF-IDF
        title = str(row.get("title", "")).strip()
        if title:
            parts.extend([title] * 3)

        # Category dan category_source
        for col in ["category", "category_source"]:
            val = str(row.get(col, "")).strip()
            if val and val not in ("nan", "none", ""):
                parts.append(val)

        # Description
        desc = str(row.get("description", "")).strip()
        if desc and desc not in ("nan", "none", ""):
            parts.append(desc)

        # Review text (kalau ada, ambil 200 karakter pertama)
        rev = str(row.get("review_text", "")).strip()
        if rev and rev not in ("nan", "none", ""):
            parts.append(rev[:200])

        return " ".join(parts)

    df["text_features"] = df.apply(build_text, axis=1)

    empty_count = (df["text_features"].str.strip() == "").sum()
    if empty_count > 0:
        logger.warning(f"⚠️  {empty_count} produk memiliki text_features kosong")

    logger.info(f"✅ text_features dibuat untuk {len(df):,} produk")
    return df


# ── Popularity score ───────────────────────────────────────────

def create_popularity_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung popularity_score dari rating dan review_count.

    Formula:
        popularity_score = W_RATING * norm_rating
                         + W_REVIEW * norm_review_count

    Bobot W_RATING dan W_REVIEW diambil dari config.py.
    Default: masing-masing 0.5 (50/50).

    Kenapa perlu normalize dulu?
    Rating range 0–5, review_count bisa ratusan ribu.
    Kalau langsung dijumlah, review_count akan mendominasi.
    Normalisasi bikin keduanya fair dalam range 0–1.
    """
    df = df.copy()

    # Pastikan kolom ada dan numerik
    if "rating" not in df.columns:
        df["rating"] = 0.0
    if "review_count" not in df.columns:
        df["review_count"] = 0.0

    # Fill NaN dengan 0 sebelum normalisasi
    rating       = df["rating"].fillna(0).astype(float)
    review_count = df["review_count"].fillna(0).astype(float)

    # Normalize
    norm_rating  = normalize_score(rating)
    norm_review  = normalize_score(review_count)

    # Hitung popularity score
    df["popularity_score"] = (
        W_RATING * norm_rating
        + W_REVIEW * norm_review
    ).round(6)

    logger.info(
        f"✅ popularity_score dibuat — "
        f"min: {df['popularity_score'].min():.3f}, "
        f"max: {df['popularity_score'].max():.3f}, "
        f"mean: {df['popularity_score'].mean():.3f}"
    )

    return df


# ── Product lookup ────────────────────────────────────────────

def create_product_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Buat dictionary untuk lookup produk berdasarkan product_id.

    Kenapa perlu lookup dict?
    Saat predict, kita butuh cari detail produk berdasarkan ID.
    Kalau pakai DataFrame.loc tiap kali, lambat untuk dataset besar.
    Dict lookup → O(1).

    Returns:
        { product_id: { title, category, rating, ... } }
    """
    cols_to_keep = [
        "product_id", "title", "category", "category_source",
        "rating", "review_count", "price", "popularity_score",
    ]

    # Hanya ambil kolom yang ada
    available_cols = [c for c in cols_to_keep if c in df.columns]
    subset = df[available_cols].copy()

    lookup = {}
    for _, row in subset.iterrows():
        pid = str(row["product_id"])
        lookup[pid] = row.to_dict()

    logger.info(f"✅ Product lookup dibuat — {len(lookup):,} produk")
    return lookup


# ── Main orchestrator ─────────────────────────────────────────

def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Jalankan semua feature engineering sekaligus.

    Input  : DataFrame hasil preprocess_products()
    Output : DataFrame dengan kolom tambahan:
             - text_features
             - popularity_score

    Artifact yang disimpan otomatis:
             - models/product_lookup.pkl
    """
    logger.info("=" * 55)
    logger.info(f"⚙️  Memulai feature engineering — input: {df.shape}")
    logger.info("=" * 55)

    # 1. Text features
    logger.info("Step 1: Membuat text_features...")
    df = create_product_text_features(df)

    # 2. Popularity score
    logger.info("Step 2: Menghitung popularity_score...")
    df = create_popularity_score(df)

    # 3. Product lookup
    logger.info("Step 3: Membuat product lookup...")
    lookup = create_product_lookup(df)
    save_pickle(lookup, PRODUCT_LOOKUP_PATH)

    logger.info("-" * 55)
    logger.info(f"✅ Feature engineering selesai → output: {df.shape}")
    logger.info(f"   Kolom baru: text_features, popularity_score")
    logger.info("=" * 55)

    return df
