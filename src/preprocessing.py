# src/preprocessing.py
# ============================================================
# Tanggung jawab: bersihin data mentah supaya siap dipakai
# untuk feature engineering dan modeling.
#
# Urutan yang benar:
#   1. clean_text       → normalisasi string
#   2. clean_rating     → pastikan rating numerik dan valid
#   3. clean_price      → ekstrak angka dari string harga
#   4. clean_review_count → pastikan jumlah review numerik
#   5. handle_missing_values → isi / drop NaN
#   6. remove_duplicates    → buang produk duplikat
#   7. filter_valid_products → buang produk yang terlalu jelek
#   8. preprocess_products  → orchestrator semua langkah di atas
# ============================================================

import re
import pandas as pd
import numpy as np
from typing import Optional

from src.config import MIN_RATING, MIN_TITLE_LEN
from src.utils import setup_logger

logger = setup_logger("preprocessing")


# ── Text cleaning ─────────────────────────────────────────────

def clean_text(text) -> str:
    """
    Normalisasi teks produk supaya siap untuk TF-IDF.

    Yang dilakukan:
    - Lowercase
    - Hapus HTML tag
    - Hapus karakter non-alphanumeric kecuali spasi
    - Collapse whitespace berlebih
    - Strip leading/trailing whitespace
    """
    if pd.isna(text) or text is None:
        return ""

    text = str(text)
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)          # hapus HTML tag
    text = re.sub(r"[^a-z0-9\s]", " ", text)      # hapus karakter aneh
    text = re.sub(r"\s+", " ", text)               # collapse whitespace
    return text.strip()


# ── Numeric cleaning ──────────────────────────────────────────

def clean_rating(value) -> Optional[float]:
    """
    Ubah nilai rating ke float.

    Handle kasus:
    - Sudah float/int → langsung
    - String seperti '4.5 out of 5' → ambil angka pertama
    - String seperti '4,5' → ubah koma ke titik
    - Tidak bisa diparse → return NaN
    """
    if pd.isna(value) or value is None:
        return np.nan

    text = str(value).strip()

    # Ambil angka pertama yang ketemu (termasuk desimal)
    match = re.search(r"\d+[.,]?\d*", text)
    if not match:
        return np.nan

    try:
        return float(match.group().replace(",", "."))
    except ValueError:
        return np.nan


def clean_price(value) -> Optional[float]:
    """
    Ekstrak angka dari string harga.

    Handle kasus:
    - '$12.99' → 12.99
    - '₹1,299' → 1299.0
    - 'Rp 12.000.000' → 12000000.0
    - Sudah float → langsung
    - Tidak ada angka → NaN
    """
    if pd.isna(value) or value is None:
        return np.nan

    text = str(value).strip()

    # Hapus simbol mata uang dan spasi
    text = re.sub(r"[^\d.,]", "", text)

    # Jika ada koma sebagai ribuan (12,000) → hapus koma
    # Jika ada titik sebagai ribuan (12.000) → hapus titik
    # Heuristik: kalau ada lebih dari satu koma/titik, asumsi pemisah ribuan
    dot_count   = text.count(".")
    comma_count = text.count(",")

    if comma_count > 1:
        text = text.replace(",", "")
    elif dot_count > 1:
        text = text.replace(".", "")
    elif comma_count == 1 and dot_count == 0:
        # Bisa jadi desimal (12,99) atau ribuan (12,000)
        # Kalau ada 3 digit setelah koma → ribuan
        after_comma = text.split(",")[-1]
        if len(after_comma) == 3:
            text = text.replace(",", "")
        else:
            text = text.replace(",", ".")

    try:
        return float(text)
    except ValueError:
        return np.nan


def clean_review_count(value) -> Optional[float]:
    """
    Ekstrak jumlah review sebagai float.

    Handle kasus:
    - '1,234' → 1234.0
    - '2.3K' → 2300.0
    - '10K+' → 10000.0
    - Sudah int/float → langsung
    """
    if pd.isna(value) or value is None:
        return np.nan

    text = str(value).strip().lower().replace(",", "")

    # Handle '2.3k', '10k+'
    k_match = re.search(r"([\d.]+)\s*k\+?", text)
    if k_match:
        try:
            return float(k_match.group(1)) * 1000
        except ValueError:
            pass

    match = re.search(r"[\d.]+", text)
    if not match:
        return np.nan

    try:
        return float(match.group())
    except ValueError:
        return np.nan


# ── DataFrame-level cleaning ──────────────────────────────────

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategi handling missing value per kolom:
    - title         : drop baris (wajib ada, kalau tidak ada tidak berguna)
    - rating        : isi dengan median rating per category_source
    - review_count  : isi dengan 0
    - price         : isi dengan NaN (tidak wajib)
    - description   : isi dengan string kosong
    - review_text   : isi dengan string kosong
    - category      : isi dengan category_source
    """
    df = df.copy()
    original_len = len(df)

    # Title wajib ada
    df = df.dropna(subset=["title"])
    df = df[df["title"].str.strip().str.len() >= MIN_TITLE_LEN]
    dropped = original_len - len(df)
    if dropped > 0:
        logger.info(f"   Dropped {dropped:,} baris karena title kosong/terlalu pendek")

    # Rating: isi dengan median per kategori
    if "rating" in df.columns:
        median_rating = df.groupby("category_source")["rating"].transform("median")
        global_median = df["rating"].median()
        df["rating"] = df["rating"].fillna(median_rating).fillna(global_median)

    # Review count: isi 0
    if "review_count" in df.columns:
        df["review_count"] = df["review_count"].fillna(0)

    # Text columns: isi string kosong
    for col in ["description", "review_text"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # Category: fallback ke category_source
    if "category" in df.columns:
        df["category"] = df["category"].fillna(df["category_source"])

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hapus produk duplikat.

    Strategi: cek kombinasi title + category_source.
    Bukan product_id karena product_id bisa di-generate (tidak unik aslinya).
    """
    original_len = len(df)

    df = df.drop_duplicates(subset=["title", "category_source"], keep="first")

    removed = original_len - len(df)
    if removed > 0:
        logger.info(f"   Removed {removed:,} duplikat (title + category_source)")

    return df


def filter_valid_products(df: pd.DataFrame) -> pd.DataFrame:
    """
    Buang produk yang jelas-jelas tidak valid:
    - Rating di luar range 0–5
    - Rating di bawah MIN_RATING (dari config)
    """
    original_len = len(df)

    if "rating" in df.columns:
        df = df[df["rating"].between(MIN_RATING, 5.0, inclusive="both")]

    removed = original_len - len(df)
    if removed > 0:
        logger.info(f"   Filtered {removed:,} produk dengan rating tidak valid")

    return df


# ── Main orchestrator ─────────────────────────────────────────

def preprocess_products(df: pd.DataFrame) -> pd.DataFrame:
    """
    Jalankan semua langkah preprocessing secara berurutan.

    Input  : DataFrame hasil standardize_columns() dari column_mapper
    Output : DataFrame bersih, siap untuk feature_engineering

    Langkah:
        1. Bersihkan kolom numerik (rating, price, review_count)
        2. Bersihkan kolom teks (title, description, review_text)
        3. Handle missing values
        4. Buang duplikat
        5. Filter produk tidak valid
    """
    logger.info("=" * 55)
    logger.info(f"🧹 Memulai preprocessing — input: {df.shape}")
    logger.info("=" * 55)

    df = df.copy()

    # ── Step 1: Clean numerics ────────────────────────────────
    logger.info("Step 1: Membersihkan kolom numerik...")

    if "rating" in df.columns:
        df["rating"] = df["rating"].apply(clean_rating).astype(float)

    if "price" in df.columns:
        df["price"] = df["price"].apply(clean_price).astype(float)

    if "review_count" in df.columns:
        df["review_count"] = df["review_count"].apply(clean_review_count).astype(float)

    # ── Step 2: Clean text columns ────────────────────────────
    logger.info("Step 2: Membersihkan kolom teks...")

    df["title"] = df["title"].apply(clean_text)

    if "description" in df.columns:
        df["description"] = df["description"].apply(clean_text)

    if "review_text" in df.columns:
        df["review_text"] = df["review_text"].apply(clean_text)

    if "category" in df.columns:
        df["category"] = df["category"].apply(
            lambda x: str(x).lower().strip() if pd.notna(x) else x
        )

    # ── Step 3: Handle missing values ────────────────────────
    logger.info("Step 3: Handle missing values...")
    df = handle_missing_values(df)

    # ── Step 4: Remove duplicates ────────────────────────────
    logger.info("Step 4: Hapus duplikat...")
    df = remove_duplicates(df)

    # ── Step 5: Filter invalid products ──────────────────────
    logger.info("Step 5: Filter produk tidak valid...")
    df = filter_valid_products(df)

    # ── Step 6: Reset index ───────────────────────────────────
    df = df.reset_index(drop=True)

    # Pastikan product_id unik setelah semua proses
    df["product_id"] = (
        df["category_source"].astype(str)
        + "_"
        + df.index.astype(str)
    )

    logger.info("-" * 55)
    logger.info(f"✅ Preprocessing selesai → output: {df.shape}")
    logger.info(
        f"   Distribusi kategori:\n"
        + df["category_source"].value_counts().to_string()
    )
    logger.info("=" * 55)

    return df
