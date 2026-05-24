# src/column_mapper.py
# ============================================================
# Masalah utama: 5 dataset Amazon bisa punya nama kolom berbeda.
# Contoh: satu dataset punya "product_name", yang lain "title",
# yang lain "name". Kalau kita hardcode nama kolom, kode crash.
#
# Solusi di sini: fuzzy matching nama kolom + fallback ke None.
# Kolom yang tidak ketemu diisi NaN, bukan error.
# ============================================================

import re
import warnings
import pandas as pd
from typing import Optional

from src.utils import setup_logger

logger = setup_logger("column_mapper")


# ── Keyword mapping ───────────────────────────────────────────
# Daftar keyword yang mungkin ada di nama kolom asli dataset.
# Urutan = prioritas (yang pertama paling diprioritaskan).

TITLE_KEYWORDS       = ["product_name", "title", "name", "product_title", "item_name", "products"]
RATING_KEYWORDS      = ["rating", "average_rating", "avg_rating", "stars", "star_rating", "rate"]
REVIEW_COUNT_KEYWORDS = ["review_count", "no_of_ratings", "number_of_reviews", "ratings_count",
                          "num_reviews", "total_reviews", "count_reviews", "reviewcount", "reviews"]
PRICE_KEYWORDS       = ["price", "actual_price", "discounted_price", "selling_price", "cost"]
DESCRIPTION_KEYWORDS = ["description", "about_product", "product_description", "detail",
                         "product_details", "info"]
REVIEW_TEXT_KEYWORDS = ["review_text", "review_content", "reviews", "reviewtext",
                         "summary", "review_summary", "review_body"]
PRODUCT_ID_KEYWORDS  = ["product_id", "asin", "id", "item_id", "sku", "product_link"]
CATEGORY_KEYWORDS    = ["category", "main_category", "sub_category", "category_name",
                         "product_category", "type"]


def _find_column(df: pd.DataFrame, keywords: list[str]) -> Optional[str]:
    """
    Cari nama kolom DataFrame yang paling cocok dengan daftar keyword.
    Matching case-insensitive dan ignore karakter spesial.

    Returns:
        Nama kolom yang cocok, atau None kalau tidak ketemu.
    """
    cols_lower = {col.lower().strip(): col for col in df.columns}

    for kw in keywords:
        # Exact match dulu (paling aman)
        if kw.lower() in cols_lower:
            return cols_lower[kw.lower()]

    # Kalau tidak exact, coba substring match
    for kw in keywords:
        for col_lower, col_orig in cols_lower.items():
            if kw.lower() in col_lower:
                return col_orig

    return None


def inspect_columns(df: pd.DataFrame) -> None:
    """
    Tampilkan informasi kolom dataset secara detail.
    Berguna untuk debugging waktu pertama kali cek dataset baru.
    """
    print("\n" + "=" * 60)
    print(f"📋 INSPEKSI KOLOM — Total: {len(df.columns)} kolom, {len(df):,} baris")
    print("=" * 60)

    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_pct   = 100 * null_count / len(df)
        dtype      = df[col].dtype
        sample     = df[col].dropna().iloc[0] if not df[col].dropna().empty else "—"
        # Potong sample kalau terlalu panjang
        sample_str = str(sample)[:60] + "..." if len(str(sample)) > 60 else str(sample)

        print(
            f"  {col:<35} | {str(dtype):<10} | "
            f"null: {null_count:>6} ({null_pct:.1f}%) | "
            f"sample: {sample_str}"
        )
    print("=" * 60 + "\n")


def detect_title_column(df: pd.DataFrame) -> Optional[str]:
    """Deteksi kolom yang berisi judul/nama produk."""
    return _find_column(df, TITLE_KEYWORDS)


def detect_rating_column(df: pd.DataFrame) -> Optional[str]:
    """Deteksi kolom yang berisi rating produk."""
    return _find_column(df, RATING_KEYWORDS)


def detect_review_count_column(df: pd.DataFrame) -> Optional[str]:
    """Deteksi kolom yang berisi jumlah review."""
    return _find_column(df, REVIEW_COUNT_KEYWORDS)


def detect_price_column(df: pd.DataFrame) -> Optional[str]:
    """Deteksi kolom yang berisi harga produk."""
    return _find_column(df, PRICE_KEYWORDS)


def detect_description_column(df: pd.DataFrame) -> Optional[str]:
    """Deteksi kolom yang berisi deskripsi produk."""
    return _find_column(df, DESCRIPTION_KEYWORDS)


def detect_review_text_column(df: pd.DataFrame) -> Optional[str]:
    """Deteksi kolom yang berisi teks review."""
    return _find_column(df, REVIEW_TEXT_KEYWORDS)


def detect_product_id_column(df: pd.DataFrame) -> Optional[str]:
    """Deteksi kolom yang berisi ID produk."""
    return _find_column(df, PRODUCT_ID_KEYWORDS)


def detect_category_column(df: pd.DataFrame) -> Optional[str]:
    """Deteksi kolom yang berisi kategori produk."""
    return _find_column(df, CATEGORY_KEYWORDS)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ubah DataFrame dengan nama kolom sembarang menjadi DataFrame
    dengan kolom standar yang dipakai seluruh pipeline.

    Kolom standar:
        product_id, title, category, rating, review_count,
        price, description, review_text, category_source

    Kalau kolom tidak ketemu → diisi NaN (tidak crash).
    category_source dipertahankan dari hasil load_single_csv().
    """
    mapping_results = {}

    # ── Deteksi setiap kolom ──────────────────────────────────
    detectors = {
        "product_id":    detect_product_id_column(df),
        "title":         detect_title_column(df),
        "category":      detect_category_column(df),
        "rating":        detect_rating_column(df),
        "review_count":  detect_review_count_column(df),
        "price":         detect_price_column(df),
        "description":   detect_description_column(df),
        "review_text":   detect_review_text_column(df),
    }

    logger.info("🔍 Hasil deteksi kolom:")
    for std_col, found_col in detectors.items():
        status = f"→ '{found_col}'" if found_col else "→ ⚠️  TIDAK DITEMUKAN (akan diisi NaN)"
        logger.info(f"   {std_col:<15} {status}")
        mapping_results[std_col] = found_col

    # ── Bangun DataFrame standar ──────────────────────────────
    result = pd.DataFrame()

    for std_col, src_col in mapping_results.items():
        if src_col and src_col in df.columns:
            result[std_col] = df[src_col].values
        else:
            result[std_col] = None  # akan jadi NaN

    # Pertahankan category_source (sudah ditambahkan oleh data_loader)
    if "category_source" in df.columns:
        result["category_source"] = df["category_source"].values
    else:
        result["category_source"] = "unknown"

    # Buat product_id dari index kalau tidak ditemukan
    if result["product_id"].isnull().all():
        logger.warning(
            "⚠️  Kolom product_id tidak ditemukan. "
            "Membuat product_id dari index + category_source."
        )
        result["product_id"] = (
            result["category_source"].astype(str)
            + "_"
            + pd.RangeIndex(len(result)).astype(str)
        )

    logger.info(
        f"✅ Standardisasi selesai → {result.shape[0]:,} baris, "
        f"{result.shape[1]} kolom standar"
    )

    return result
