# src/config.py
# ============================================================
# Semua path, parameter, dan konstanta ada di sini.
# Kalau mau ganti path atau tuning parameter, cukup ubah file ini.
# ============================================================

from pathlib import Path

# ── Root project ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Folder paths ──────────────────────────────────────────────
RAW_DATA_DIR       = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR          = BASE_DIR / "models"
REPORTS_DIR        = BASE_DIR / "reports" / "figures"

# ── Dataset files ─────────────────────────────────────────────
# Key  = nama kategori internal
# Value = nama file CSV di folder data/raw/
DATA_FILES = {
    "books":     "amazon_books_Data.csv",
    "ebook":     "amazon_ebook_Data.csv",
    "grocery":   "amazon_grocery_Data.csv",
    "jewellery": "amazon_jwellery_Data.csv",
    "pc":        "amazon_pc_Data.csv",
}

# ── Processed data output ─────────────────────────────────────
CLEAN_DATA_PATH    = PROCESSED_DATA_DIR / "products_clean.csv"
FEATURES_DATA_PATH = PROCESSED_DATA_DIR / "products_features.csv"

# ── Model / artifact paths ────────────────────────────────────
TFIDF_VECTORIZER_PATH    = MODEL_DIR / "tfidf_vectorizer.pkl"
CONTENT_SIMILARITY_PATH  = MODEL_DIR / "content_similarity.pkl"
PRODUCT_LOOKUP_PATH      = MODEL_DIR / "product_lookup.pkl"
POPULARITY_MODEL_PATH    = MODEL_DIR / "popularity_model.pkl"
HYBRID_MODEL_PATH        = MODEL_DIR / "hybrid_model.pkl"

# ── Preprocessing params ──────────────────────────────────────
SAMPLE_SIZE    = 50_000   # batas max row per kategori supaya tidak OOM
RANDOM_STATE   = 42
MIN_RATING     = 0.0      # filter produk dengan rating di bawah ini
MIN_TITLE_LEN  = 3        # filter produk dengan judul terlalu pendek (karakter)

# ── Feature engineering params ────────────────────────────────
# Bobot untuk menghitung popularity_score:
#   popularity_score = W_RATING * norm_rating + W_REVIEW * norm_review_count
W_RATING = 0.5
W_REVIEW = 0.5

# ── TF-IDF params ─────────────────────────────────────────────
TFIDF_MAX_FEATURES = 10_000
TFIDF_NGRAM_RANGE  = (1, 2)   # unigram + bigram

# ── Recommendation params ─────────────────────────────────────
TOP_K = 10   # jumlah rekomendasi default

# Bobot hybrid:
#   final_score = HYBRID_CONTENT_WEIGHT * content_score
#               + HYBRID_POPULARITY_WEIGHT * popularity_score
HYBRID_CONTENT_WEIGHT    = 0.7
HYBRID_POPULARITY_WEIGHT = 0.3

# ── Kolom final yang diharapkan setelah standardisasi ─────────
EXPECTED_COLUMNS = [
    "product_id",
    "title",
    "category",
    "rating",
    "review_count",
    "price",
    "description",
    "review_text",
    "category_source",
]
