# scripts/run_preprocessing.py
# ============================================================
# Jalankan: python scripts/run_preprocessing.py
#
# Alur:
#   1. Load semua CSV dari data/raw/
#   2. Standarisasi kolom (column_mapper)
#   3. Preprocessing (clean text, numerik, duplikat, dll)
#   4. Feature engineering (text_features, popularity_score)
#   5. Simpan ke data/processed/products_clean.csv
#              dan data/processed/products_features.csv
# ============================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_all_datasets, save_processed_data
from src.column_mapper import standardize_columns
from src.preprocessing import preprocess_products
from src.feature_engineering import run_feature_engineering
from src.config import CLEAN_DATA_PATH, FEATURES_DATA_PATH
from src.utils import setup_logger, ensure_dir

logger = setup_logger("run_preprocessing")


def main():
    logger.info("🚀 Memulai preprocessing pipeline...")

    # Pastikan folder output ada
    ensure_dir(CLEAN_DATA_PATH.parent)

    # Step 1: Load semua CSV
    df_raw = load_all_datasets()

    # Step 2: Standarisasi kolom
    logger.info("🔄 Standarisasi kolom...")
    df_std = standardize_columns(df_raw)

    # Step 3: Preprocessing
    df_clean = preprocess_products(df_std)

    # Step 4: Simpan products_clean.csv
    save_processed_data(df_clean, CLEAN_DATA_PATH)
    logger.info(f"💾 Saved: {CLEAN_DATA_PATH}")

    # Step 5: Feature engineering
    df_features = run_feature_engineering(df_clean)

    # Simpan products_features.csv
    save_processed_data(df_features, FEATURES_DATA_PATH)
    logger.info(f"💾 Saved: {FEATURES_DATA_PATH}")

    logger.info("✅ Preprocessing pipeline selesai!")
    logger.info(f"   products_clean.csv    → {df_clean.shape}")
    logger.info(f"   products_features.csv → {df_features.shape}")


if __name__ == "__main__":
    main()
