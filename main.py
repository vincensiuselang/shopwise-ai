# main.py
# ============================================================
# Orchestrator utama ShopWise AI.
# Jalankan full pipeline dari satu tempat.
#
# Usage:
#   python main.py                   → jalankan full pipeline
#   python main.py --check           → cek dataset saja
#   python main.py --preprocess      → preprocessing saja
#   python main.py --train           → training saja
#   python main.py --skip-eval       → training tanpa evaluasi
# ============================================================

import sys
import argparse
from pathlib import Path

from src.utils import setup_logger, ensure_dir
from src.config import PROCESSED_DATA_DIR, MODEL_DIR

logger = setup_logger("main")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ShopWise AI — Amazon Product Recommender Pipeline"
    )
    parser.add_argument("--check",      action="store_true", help="Hanya cek dataset")
    parser.add_argument("--preprocess", action="store_true", help="Hanya preprocessing")
    parser.add_argument("--train",      action="store_true", help="Hanya training")
    parser.add_argument("--skip-eval",  action="store_true", help="Skip evaluasi setelah training")
    return parser.parse_args()


def run_data_check():
    from scripts.run_data_check import run_data_check as _check
    _check()


def run_preprocessing():
    from src.data_loader import load_all_datasets, save_processed_data
    from src.column_mapper import standardize_columns
    from src.preprocessing import preprocess_products
    from src.feature_engineering import run_feature_engineering
    from src.config import CLEAN_DATA_PATH, FEATURES_DATA_PATH

    ensure_dir(PROCESSED_DATA_DIR)
    df_raw    = load_all_datasets()
    df_std    = standardize_columns(df_raw)
    df_clean  = preprocess_products(df_std)
    save_processed_data(df_clean, CLEAN_DATA_PATH)

    df_feat   = run_feature_engineering(df_clean)
    save_processed_data(df_feat, FEATURES_DATA_PATH)
    logger.info("✅ Preprocessing selesai")


def run_training(skip_eval: bool = False):
    from src.train_pipeline import run_training_pipeline
    ensure_dir(MODEL_DIR)
    run_training_pipeline(run_eval=not skip_eval)
    logger.info("✅ Training selesai")


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("🛒 ShopWise AI — Amazon Product Recommender")
    logger.info("=" * 60)

    if args.check:
        run_data_check()
    elif args.preprocess:
        run_preprocessing()
    elif args.train:
        run_training(skip_eval=args.skip_eval)
    else:
        # Full pipeline
        logger.info("🚀 Menjalankan full pipeline...")
        run_data_check()
        run_preprocessing()
        run_training(skip_eval=args.skip_eval)
        logger.info("\n✅ Full pipeline selesai!")
        logger.info("   Jalankan web app: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
