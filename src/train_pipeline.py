# src/train_pipeline.py
# ============================================================
# Orchestrator untuk training semua recommender.
# Dipanggil oleh scripts/run_training.py.
#
# Urutan training:
#   1. Load processed data (products_features.csv)
#   2. Cek apakah kolom user_id ada
#      → ada : bisa collaborative filtering (future)
#      → tidak ada : skip, pakai popularity + content + hybrid
#   3. Fit PopularityRecommender
#   4. Fit ContentBasedRecommender
#   5. Fit HybridRecommender
#   6. Save semua artifact ke models/
#   7. Quick evaluation
# ============================================================

import pandas as pd
from pathlib import Path

from src.config import MODEL_DIR, FEATURES_DATA_PATH
from src.data_loader import load_processed_data
from src.recommender_popularity import PopularityRecommender
from src.recommender_content import ContentBasedRecommender
from src.recommender_hybrid import HybridRecommender
from src.evaluation import run_quick_eval
from src.utils import setup_logger, ensure_dir

logger = setup_logger("train_pipeline")


def _check_collaborative_filtering(df: pd.DataFrame) -> bool:
    """
    Cek apakah dataset mendukung collaborative filtering.
    CF butuh kolom user_id (dan biasanya item_id + rating).
    """
    has_user_id = "user_id" in df.columns and df["user_id"].notna().any()

    if not has_user_id:
        logger.info(
            "ℹ️  Collaborative filtering dilewati karena kolom "
            "user_id tidak tersedia."
        )
    else:
        logger.info(
            "✅ user_id ditemukan — dataset mendukung collaborative filtering. "
            "(Implementasi CF belum aktif di versi ini.)"
        )

    return has_user_id


def run_training_pipeline(
    data_path: Path = FEATURES_DATA_PATH,
    run_eval: bool = True,
    eval_samples: int = 5,
) -> dict:
    """
    Jalankan full training pipeline.

    Args:
        data_path    : Path ke products_features.csv.
        run_eval     : Jalankan quick evaluation setelah training.
        eval_samples : Jumlah produk untuk evaluasi.

    Returns:
        Dict berisi semua fitted recommender:
        {
            "popularity": PopularityRecommender,
            "content":    ContentBasedRecommender,
            "hybrid":     HybridRecommender,
        }
    """
    logger.info("=" * 60)
    logger.info("🚀 Memulai training pipeline — ShopWise AI")
    logger.info("=" * 60)

    # ── Pastikan folder models ada ────────────────────────────
    ensure_dir(MODEL_DIR)

    # ── Load data ─────────────────────────────────────────────
    logger.info(f"📥 Loading data dari: {data_path}")
    df = load_processed_data(data_path)
    logger.info(f"   Shape: {df.shape}")

    # ── Cek CF ───────────────────────────────────────────────
    _check_collaborative_filtering(df)

    # ── Validasi kolom minimum ────────────────────────────────
    if "text_features" not in df.columns:
        raise ValueError(
            "Kolom 'text_features' tidak ditemukan. "
            "Pastikan sudah menjalankan: python scripts/run_preprocessing.py"
        )

    # ── Training ──────────────────────────────────────────────

    # 1. Popularity Recommender
    logger.info("\n[1/3] Training PopularityRecommender...")
    popularity_rec = PopularityRecommender()
    popularity_rec.fit(df)
    popularity_rec.save()

    # 2. Content-Based Recommender
    logger.info("\n[2/3] Training ContentBasedRecommender (TF-IDF)...")
    content_rec = ContentBasedRecommender()
    content_rec.fit(df)
    content_rec.save()

    # 3. Hybrid Recommender
    logger.info("\n[3/3] Training HybridRecommender...")
    hybrid_rec = HybridRecommender()
    hybrid_rec.fit(df, content_rec, popularity_rec)
    hybrid_rec.save()

    logger.info("\n" + "=" * 60)
    logger.info("✅ Semua model berhasil ditraining dan disimpan!")
    logger.info(f"   📁 Models folder: {MODEL_DIR}")
    logger.info("=" * 60)

    # ── Quick evaluation ──────────────────────────────────────
    if run_eval:
        logger.info("\n🧪 Menjalankan quick evaluation...")
        run_quick_eval(
            df=df,
            content_rec=content_rec,
            popularity_rec=popularity_rec,
            hybrid_rec=hybrid_rec,
            n_samples=eval_samples,
        )

    return {
        "popularity": popularity_rec,
        "content":    content_rec,
        "hybrid":     hybrid_rec,
        "df":         df,
    }
