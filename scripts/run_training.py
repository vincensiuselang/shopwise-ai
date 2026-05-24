# scripts/run_training.py
# ============================================================
# Jalankan: python scripts/run_training.py
#
# Alur:
#   1. Load products_features.csv (hasil preprocessing)
#   2. Fit PopularityRecommender → popularity_model.pkl
#   3. Fit ContentBasedRecommender → content_similarity.pkl + tfidf_vectorizer.pkl
#   4. Fit HybridRecommender → hybrid_model.pkl
#   5. Quick evaluation
# ============================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train_pipeline import run_training_pipeline
from src.utils import setup_logger

logger = setup_logger("run_training")


if __name__ == "__main__":
    logger.info("🏋️  Memulai training semua recommender...")
    results = run_training_pipeline(run_eval=True, eval_samples=5)
    logger.info("🎉 Training selesai! Siap untuk dipakai di Streamlit.")
    logger.info("   Jalankan: streamlit run app/streamlit_app.py")
