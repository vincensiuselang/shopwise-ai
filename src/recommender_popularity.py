# src/recommender_popularity.py
# ============================================================
# PopularityRecommender
#
# Ide dasarnya simpel: tampilkan produk yang paling populer.
# "Populer" = kombinasi rating tinggi + banyak review.
# Skornya sudah dihitung di feature_engineering.py
# sebagai kolom popularity_score.
#
# Keunggulan: tidak butuh input user, cocok untuk cold-start
# (user baru yang belum punya history).
# ============================================================

import pandas as pd
from typing import Optional

from src.config import TOP_K, POPULARITY_MODEL_PATH
from src.utils import setup_logger, save_pickle, load_pickle

logger = setup_logger("recommender_popularity")


class PopularityRecommender:
    """
    Merekomendasikan produk paling populer berdasarkan popularity_score.
    Bisa difilter per kategori.
    """

    def __init__(self):
        self._df: Optional[pd.DataFrame] = None
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "PopularityRecommender":
        """
        Simpan DataFrame yang sudah punya popularity_score.
        Urutkan dari yang paling populer.

        Args:
            df : DataFrame hasil feature_engineering (wajib ada
                 kolom popularity_score, title, category_source).
        """
        required = ["popularity_score", "title", "category_source", "product_id"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

        # Simpan versi sudah diurutkan supaya recommend() cepat
        self._df = (
            df.sort_values("popularity_score", ascending=False)
            .reset_index(drop=True)
        )
        self.is_fitted = True
        logger.info(
            f"✅ PopularityRecommender fitted — "
            f"{len(self._df):,} produk, "
            f"{self._df['category_source'].nunique()} kategori"
        )
        return self

    def recommend(
        self,
        top_k: int = TOP_K,
        category: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Kembalikan top-K produk paling populer.

        Args:
            top_k    : Jumlah rekomendasi yang diminta.
            category : Filter berdasarkan category_source.
                       None = semua kategori.

        Returns:
            DataFrame berisi kolom:
            product_id, title, category_source, rating,
            review_count, popularity_score
        """
        if not self.is_fitted:
            raise RuntimeError("Model belum di-fit. Panggil fit() dulu.")

        df = self._df.copy()

        # Filter kategori kalau diminta
        if category and category.lower() != "all":
            df = df[df["category_source"].str.lower() == category.lower()]
            if df.empty:
                logger.warning(f"⚠️  Tidak ada produk untuk kategori '{category}'")
                return pd.DataFrame()

        cols = [
            c for c in
            ["product_id", "title", "category_source", "rating",
             "review_count", "popularity_score"]
            if c in df.columns
        ]
        return df[cols].head(top_k).reset_index(drop=True)

    def save(self, path=POPULARITY_MODEL_PATH) -> None:
        """Simpan model ke file .pkl."""
        if not self.is_fitted:
            raise RuntimeError("Model belum di-fit.")
        save_pickle(self, path)

    @classmethod
    def load(cls, path=POPULARITY_MODEL_PATH) -> "PopularityRecommender":
        """Load model dari file .pkl."""
        obj = load_pickle(path)
        logger.info("✅ PopularityRecommender loaded")
        return obj
