# src/recommender_hybrid.py
# ============================================================
# HybridRecommender
#
# Masalah content-based murni:
#   - Produk baru yang teksnya bagus tapi tidak populer → ikut masuk
#   - Produk teks mirip tapi ratingnya jelek → ikut masuk
#
# Solusi: gabungkan content similarity + popularity score.
# Produk yang direkomendasikan harus mirip DAN populer.
#
# Formula (bobot dari config.py):
#   final_score = 0.7 × content_score + 0.3 × popularity_score
#
# Kenapa 70/30?
#   Content similarity lebih penting (kita mau yang relevan),
#   tapi popularity sebagai tiebreaker kalau dua produk sama miripnya.
#   Bobot bisa diubah di config.py tanpa ganti kode ini.
# ============================================================

import numpy as np
import pandas as pd
from typing import Optional

from src.config import (
    TOP_K,
    HYBRID_CONTENT_WEIGHT,
    HYBRID_POPULARITY_WEIGHT,
    HYBRID_MODEL_PATH,
)
from src.utils import setup_logger, save_pickle, load_pickle
from src.recommender_content import ContentBasedRecommender
from src.recommender_popularity import PopularityRecommender

logger = setup_logger("recommender_hybrid")


class HybridRecommender:
    """
    Hybrid recommender: content similarity + popularity score.
    """

    def __init__(
        self,
        content_weight: float = HYBRID_CONTENT_WEIGHT,
        popularity_weight: float = HYBRID_POPULARITY_WEIGHT,
    ):
        self.content_weight    = content_weight
        self.popularity_weight = popularity_weight

        self._content_rec: Optional[ContentBasedRecommender] = None
        self._popularity_rec: Optional[PopularityRecommender] = None
        self._df: Optional[pd.DataFrame] = None
        self.is_fitted: bool = False

    def fit(
        self,
        df: pd.DataFrame,
        content_recommender: ContentBasedRecommender,
        popularity_recommender: PopularityRecommender,
    ) -> "HybridRecommender":
        """
        Attach dua recommender yang sudah di-fit.
        HybridRecommender tidak melatih model baru — dia
        hanya menggabungkan hasil dari keduanya.

        Args:
            df                    : DataFrame dengan kolom popularity_score.
            content_recommender   : Sudah di-fit.
            popularity_recommender: Sudah di-fit.
        """
        if not content_recommender.is_fitted:
            raise ValueError("content_recommender belum di-fit.")
        if not popularity_recommender.is_fitted:
            raise ValueError("popularity_recommender belum di-fit.")

        self._content_rec    = content_recommender
        self._popularity_rec = popularity_recommender
        self._df             = df.copy()

        self.is_fitted = True
        logger.info(
            f"✅ HybridRecommender fitted — "
            f"weights: content={self.content_weight}, "
            f"popularity={self.popularity_weight}"
        )
        return self

    def combine_scores(
        self,
        content_scores: np.ndarray,
        popularity_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Gabungkan content score dan popularity score dengan bobot.

        Args:
            content_scores    : Array similarity score dari TF-IDF [0,1]
            popularity_scores : Array popularity score [0,1]

        Returns:
            Array final_score [0,1]
        """
        return (
            self.content_weight    * content_scores
            + self.popularity_weight * popularity_scores
        )

    def recommend(
        self,
        product_id: str,
        top_k: int = TOP_K,
    ) -> pd.DataFrame:
        """
        Rekomendasi hybrid untuk satu product_id.

        Alur:
            1. Ambil content similarity scores (semua produk vs query)
            2. Ambil popularity scores (dari DataFrame)
            3. Normalisasi keduanya ke [0,1]
            4. Hitung final_score = 0.7*content + 0.3*popularity
            5. Return top-K (skip produk itu sendiri)

        Args:
            product_id : ID produk referensi.
            top_k      : Jumlah rekomendasi.

        Returns:
            DataFrame top-K rekomendasi dengan kolom final_score.
        """
        if not self.is_fitted:
            raise RuntimeError("Model belum di-fit.")

        # ── Step 1: Content scores ─────────────────────────────
        content_sim = self._content_rec.get_similarity_scores(product_id)
        if content_sim is None:
            logger.warning(
                f"⚠️  product_id '{product_id}' tidak ada di content model. "
                "Fallback ke popularity-only."
            )
            return self._popularity_rec.recommend(top_k=top_k)

        product_ids = self._content_rec.product_ids

        # ── Step 2: Popularity scores (align dengan urutan product_ids) ──
        pop_lookup = {}
        if "popularity_score" in self._df.columns and "product_id" in self._df.columns:
            pop_lookup = dict(
                zip(
                    self._df["product_id"].astype(str),
                    self._df["popularity_score"].fillna(0),
                )
            )

        pop_scores = np.array([
            pop_lookup.get(str(pid), 0.0) for pid in product_ids
        ])

        # ── Step 3: Normalize ──────────────────────────────────
        def _norm(arr: np.ndarray) -> np.ndarray:
            mn, mx = arr.min(), arr.max()
            if mx == mn:
                return np.full_like(arr, 0.5)
            return (arr - mn) / (mx - mn)

        norm_content = _norm(content_sim)
        norm_pop     = _norm(pop_scores)

        # ── Step 4: Final score ────────────────────────────────
        final_scores = self.combine_scores(norm_content, norm_pop)

        # ── Step 5: Top-K (exclude self) ──────────────────────
        query_idx   = product_ids.index(product_id)
        sorted_idxs = np.argsort(final_scores)[::-1]
        top_indices = [i for i in sorted_idxs if i != query_idx][:top_k]

        # Build result DataFrame
        result_rows = []
        for i in top_indices:
            pid = product_ids[i]
            row = {"product_id": pid}

            # Ambil detail dari df_index content recommender
            if self._content_rec.df_index is not None:
                detail = self._content_rec.df_index.iloc[i].to_dict()
                row.update(detail)

            row["content_score"]    = round(float(norm_content[i]), 4)
            row["popularity_score"] = round(float(norm_pop[i]), 4)
            row["final_score"]      = round(float(final_scores[i]), 4)
            result_rows.append(row)

        result = pd.DataFrame(result_rows)
        if not result.empty:
            result = result.sort_values("final_score", ascending=False)
            result = result.reset_index(drop=True)

        return result

    def save(self, path=HYBRID_MODEL_PATH) -> None:
        """Simpan model ke .pkl."""
        if not self.is_fitted:
            raise RuntimeError("Model belum di-fit.")
        save_pickle(self, path)

    @classmethod
    def load(cls, path=HYBRID_MODEL_PATH) -> "HybridRecommender":
        """Load model dari .pkl."""
        obj = load_pickle(path)
        logger.info("✅ HybridRecommender loaded")
        return obj
