# src/predict_pipeline.py
# ============================================================
# RecommendationPipeline — jembatan antara model dan Streamlit.
#
# Streamlit TIDAK boleh tahu cara kerja model di balik layar.
# Dia cukup panggil method dari class ini.
#
# Contoh pemakaian di streamlit_app.py:
#
#   pipeline = RecommendationPipeline()
#   pipeline.load_artifacts()
#
#   # Popular products
#   df = pipeline.get_popular_recommendations(top_k=10, category="books")
#
#   # Content-based
#   df = pipeline.get_content_recommendations("books_42", top_k=10)
#
#   # Hybrid
#   df = pipeline.get_hybrid_recommendations("books_42", top_k=10)
#
#   # Search
#   results = pipeline.search_products("gaming laptop", category="pc")
# ============================================================

import pandas as pd
from typing import Optional, List, Dict, Any

from src.config import (
    FEATURES_DATA_PATH,
    POPULARITY_MODEL_PATH,
    CONTENT_SIMILARITY_PATH,
    HYBRID_MODEL_PATH,
    PRODUCT_LOOKUP_PATH,
    TOP_K,
)
from src.recommender_popularity import PopularityRecommender
from src.recommender_content import ContentBasedRecommender
from src.recommender_hybrid import HybridRecommender
from src.utils import setup_logger, load_pickle
from src.data_loader import load_processed_data

logger = setup_logger("predict_pipeline")


class RecommendationPipeline:
    """
    Pipeline utama untuk prediksi rekomendasi.
    Load semua artifact sekali, lalu siap dipakai berkali-kali.
    """

    def __init__(self):
        self._popularity_rec: Optional[PopularityRecommender]  = None
        self._content_rec:    Optional[ContentBasedRecommender] = None
        self._hybrid_rec:     Optional[HybridRecommender]       = None
        self._product_lookup: Optional[Dict[str, Any]]          = None
        self._df:             Optional[pd.DataFrame]            = None
        self.is_loaded: bool = False

    def load_artifacts(self) -> "RecommendationPipeline":
        """
        Load semua artifact dari folder models/.
        Panggil method ini sekali di awal (bukan tiap request).

        Raise FileNotFoundError kalau ada artifact yang belum ada.
        Pesan error-nya akan menginstruksikan user untuk run training dulu.
        """
        logger.info("📦 Loading artifacts...")

        self._popularity_rec = PopularityRecommender.load(POPULARITY_MODEL_PATH)
        self._content_rec    = ContentBasedRecommender.load(CONTENT_SIMILARITY_PATH)
        self._hybrid_rec     = HybridRecommender.load(HYBRID_MODEL_PATH)
        self._product_lookup = load_pickle(PRODUCT_LOOKUP_PATH)

        # Load DataFrame untuk keperluan search dan EDA
        self._df = load_processed_data(FEATURES_DATA_PATH)

        self.is_loaded = True
        logger.info(
            f"✅ Semua artifact loaded — "
            f"{len(self._df):,} produk siap"
        )
        return self

    def _check_loaded(self) -> None:
        """Guard: pastikan load_artifacts() sudah dipanggil."""
        if not self.is_loaded:
            raise RuntimeError(
                "Artifacts belum di-load. Panggil pipeline.load_artifacts() dulu."
            )

    # ── Product info ──────────────────────────────────────────

    def get_product_detail(self, product_id: str) -> Optional[Dict[str, Any]]:
        """
        Ambil detail satu produk berdasarkan product_id.

        Returns:
            Dict detail produk, atau None kalau tidak ketemu.
        """
        self._check_loaded()
        return self._product_lookup.get(str(product_id))

    def get_all_categories(self) -> List[str]:
        """Kembalikan daftar kategori yang tersedia."""
        self._check_loaded()
        return sorted(self._df["category_source"].unique().tolist())

    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Statistik dasar dataset untuk ditampilkan di halaman Dataset Overview.
        """
        self._check_loaded()
        df = self._df

        stats = {
            "total_products":   len(df),
            "total_categories": df["category_source"].nunique(),
            "avg_rating":       round(df["rating"].mean(), 2) if "rating" in df.columns else None,
            "columns":          list(df.columns),
            "products_per_cat": df["category_source"].value_counts().to_dict(),
        }
        return stats

    # ── Search ────────────────────────────────────────────────

    def search_products(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: int = 20,
    ) -> pd.DataFrame:
        """
        Cari produk berdasarkan keyword di kolom title.
        Case-insensitive substring match.

        Args:
            query    : Kata kunci pencarian.
            category : Filter kategori (optional).
            top_k    : Maksimum hasil yang dikembalikan.

        Returns:
            DataFrame produk yang cocok.
        """
        self._check_loaded()

        df = self._df.copy()
        query = query.strip().lower()

        if not query:
            return pd.DataFrame()

        # Search di title (case-insensitive)
        mask = df["title"].str.lower().str.contains(query, na=False)
        results = df[mask].copy()

        # Filter kategori kalau diminta
        if category and category.lower() != "all":
            results = results[
                results["category_source"].str.lower() == category.lower()
            ]

        # Urutkan by popularity kalau ada
        if "popularity_score" in results.columns:
            results = results.sort_values("popularity_score", ascending=False)

        cols = [
            c for c in
            ["product_id", "title", "category_source", "rating",
             "review_count", "popularity_score"]
            if c in results.columns
        ]
        return results[cols].head(top_k).reset_index(drop=True)

    # ── Recommendations ───────────────────────────────────────

    def get_popular_recommendations(
        self,
        top_k: int = TOP_K,
        category: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Rekomendasi produk paling populer.

        Args:
            top_k    : Jumlah rekomendasi.
            category : Filter kategori (None = semua).
        """
        self._check_loaded()
        return self._popularity_rec.recommend(top_k=top_k, category=category)

    def get_content_recommendations(
        self,
        product_id: str,
        top_k: int = TOP_K,
    ) -> pd.DataFrame:
        """
        Rekomendasi berdasarkan kemiripan teks (content-based).

        Args:
            product_id : ID produk referensi.
            top_k      : Jumlah rekomendasi.
        """
        self._check_loaded()
        return self._content_rec.recommend(product_id=product_id, top_k=top_k)

    def get_hybrid_recommendations(
        self,
        product_id: str,
        top_k: int = TOP_K,
    ) -> pd.DataFrame:
        """
        Rekomendasi hybrid (content + popularity).

        Args:
            product_id : ID produk referensi.
            top_k      : Jumlah rekomendasi.
        """
        self._check_loaded()
        return self._hybrid_rec.recommend(product_id=product_id, top_k=top_k)

    # ── EDA data ──────────────────────────────────────────────

    def get_eda_data(self) -> Dict[str, Any]:
        """
        Siapkan semua data yang dibutuhkan halaman EDA Summary.

        Returns:
            Dict berisi:
            - rating_distribution : Series
            - top_categories      : Series
            - avg_rating_per_cat  : Series
            - products_per_cat    : Series
            - top_products        : DataFrame
        """
        self._check_loaded()
        df = self._df

        eda = {}

        # Distribusi rating
        if "rating" in df.columns:
            eda["rating_distribution"] = df["rating"].dropna()

        # Top categories by product count
        eda["products_per_cat"] = (
            df["category_source"].value_counts()
        )

        # Average rating per category
        if "rating" in df.columns:
            eda["avg_rating_per_cat"] = (
                df.groupby("category_source")["rating"]
                .mean()
                .round(2)
                .sort_values(ascending=False)
            )

        # Top 10 products by popularity
        if "popularity_score" in df.columns:
            cols = [
                c for c in
                ["product_id", "title", "category_source", "rating",
                 "review_count", "popularity_score"]
                if c in df.columns
            ]
            eda["top_products"] = (
                df[cols]
                .sort_values("popularity_score", ascending=False)
                .head(10)
                .reset_index(drop=True)
            )

        return eda

    def get_dataframe(self) -> pd.DataFrame:
        """Return raw DataFrame (untuk preview di Dataset Overview page)."""
        self._check_loaded()
        return self._df.copy()
