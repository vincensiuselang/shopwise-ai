# src/recommender_content.py
# ============================================================
# ContentBasedRecommender
#
# Ide: dua produk itu "mirip" kalau teks-nya mirip.
# Kita ubah teks produk ke vektor angka (TF-IDF),
# lalu hitung seberapa dekat vektor dua produk (cosine similarity).
# Makin dekat vektornya, makin mirip produknya.
#
# TF-IDF = Term Frequency × Inverse Document Frequency
# - TF  : kata ini sering muncul di produk ini?
# - IDF : kata ini langka di seluruh dataset? (lebih informatif)
# - Kalikan keduanya → kata yang penting untuk produk ini
#
# Cosine Similarity = sudut antara dua vektor.
# Nilainya 0 (beda total) sampai 1 (identik).
#
# Kenapa tidak pakai embedding model?
# → Lebih ringan, tidak butuh GPU, cocok untuk portfolio.
# → Untuk dataset produk, TF-IDF sudah cukup bagus.
# ============================================================

import numpy as np
import pandas as pd
from typing import Optional, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

from src.config import (
    TOP_K,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_VECTORIZER_PATH,
    CONTENT_SIMILARITY_PATH,
)
from src.utils import setup_logger, save_pickle, load_pickle

logger = setup_logger("recommender_content")


class ContentBasedRecommender:
    """
    Rekomendasi berdasarkan kemiripan teks produk (TF-IDF + cosine similarity).
    """

    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.similarity_matrix: Optional[np.ndarray] = None
        self.product_ids: Optional[List[str]] = None
        self.df_index: Optional[pd.DataFrame] = None   # mapping idx → product info
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "ContentBasedRecommender":
        """
        Fit TF-IDF vectorizer dan hitung cosine similarity matrix.

        Args:
            df : DataFrame dengan kolom 'text_features' dan 'product_id'.
                 Hasil dari feature_engineering.run_feature_engineering().

        Catatan memory:
            Similarity matrix berukuran N×N.
            Untuk 50.000 produk → 50k×50k float32 = ~10GB (tidak muat RAM).
            Solusi: kita simpan sparse TF-IDF matrix saja, similarity
            dihitung on-demand saat predict (hanya satu baris vs semua).
        """
        required = ["text_features", "product_id"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

        # Pastikan tidak ada text_features yang kosong
        df = df[df["text_features"].str.strip().str.len() > 0].copy()
        df = df.reset_index(drop=True)

        logger.info(f"🔧 Fitting TF-IDF — {len(df):,} produk...")

        # Fit TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            strip_accents="unicode",
            analyzer="word",
            sublinear_tf=True,   # log(tf+1) supaya kata sangat sering tidak dominan
        )
        tfidf_matrix = self.vectorizer.fit_transform(df["text_features"])

        # Simpan sebagai sparse matrix (hemat RAM)
        self.tfidf_matrix = tfidf_matrix
        self.product_ids  = df["product_id"].tolist()

        # Simpan info produk untuk hasil rekomendasi
        info_cols = [
            c for c in
            ["product_id", "title", "category_source", "rating",
             "review_count", "popularity_score"]
            if c in df.columns
        ]
        self.df_index = df[info_cols].reset_index(drop=True)

        self.is_fitted = True
        logger.info(
            f"✅ ContentBasedRecommender fitted — "
            f"vocab: {len(self.vectorizer.vocabulary_):,} terms, "
            f"matrix: {tfidf_matrix.shape}"
        )
        return self

    def recommend(
        self,
        product_id: str,
        top_k: int = TOP_K,
    ) -> pd.DataFrame:
        """
        Rekomendasi produk mirip berdasarkan satu product_id.

        Args:
            product_id : ID produk yang dijadikan referensi.
            top_k      : Jumlah rekomendasi.

        Returns:
            DataFrame berisi top-K produk paling mirip
            (tidak termasuk produk itu sendiri).
        """
        if not self.is_fitted:
            raise RuntimeError("Model belum di-fit.")

        # Cari index produk
        if product_id not in self.product_ids:
            logger.warning(f"⚠️  product_id '{product_id}' tidak ditemukan.")
            return pd.DataFrame()

        idx = self.product_ids.index(product_id)

        # Hitung similarity hanya untuk satu baris (hemat RAM vs full matrix)
        # Shape: (1, N) → flatten ke (N,)
        query_vec   = self.tfidf_matrix[idx]
        sim_scores  = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Ambil top-K, skip index produk itu sendiri
        # argsort dari kecil ke besar → kita ambil dari belakang
        top_indices = np.argsort(sim_scores)[::-1]
        top_indices = [i for i in top_indices if i != idx][: top_k]

        result = self.df_index.iloc[top_indices].copy()
        result["content_score"] = sim_scores[top_indices]
        result = result.sort_values("content_score", ascending=False)
        result = result.reset_index(drop=True)

        return result

    def get_similarity_scores(self, product_id: str) -> Optional[np.ndarray]:
        """
        Kembalikan array similarity score satu produk vs semua produk.
        Dipakai oleh HybridRecommender untuk combine score.
        """
        if not self.is_fitted or product_id not in self.product_ids:
            return None

        idx        = self.product_ids.index(product_id)
        query_vec  = self.tfidf_matrix[idx]
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        return sim_scores

    def save(self, path=CONTENT_SIMILARITY_PATH) -> None:
        """Simpan model ke .pkl."""
        if not self.is_fitted:
            raise RuntimeError("Model belum di-fit.")
        save_pickle(self, path)

    @classmethod
    def load(cls, path=CONTENT_SIMILARITY_PATH) -> "ContentBasedRecommender":
        """Load model dari .pkl."""
        obj = load_pickle(path)
        logger.info("✅ ContentBasedRecommender loaded")
        return obj
