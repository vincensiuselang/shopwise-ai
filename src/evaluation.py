# src/evaluation.py
# ============================================================
# Evaluasi sederhana untuk sistem rekomendasi tanpa user_id.
#
# Kalau ada user_id + interaction history, kita bisa pakai
# metrik standar seperti Precision@K, Recall@K, NDCG.
# Tapi dataset ini adalah product catalog, bukan user-item matrix.
#
# Evaluasi yang kita pakai:
#   1. Coverage   : Berapa % produk yang pernah direkomendasikan?
#   2. Diversity  : Berapa banyak kategori berbeda dalam rekomendasi?
#   3. Sanity check: Apakah rekomendasi masuk akal secara manual?
#
# Ini bukan angka final untuk paper akademik,
# tapi cukup untuk menunjukkan model bekerja dengan benar.
# ============================================================

import pandas as pd
import numpy as np
from typing import List, Dict, Any

from src.utils import setup_logger

logger = setup_logger("evaluation")


def evaluate_coverage(
    all_product_ids: List[str],
    recommended_ids: List[str],
) -> Dict[str, Any]:
    """
    Hitung catalog coverage: berapa persen produk yang pernah direkomendasikan.

    Makin tinggi coverage, makin beragam rekomendasi yang bisa diberikan.
    Coverage rendah = sistem selalu merekomendasikan produk yang sama-sama itu.

    Args:
        all_product_ids   : Semua product_id yang ada di dataset.
        recommended_ids   : Semua product_id yang pernah direkomendasikan.

    Returns:
        Dict berisi coverage_count, total, coverage_pct.
    """
    covered = len(set(recommended_ids) & set(all_product_ids))
    total   = len(set(all_product_ids))
    pct     = 100 * covered / total if total > 0 else 0.0

    result = {
        "covered_products": covered,
        "total_products":   total,
        "coverage_pct":     round(pct, 2),
    }
    logger.info(
        f"📊 Coverage: {covered}/{total} ({pct:.1f}%)"
    )
    return result


def evaluate_diversity(
    recommendations: pd.DataFrame,
    category_col: str = "category_source",
) -> Dict[str, Any]:
    """
    Hitung diversity: berapa banyak kategori berbeda dalam satu set rekomendasi.

    Diversity tinggi → rekomendasi tidak terlalu narrow (tidak semua buku saja).
    Diversity rendah → bisa jadi feature, bisa jadi bug, tergantung use case.

    Args:
        recommendations : DataFrame hasil recommend() dari recommender.
        category_col    : Nama kolom kategori.

    Returns:
        Dict berisi n_unique_categories, category_distribution, diversity_score.
    """
    if recommendations.empty:
        return {"n_unique_categories": 0, "diversity_score": 0.0}

    if category_col not in recommendations.columns:
        logger.warning(f"⚠️  Kolom '{category_col}' tidak ada di recommendations.")
        return {"n_unique_categories": 0, "diversity_score": 0.0}

    cats = recommendations[category_col].value_counts()
    n    = len(recommendations)
    n_unique = len(cats)

    # Diversity score = normalized entropy
    # Maksimal kalau setiap rekomendasi dari kategori berbeda
    probs = cats / n
    entropy = -np.sum(probs * np.log(probs + 1e-9))
    max_entropy = np.log(n_unique + 1e-9)
    diversity_score = entropy / max_entropy if max_entropy > 0 else 0.0

    result = {
        "n_unique_categories":   n_unique,
        "category_distribution": cats.to_dict(),
        "diversity_score":       round(float(diversity_score), 4),
    }
    logger.info(
        f"📊 Diversity: {n_unique} kategori, score={diversity_score:.3f}"
    )
    return result


def simple_recommendation_report(
    product_query: pd.Series,
    recommendations: pd.DataFrame,
    method: str = "unknown",
) -> None:
    """
    Print laporan sederhana rekomendasi untuk sanity check manual.

    Args:
        product_query   : Row produk yang dijadikan query (pd.Series).
        recommendations : DataFrame hasil recommend().
        method          : Nama metode ('content', 'popularity', 'hybrid').
    """
    print("\n" + "=" * 65)
    print(f"🔎  RECOMMENDATION REPORT — Method: {method.upper()}")
    print("=" * 65)

    if not product_query.empty:
        print(f"Query Product:")
        print(f"  ID       : {product_query.get('product_id', 'N/A')}")
        print(f"  Title    : {product_query.get('title', 'N/A')}")
        print(f"  Category : {product_query.get('category_source', 'N/A')}")
        print(f"  Rating   : {product_query.get('rating', 'N/A')}")
        print()

    if recommendations.empty:
        print("⚠️  Tidak ada rekomendasi yang dihasilkan.")
        return

    print(f"Top-{len(recommendations)} Recommendations:")
    print(f"{'#':<4} {'Title':<45} {'Cat':<12} {'Rating':<8} {'Score':<8}")
    print("-" * 80)

    score_col = next(
        (c for c in ["final_score", "content_score", "popularity_score"]
         if c in recommendations.columns),
        None,
    )

    for i, row in recommendations.iterrows():
        title   = str(row.get("title", ""))[:43]
        cat     = str(row.get("category_source", ""))[:10]
        rating  = f"{row.get('rating', 'N/A')}"
        score   = f"{row.get(score_col, 0.0):.4f}" if score_col else "N/A"
        print(f"{i+1:<4} {title:<45} {cat:<12} {rating:<8} {score:<8}")

    print("=" * 65 + "\n")


def run_quick_eval(
    df: pd.DataFrame,
    content_rec,
    popularity_rec,
    hybrid_rec,
    n_samples: int = 5,
) -> None:
    """
    Jalankan evaluasi cepat terhadap semua recommender.
    Sample beberapa produk, generate rekomendasi, tampilkan hasilnya.

    Args:
        df            : DataFrame produk lengkap.
        content_rec   : ContentBasedRecommender (sudah di-fit).
        popularity_rec: PopularityRecommender (sudah di-fit).
        hybrid_rec    : HybridRecommender (sudah di-fit).
        n_samples     : Jumlah produk yang di-sample untuk evaluasi.
    """
    logger.info("🧪 Menjalankan quick evaluation...")

    sample_products = df.sample(
        n=min(n_samples, len(df)), random_state=42
    )

    all_content_ids    = []
    all_hybrid_ids     = []
    all_diversity_info = []

    for _, query_product in sample_products.iterrows():
        pid = str(query_product["product_id"])

        # Content recommendations
        try:
            content_recs = content_rec.recommend(pid, top_k=10)
            all_content_ids.extend(content_recs["product_id"].tolist())
        except Exception as e:
            logger.warning(f"Content rec error for {pid}: {e}")
            content_recs = pd.DataFrame()

        # Hybrid recommendations
        try:
            hybrid_recs = hybrid_rec.recommend(pid, top_k=10)
            all_hybrid_ids.extend(hybrid_recs["product_id"].tolist())

            div = evaluate_diversity(hybrid_recs)
            all_diversity_info.append(div["diversity_score"])
        except Exception as e:
            logger.warning(f"Hybrid rec error for {pid}: {e}")

    # Coverage
    all_ids = df["product_id"].tolist()
    content_cov = evaluate_coverage(all_ids, all_content_ids)
    hybrid_cov  = evaluate_coverage(all_ids, all_hybrid_ids)

    avg_diversity = np.mean(all_diversity_info) if all_diversity_info else 0.0

    print("\n" + "=" * 55)
    print("📈 EVALUATION SUMMARY")
    print("=" * 55)
    print(f"  Total produk di dataset  : {len(df):,}")
    print(f"  Sample queries           : {n_samples}")
    print()
    print(f"  Content-Based Coverage   : {content_cov['coverage_pct']:.1f}%")
    print(f"  Hybrid Coverage          : {hybrid_cov['coverage_pct']:.1f}%")
    print(f"  Avg Hybrid Diversity     : {avg_diversity:.3f} (0=no diversity, 1=max)")
    print("=" * 55 + "\n")
