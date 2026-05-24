# app/components.py
# ============================================================
# Reusable UI components untuk Streamlit.
# Semua fungsi render ada di sini supaya streamlit_app.py
# tetap bersih dan fokus ke logic halaman saja.
# ============================================================

import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any


def render_sidebar() -> str:
    """
    Render sidebar navigasi dan kembalikan nama halaman aktif.

    Returns:
        Nama halaman yang dipilih user.
    """
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align:center; padding: 10px 0 20px 0;'>
                <h2 style='color:#FF9900; margin:0;'>🛒 ShopWise AI</h2>
                <small style='color:#888;'>Amazon Product Recommender</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        page = st.radio(
            "Navigasi",
            options=[
                "🏠 Home",
                "📊 Dataset Overview",
                "🔥 Popular Products",
                "🔍 Product Search",
                "🤖 Recommendation",
                "📈 EDA Summary",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        st.caption("Powered by TF-IDF + Cosine Similarity")
        st.caption("Built with ❤️ using Streamlit")

    return page


def render_metric_cards(metrics: Dict[str, Any]) -> None:
    """
    Tampilkan kartu metrik dalam satu baris.

    Args:
        metrics : Dict { label: value }
    """
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label=label, value=value)


def render_product_card(product: Dict[str, Any]) -> None:
    """
    Tampilkan detail satu produk dalam sebuah card.

    Args:
        product : Dict hasil get_product_detail() dari pipeline.
    """
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])

        with col1:
            title = str(product.get("title", "Unknown Product")).title()
            st.markdown(f"**{title}**")

            cat = str(product.get("category_source", "")).upper()
            st.caption(f"📦 Kategori: {cat}")

        with col2:
            rating = product.get("rating")
            if rating is not None:
                stars = "⭐" * int(round(float(rating)))
                st.markdown(f"{stars}")
                st.caption(f"{rating:.1f} / 5.0")

        col3, col4 = st.columns(2)
        with col3:
            reviews = product.get("review_count")
            if reviews is not None:
                st.caption(f"💬 {int(reviews):,} reviews")
        with col4:
            price = product.get("price")
            if price is not None and str(price) not in ("nan", "None", ""):
                st.caption(f"💰 ${float(price):,.2f}")


def render_recommendation_table(
    df: pd.DataFrame,
    score_col: Optional[str] = None,
    show_rank: bool = True,
) -> None:
    """
    Tampilkan tabel rekomendasi yang rapi dengan warna.

    Args:
        df        : DataFrame hasil recommend().
        score_col : Nama kolom skor untuk ditampilkan.
        show_rank : Tampilkan nomor urut.
    """
    if df.empty:
        st.warning("⚠️ Tidak ada rekomendasi yang ditemukan.")
        return

    display_df = df.copy()

    # Tambah rank
    if show_rank:
        display_df.insert(0, "#", range(1, len(display_df) + 1))

    # Kolom yang ditampilkan
    priority_cols = ["#", "title", "category_source", "rating",
                     "review_count", score_col or ""]
    show_cols = [c for c in priority_cols if c in display_df.columns]

    # Rename kolom supaya lebih user-friendly
    rename_map = {
        "title":            "Product",
        "category_source":  "Category",
        "rating":           "⭐ Rating",
        "review_count":     "💬 Reviews",
        "popularity_score": "🔥 Popularity",
        "content_score":    "🎯 Relevance",
        "final_score":      "🏆 Score",
    }

    result = display_df[show_cols].rename(columns=rename_map)

    # Format angka
    for col in result.columns:
        if "Rating" in col or "Score" in col or "Relevance" in col or "Popularity" in col:
            result[col] = result[col].apply(
                lambda x: f"{float(x):.3f}" if pd.notna(x) else "—"
            )
        if "Reviews" in col:
            result[col] = result[col].apply(
                lambda x: f"{int(x):,}" if pd.notna(x) else "—"
            )

    # Capitalize Product column
    if "Product" in result.columns:
        result["Product"] = result["Product"].apply(
            lambda x: str(x).title() if pd.notna(x) else x
        )

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True,
    )


def render_not_loaded_warning() -> None:
    """Tampilkan pesan kalau artifact belum di-load."""
    st.error(
        "⚠️ **Model belum siap!**\n\n"
        "Jalankan pipeline berikut dulu di terminal:\n\n"
        "```bash\n"
        "# Step 1: Taruh CSV dataset di folder data/raw/\n"
        "python scripts/run_data_check.py\n\n"
        "# Step 2: Preprocessing\n"
        "python scripts/run_preprocessing.py\n\n"
        "# Step 3: Training\n"
        "python scripts/run_training.py\n"
        "```"
    )


def render_empty_state(message: str = "Tidak ada data.") -> None:
    """Tampilkan state kosong yang informatif."""
    st.info(f"ℹ️ {message}")
