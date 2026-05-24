# app/streamlit_app.py
# ============================================================
# Entry point Streamlit — ShopWise AI
#
# Jalankan dengan: streamlit run app/streamlit_app.py
#
# PENTING:
#   File ini HANYA boleh berisi UI logic.
#   Semua ML/data logic dipanggil lewat RecommendationPipeline.
#   Jangan import recommender atau preprocessing langsung di sini.
# ============================================================

import sys
from pathlib import Path

# Tambah root ke sys.path supaya import src.* bisa jalan
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.predict_pipeline import RecommendationPipeline
from app.components import (
    render_sidebar,
    render_metric_cards,
    render_product_card,
    render_recommendation_table,
    render_not_loaded_warning,
    render_empty_state,
)

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopWise AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; }
        h1 { color: #FF9900; }
        .stRadio > label { font-weight: bold; }
        div[data-testid="metric-container"] {
            background: #1a1a2e;
            border: 1px solid #FF9900;
            border-radius: 8px;
            padding: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Load pipeline (cached — hanya load sekali per session) ──────
@st.cache_resource(show_spinner="⏳ Loading model artifacts...")
def load_pipeline() -> RecommendationPipeline:
    """
    Load semua artifact ke memori.
    Di-cache oleh Streamlit supaya tidak reload tiap interaksi.
    """
    pipeline = RecommendationPipeline()
    pipeline.load_artifacts()
    return pipeline


# ── Sidebar & routing ──────────────────────────────────────────
page = render_sidebar()

# Coba load pipeline
pipeline_ready = False
pipeline       = None

try:
    pipeline       = load_pipeline()
    pipeline_ready = True
except FileNotFoundError as e:
    pass  # Akan ditangani di masing-masing halaman


# ==============================================================
# HALAMAN: HOME
# ==============================================================
if page == "🏠 Home":
    st.title("🛒 ShopWise AI")
    st.subheader("Hybrid Amazon Product Recommendation System")

    st.markdown(
        """
        Selamat datang di **ShopWise AI** — sistem rekomendasi produk Amazon
        yang dibangun menggunakan pendekatan hybrid machine learning.
        """
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📦 Dataset yang Digunakan")
        datasets = {
            "📚 Books":       "amazon_books_Data.csv",
            "📖 eBook":       "amazon_ebook_Data.csv",
            "🛒 Grocery":     "amazon_grocery_Data.csv",
            "💍 Jewellery":   "amazon_jwellery_Data.csv",
            "💻 PC / Laptop": "amazon_pc_Data.csv",
        }
        for cat, fname in datasets.items():
            st.markdown(f"- **{cat}** → `data/raw/{fname}`")

        st.markdown("### 🤖 Metode Rekomendasi")
        st.markdown(
            """
            | Metode | Cara Kerja |
            |---|---|
            | 🔥 **Popularity-Based** | Produk dengan rating tinggi + banyak review |
            | 🎯 **Content-Based** | Produk dengan teks mirip (TF-IDF + Cosine Similarity) |
            | 🏆 **Hybrid** | Gabungan content (70%) + popularity (30%) |
            """
        )

    with col2:
        st.markdown("### 🚀 Cara Pakai")
        st.code(
            """# 1. Cek dataset
python scripts/run_data_check.py

# 2. Preprocessing
python scripts/run_preprocessing.py

# 3. Training
python scripts/run_training.py

# 4. Jalankan app
streamlit run app/streamlit_app.py
""",
            language="bash",
        )

    if pipeline_ready:
        st.success("✅ Model siap dipakai! Pilih halaman di sidebar.")
    else:
        render_not_loaded_warning()


# ==============================================================
# HALAMAN: DATASET OVERVIEW
# ==============================================================
elif page == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")

    if not pipeline_ready:
        render_not_loaded_warning()
        st.stop()

    stats = pipeline.get_dataset_stats()

    # Metric cards
    render_metric_cards({
        "🗂️ Total Produk":    f"{stats['total_products']:,}",
        "📦 Kategori":        stats["total_categories"],
        "⭐ Avg Rating":      stats["avg_rating"] or "N/A",
        "📋 Total Kolom":     len(stats["columns"]),
    })

    st.divider()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Produk per Kategori")
        per_cat = pd.DataFrame(
            list(stats["products_per_cat"].items()),
            columns=["Kategori", "Jumlah"],
        ).sort_values("Jumlah", ascending=False)
        st.dataframe(per_cat, hide_index=True, use_container_width=True)

    with col2:
        fig = px.bar(
            per_cat,
            x="Kategori",
            y="Jumlah",
            color="Kategori",
            title="Distribusi Produk per Kategori",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("#### 📋 Kolom Dataset")
    st.write(stats["columns"])

    st.divider()
    st.markdown("#### 👀 Preview Data (50 baris pertama)")
    df_preview = pipeline.get_dataframe()
    preview_cols = [
        c for c in
        ["product_id", "title", "category_source", "rating",
         "review_count", "price", "popularity_score"]
        if c in df_preview.columns
    ]
    st.dataframe(
        df_preview[preview_cols].head(50),
        use_container_width=True,
        hide_index=True,
    )


# ==============================================================
# HALAMAN: POPULAR PRODUCTS
# ==============================================================
elif page == "🔥 Popular Products":
    st.title("🔥 Popular Products")
    st.caption("Produk paling populer berdasarkan rating + jumlah review")

    if not pipeline_ready:
        render_not_loaded_warning()
        st.stop()

    col1, col2 = st.columns([1, 3])

    with col1:
        categories = ["All"] + pipeline.get_all_categories()
        selected_cat = st.selectbox("Filter Kategori", options=categories)
        top_k = st.slider("Jumlah Produk", min_value=5, max_value=50, value=10)

    category_filter = None if selected_cat == "All" else selected_cat
    popular = pipeline.get_popular_recommendations(
        top_k=top_k, category=category_filter
    )

    with col2:
        st.markdown(
            f"#### Top {top_k} Produk Populer"
            + (f" — Kategori: **{selected_cat}**" if selected_cat != "All" else "")
        )

    if popular.empty:
        render_empty_state(f"Tidak ada produk untuk kategori '{selected_cat}'.")
    else:
        render_recommendation_table(popular, score_col="popularity_score")

        st.divider()
        if "popularity_score" in popular.columns and "title" in popular.columns:
            popular_display = popular.copy()
            popular_display["title"] = popular_display["title"].str.title().str[:40]
            fig = px.bar(
                popular_display.head(10),
                x="popularity_score",
                y="title",
                orientation="h",
                title="Top 10 — Popularity Score",
                labels={"popularity_score": "Score", "title": "Produk"},
                color="popularity_score",
                color_continuous_scale="Oranges",
            )
            fig.update_layout(
                yaxis={"autorange": "reversed"},
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)


# ==============================================================
# HALAMAN: PRODUCT SEARCH
# ==============================================================
elif page == "🔍 Product Search":
    st.title("🔍 Product Search")
    st.caption("Cari produk berdasarkan nama/judul")

    if not pipeline_ready:
        render_not_loaded_warning()
        st.stop()

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Cari produk...",
            placeholder="Contoh: laptop, coffee, python book...",
        )
    with col2:
        categories = ["All"] + pipeline.get_all_categories()
        cat_filter = st.selectbox("Kategori", options=categories, key="search_cat")

    if query:
        category_filter = None if cat_filter == "All" else cat_filter
        results = pipeline.search_products(
            query=query, category=category_filter, top_k=30
        )

        if results.empty:
            render_empty_state(
                f"Tidak ada produk yang cocok dengan '{query}'."
            )
        else:
            st.markdown(f"**{len(results)} produk ditemukan** untuk: `{query}`")
            render_recommendation_table(results, score_col="popularity_score")
    else:
        st.info("💡 Ketik nama produk di kotak pencarian di atas.")


# ==============================================================
# HALAMAN: RECOMMENDATION
# ==============================================================
elif page == "🤖 Recommendation":
    st.title("🤖 Product Recommendation")
    st.caption("Pilih produk dan metode rekomendasi")

    if not pipeline_ready:
        render_not_loaded_warning()
        st.stop()

    # ── Step 1: Cari produk referensi ─────────────────────────
    st.markdown("#### Step 1: Cari produk referensi")
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input(
            "Cari produk yang kamu suka...",
            placeholder="Contoh: laptop gaming, coffee, python...",
            key="rec_search",
        )
    with col2:
        categories = ["All"] + pipeline.get_all_categories()
        cat_filter = st.selectbox("Kategori", options=categories, key="rec_cat")

    selected_product_id = None
    selected_product    = None

    if search_query:
        category_filter = None if cat_filter == "All" else cat_filter
        search_results  = pipeline.search_products(
            query=search_query, category=category_filter, top_k=20
        )

        if search_results.empty:
            st.warning(f"Tidak ada produk untuk '{search_query}'.")
        else:
            # Buat label untuk selectbox
            options_map = {
                f"{row['title'].title()[:60]} [{row.get('category_source','').upper()}]": row["product_id"]
                for _, row in search_results.iterrows()
            }

            st.markdown("#### Step 2: Pilih produk")
            selected_label = st.selectbox(
                "Produk referensi:",
                options=list(options_map.keys()),
                key="rec_product",
            )

            if selected_label:
                selected_product_id = options_map[selected_label]
                selected_product    = pipeline.get_product_detail(selected_product_id)

                if selected_product:
                    st.markdown("**Produk yang dipilih:**")
                    render_product_card(selected_product)

    # ── Step 3: Pilih metode & generate ────────────────────────
    if selected_product_id:
        st.divider()
        st.markdown("#### Step 3: Pilih metode rekomendasi")

        col1, col2 = st.columns([2, 1])
        with col1:
            method = st.radio(
                "Metode:",
                options=["🏆 Hybrid (Recommended)", "🎯 Content-Based", "🔥 Popularity-Based"],
                horizontal=True,
            )
        with col2:
            top_k = st.slider("Jumlah Rekomendasi", 5, 20, 10, key="rec_topk")

        if st.button("🚀 Generate Rekomendasi", type="primary", use_container_width=True):
            with st.spinner("Mencari rekomendasi terbaik..."):
                if "Hybrid" in method:
                    recs      = pipeline.get_hybrid_recommendations(selected_product_id, top_k)
                    score_col = "final_score"
                elif "Content" in method:
                    recs      = pipeline.get_content_recommendations(selected_product_id, top_k)
                    score_col = "content_score"
                else:
                    cat = selected_product.get("category_source") if selected_product else None
                    recs      = pipeline.get_popular_recommendations(top_k, category=cat)
                    score_col = "popularity_score"

            st.markdown(f"#### 🎁 Top {top_k} Rekomendasi — {method}")

            if recs.empty:
                render_empty_state("Tidak ada rekomendasi yang ditemukan.")
            else:
                render_recommendation_table(recs, score_col=score_col)

                # Visualisasi score
                if score_col in recs.columns:
                    recs_vis = recs.copy()
                    recs_vis["title"] = recs_vis["title"].str.title().str[:40]
                    fig = px.bar(
                        recs_vis.head(10),
                        x=score_col,
                        y="title",
                        orientation="h",
                        title=f"Score Distribusi — {method}",
                        color=score_col,
                        color_continuous_scale="Blues",
                    )
                    fig.update_layout(
                        yaxis={"autorange": "reversed"},
                        plot_bgcolor="rgba(0,0,0,0)",
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)


# ==============================================================
# HALAMAN: EDA SUMMARY
# ==============================================================
elif page == "📈 EDA Summary":
    st.title("📈 EDA Summary")
    st.caption("Eksplorasi dataset Amazon Product")

    if not pipeline_ready:
        render_not_loaded_warning()
        st.stop()

    eda = pipeline.get_eda_data()

    # ── Distribusi Rating ──────────────────────────────────────
    if "rating_distribution" in eda:
        st.markdown("### ⭐ Distribusi Rating")
        fig = px.histogram(
            eda["rating_distribution"],
            nbins=20,
            title="Distribusi Rating Produk",
            labels={"value": "Rating", "count": "Jumlah Produk"},
            color_discrete_sequence=["#FF9900"],
        )
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)

    # ── Produk per Kategori ────────────────────────────────────
    with col1:
        if "products_per_cat" in eda:
            st.markdown("### 📦 Jumlah Produk per Kategori")
            per_cat_df = eda["products_per_cat"].reset_index()
            per_cat_df.columns = ["Kategori", "Jumlah"]
            fig = px.pie(
                per_cat_df,
                values="Jumlah",
                names="Kategori",
                title="Proporsi Produk per Kategori",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Avg Rating per Kategori ───────────────────────────────
    with col2:
        if "avg_rating_per_cat" in eda:
            st.markdown("### ⭐ Rata-rata Rating per Kategori")
            avg_df = eda["avg_rating_per_cat"].reset_index()
            avg_df.columns = ["Kategori", "Avg Rating"]
            fig = px.bar(
                avg_df,
                x="Kategori",
                y="Avg Rating",
                title="Average Rating per Kategori",
                color="Avg Rating",
                color_continuous_scale="Viridis",
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis_range=[0, 5],
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Top 10 Products ────────────────────────────────────────
    if "top_products" in eda:
        st.markdown("### 🏆 Top 10 Most Popular Products")
        top_df = eda["top_products"].copy()
        top_df["title"] = top_df["title"].str.title()
        render_recommendation_table(top_df, score_col="popularity_score")
