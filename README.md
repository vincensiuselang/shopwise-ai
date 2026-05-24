# 🛒 ShopWise AI — Hybrid Amazon Product Recommender

> A modular, portfolio-ready Machine Learning project that recommends Amazon products using Popularity-Based, Content-Based (TF-IDF), and Hybrid approaches — deployed as a Streamlit web app.

---

## 📸 Demo

| Home | Popular Products | Recommendations |
|---|---|---|
| ![home](reports/figures/home.png) | ![popular](reports/figures/popular.png) | ![recs](reports/figures/recs.png) |

---

## 🧠 How It Works

```
CSV Files (5 categories)
        ↓
   Data Loading     → auto-detect columns, handle missing files
        ↓
  Column Mapping    → standardize flexible column names
        ↓
  Preprocessing     → clean text, ratings, prices, remove duplicates
        ↓
Feature Engineering → TF-IDF text features + popularity score
        ↓
     Training       → PopularityRec + ContentRec + HybridRec
        ↓
  Streamlit App     → search, filter, recommend
```

**Hybrid Formula:**
```
final_score = 0.7 × content_similarity + 0.3 × popularity_score
```

---

## 📂 Project Structure

```
shopwise-ai/
├── app/
│   ├── streamlit_app.py      ← UI (calls predict_pipeline only)
│   └── components.py         ← Reusable UI components
│
├── data/
│   ├── raw/                  ← Raw CSV files (not committed to Git)
│   └── processed/            ← Cleaned & engineered data
│
├── models/                   ← Trained artifacts (.pkl)
│
├── src/                      ← All ML logic lives here
│   ├── config.py             ← Paths, params, constants
│   ├── data_loader.py        ← Load & merge CSV files
│   ├── column_mapper.py      ← Flexible column detection
│   ├── preprocessing.py      ← Clean text, ratings, prices
│   ├── feature_engineering.py← TF-IDF features + popularity score
│   ├── recommender_popularity.py
│   ├── recommender_content.py
│   ├── recommender_hybrid.py
│   ├── evaluation.py
│   ├── train_pipeline.py
│   ├── predict_pipeline.py   ← Bridge: Streamlit ↔ Models
│   └── utils.py
│
├── scripts/                  ← CLI runners
├── notebooks/                ← Exploratory analysis
├── main.py                   ← Full pipeline orchestrator
├── requirements.txt
└── README.md
```

---

## 🚀 Cara Menjalankan Project (Step by Step)

### Step 0 — Clone & Install Dependencies

```bash
git clone https://github.com/yourusername/shopwise-ai.git
cd shopwise-ai
pip install -r requirements.txt
```

---

### Step 1 — Taruh Dataset CSV di Folder `data/raw/`

Pastikan 5 file ini sudah ada sebelum lanjut:

```
shopwise-ai/
└── data/
    └── raw/
        ├── amazon_books_Data.csv
        ├── amazon_ebook_Data.csv
        ├── amazon_grocery_Data.csv
        ├── amazon_jwellery_Data.csv
        └── amazon_pc_Data.csv
```

---

### Step 2 — Cek Dataset

Verifikasi semua file terbaca, cek kolom dan missing values:

```bash
python scripts/run_data_check.py
```

Output yang diharapkan:
```
✅ [books]   loaded → shape: (10000, 9)
✅ [ebook]   loaded → shape: (5000, 7)
✅ [grocery] loaded → shape: (8000, 10)
...
✅ File ditemukan: 5
```

Kalau ada file yang `❌ tidak ditemukan`, cek ulang nama file dan lokasi folder.

---

### Step 3 — Preprocessing

Bersihkan data dan buat fitur. Output disimpan ke `data/processed/`:

```bash
python scripts/run_preprocessing.py
```

Output yang diharapkan:
```
✅ Preprocessing selesai → output: (XXXXX, 9)
💾 Saved: data/processed/products_clean.csv
💾 Saved: data/processed/products_features.csv
```

---

### Step 4 — Training Model

Latih semua recommender dan simpan artifact ke `models/`:

```bash
python scripts/run_training.py
```

Output yang diharapkan:
```
[1/3] Training PopularityRecommender... ✅
[2/3] Training ContentBasedRecommender (TF-IDF)... ✅
[3/3] Training HybridRecommender... ✅
Artifact disimpan → models/popularity_model.pkl
Artifact disimpan → models/content_similarity.pkl
Artifact disimpan → models/hybrid_model.pkl
```

---

### Step 5 — Jalankan Web App

```bash
streamlit run app/streamlit_app.py
```

Buka browser di: **http://localhost:8501**

---

### Alternatif: Jalankan Full Pipeline Sekaligus

```bash
python main.py
```

Ini akan menjalankan Step 2–4 sekaligus, lalu kamu tinggal jalankan Streamlit secara terpisah.

---

### ⚠️ Troubleshooting

| Error | Solusi |
|---|---|
| `FileNotFoundError: data/raw/...` | Pastikan file CSV sudah ada di folder `data/raw/` |
| `ModuleNotFoundError` | Jalankan `pip install -r requirements.txt` |
| `Artifact tidak ditemukan: models/...` | Jalankan dulu `python scripts/run_training.py` |
| Streamlit error saat pertama buka | Pastikan training sudah selesai (Step 4) |
| Dataset terlalu besar / RAM habis | Kecilkan `SAMPLE_SIZE` di `src/config.py` |

---

## 🤖 Recommender Engines

### Popularity-Based
- Ranks products by `popularity_score = 0.5 × norm_rating + 0.5 × norm_review_count`
- Filterable by category
- Best for: cold-start, homepage, category browsing

### Content-Based (TF-IDF)
- Combines: `title × 3 + category + description + review_text`
- TF-IDF vectorizer (10k features, unigram + bigram)
- Cosine similarity (computed on-demand — memory efficient)
- Best for: "more like this" recommendations

### Hybrid
- `final_score = 0.7 × content_score + 0.3 × popularity_score`
- Best overall recommendations
- Weights configurable in `src/config.py`

---

## 🔧 Configuration

All tunable parameters are in `src/config.py`:

```python
SAMPLE_SIZE          = 50_000     # max rows per category
TFIDF_MAX_FEATURES   = 10_000     # TF-IDF vocabulary size
HYBRID_CONTENT_WEIGHT    = 0.7    # content weight in hybrid
HYBRID_POPULARITY_WEIGHT = 0.3    # popularity weight in hybrid
TOP_K                = 10         # default recommendations count
```

---

## 📊 Web App Features

| Page | Features |
|---|---|
| 🏠 Home | Project overview, dataset info, quick start guide |
| 📊 Dataset Overview | Stats, category distribution, data preview |
| 🔥 Popular Products | Top products by category + bar chart |
| 🔍 Product Search | Keyword search with category filter |
| 🤖 Recommendation | Search → select → choose method → get top-K recs |
| 📈 EDA Summary | Rating distribution, category pie chart, avg rating |

---

## 📦 Dataset Columns

The system **auto-detects** column names across different CSV formats. Supported column name variants:

| Standard | Accepted Variants |
|---|---|
| `title` | product_name, name, item_name, products |
| `rating` | stars, average_rating, star_rating |
| `review_count` | no_of_ratings, ratings_count, num_reviews |
| `price` | actual_price, discounted_price, selling_price |
| `description` | about_product, product_description, detail |

---

## 🛠️ Tech Stack

| Component | Library |
|---|---|
| Data Processing | pandas, numpy |
| ML / Vectorization | scikit-learn (TF-IDF, cosine similarity) |
| Model Persistence | joblib |
| Web App | Streamlit |
| Visualization | Plotly, Matplotlib |

---

## 📝 Notes

- **No collaborative filtering**: This dataset doesn't contain `user_id`. The system auto-detects and skips CF if no user data is found.
- **Memory efficient**: Similarity computed on-demand (not pre-computed N×N matrix).
- **Flexible column mapping**: Works with any Amazon product CSV format.

---

## 👤 Author

Built by **Vincensius christian elang putra** as a portfolio project.

---

## 📄 License

MIT License
