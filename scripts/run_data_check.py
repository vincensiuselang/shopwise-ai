# scripts/run_data_check.py
# ============================================================
# Jalankan: python scripts/run_data_check.py
#
# Tujuan:
#   - Cek apakah semua CSV ada di data/raw/
#   - Tampilkan shape, kolom, missing value, dan sample data
#   - Coba deteksi kolom penting dengan column_mapper
#   - TIDAK mengubah data apapun
# ============================================================

import sys
from pathlib import Path

# Tambah root project ke sys.path supaya import src.* bisa jalan
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.config import RAW_DATA_DIR, DATA_FILES
from src.data_loader import load_single_csv
from src.column_mapper import inspect_columns, standardize_columns
from src.utils import setup_logger

logger = setup_logger("run_data_check")


def run_data_check():
    print("\n" + "=" * 65)
    print("🔎  ShopWise AI — DATA CHECK")
    print("=" * 65)
    print(f"📁 Lokasi data: {RAW_DATA_DIR}\n")

    found = 0
    missing = 0

    for category, filename in DATA_FILES.items():
        file_path = RAW_DATA_DIR / filename
        print(f"\n{'─' * 65}")
        print(f"📦 Kategori  : {category.upper()}")
        print(f"   File      : {filename}")

        if not file_path.exists():
            print(f"   ❌ FILE TIDAK DITEMUKAN: {file_path}")
            missing += 1
            continue

        found += 1

        # Load tanpa sampling supaya kita bisa lihat total size asli
        df = load_single_csv(file_path, category_name=category, sample_size=None)
        if df is None:
            continue

        # Info dasar
        print(f"\n   📊 Shape      : {df.shape[0]:,} baris × {df.shape[1]} kolom")
        print(f"   📝 Kolom      : {list(df.columns)}")

        # Missing value summary
        null_summary = df.isnull().sum()
        null_cols = null_summary[null_summary > 0]
        if null_cols.empty:
            print("   ✅ Missing values: tidak ada")
        else:
            print(f"   ⚠️  Missing values:")
            for col, n in null_cols.items():
                pct = 100 * n / len(df)
                print(f"      {col:<35} {n:>7,} ({pct:.1f}%)")

        # Kolom deteksi otomatis
        print("\n   🔍 Deteksi kolom otomatis:")
        std_df = standardize_columns(df)

        # Preview data
        print("\n   👀 Preview 3 baris pertama:")
        print(df.head(3).to_string(max_colwidth=40))

    # Ringkasan akhir
    print(f"\n{'=' * 65}")
    print(f"✅ File ditemukan : {found}")
    print(f"❌ File tidak ada : {missing}")
    print(f"{'=' * 65}\n")

    if missing > 0:
        print(
            "⚠️  PERHATIAN: Letakkan file CSV yang hilang di folder:\n"
            f"   {RAW_DATA_DIR}\n"
        )


if __name__ == "__main__":
    run_data_check()
