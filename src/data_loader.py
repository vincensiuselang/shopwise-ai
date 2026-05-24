# src/data_loader.py
# ============================================================
# Tanggung jawab: baca semua CSV dari data/raw/, gabungkan,
# dan simpan/load data processed.
#
# Kenapa dipisah dari preprocessing?
# → Supaya kalau kamu ganti sumber data (misal dari DB atau API),
#   cukup ubah file ini saja. Preprocessing tidak perlu tahu
#   datanya dari mana asalnya.
# ============================================================

import pandas as pd
from pathlib import Path
from typing import Optional

from src.config import (
    RAW_DATA_DIR,
    DATA_FILES,
    CLEAN_DATA_PATH,
    SAMPLE_SIZE,
    RANDOM_STATE,
)
from src.utils import setup_logger, ensure_dir, reduce_memory_usage

logger = setup_logger("data_loader")


def load_single_csv(
    file_path: Path,
    category_name: str,
    sample_size: Optional[int] = SAMPLE_SIZE,
) -> Optional[pd.DataFrame]:
    """
    Baca satu file CSV dan tambahkan kolom category_source.

    Args:
        file_path     : Path lengkap ke file CSV.
        category_name : Nama kategori (misal 'books', 'pc').
        sample_size   : Kalau dataset terlalu besar, ambil sebagian.
                        Set None untuk load semua baris.

    Returns:
        DataFrame kalau sukses, None kalau file tidak ditemukan.
    """
    if not Path(file_path).exists():
        logger.warning(f"⚠️  File tidak ditemukan, dilewati: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path, low_memory=False)
        logger.info(f"✅ [{category_name}] loaded → shape: {df.shape}")
        logger.info(f"   Kolom: {list(df.columns)}")

        # Tambah penanda dari mana data ini berasal
        df["category_source"] = category_name

        # Sampling kalau terlalu besar
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=RANDOM_STATE)
            logger.info(
                f"   ⚡ Sampled: {sample_size} rows dari {len(df)} total"
            )

        return df

    except Exception as e:
        logger.error(f"❌ Gagal membaca {file_path}: {e}")
        return None


def load_all_datasets(
    raw_dir: Path = RAW_DATA_DIR,
    data_files: dict = DATA_FILES,
    sample_size: Optional[int] = SAMPLE_SIZE,
) -> pd.DataFrame:
    """
    Load semua file CSV dari DATA_FILES, gabungkan jadi satu DataFrame.

    Kalau ada file yang tidak ditemukan, dilewati (tidak error brutal).
    Kalau semua file tidak ada, baru raise error.

    Returns:
        DataFrame gabungan dari semua dataset.
    """
    logger.info("=" * 55)
    logger.info("📂 Memulai load semua dataset...")
    logger.info("=" * 55)

    frames = []
    missing_files = []

    for category, filename in data_files.items():
        file_path = Path(raw_dir) / filename
        df = load_single_csv(file_path, category_name=category, sample_size=sample_size)

        if df is not None:
            frames.append(df)
        else:
            missing_files.append(filename)

    if not frames:
        raise FileNotFoundError(
            f"Tidak ada file CSV yang berhasil dibaca dari: {raw_dir}\n"
            f"File yang dicari: {list(data_files.values())}\n"
            "Pastikan file CSV sudah ada di folder data/raw/"
        )

    if missing_files:
        logger.warning(f"⚠️  File berikut tidak ditemukan dan dilewati: {missing_files}")

    # Gabungkan semua DataFrame
    # ignore_index=True → reset index supaya tidak duplikat
    combined = pd.concat(frames, axis=0, ignore_index=True)

    logger.info("-" * 55)
    logger.info(f"✅ Dataset gabungan → shape: {combined.shape}")
    logger.info(f"   Kategori tersedia: {combined['category_source'].unique().tolist()}")
    logger.info(f"   Kolom: {list(combined.columns)}")
    logger.info("=" * 55)

    # Optimasi memory
    combined = reduce_memory_usage(combined)

    return combined


def save_processed_data(df: pd.DataFrame, path: Path = CLEAN_DATA_PATH) -> None:
    """
    Simpan DataFrame hasil preprocessing ke file CSV.

    Args:
        df   : DataFrame yang sudah diproses.
        path : Path tujuan (default ke data/processed/products_clean.csv).
    """
    ensure_dir(Path(path).parent)
    df.to_csv(path, index=False)
    logger.info(f"💾 Data disimpan → {path} (shape: {df.shape})")


def load_processed_data(path: Path = CLEAN_DATA_PATH) -> pd.DataFrame:
    """
    Load DataFrame dari file CSV hasil preprocessing.

    Raise FileNotFoundError dengan pesan yang jelas
    kalau file belum ada (belum jalankan preprocessing).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data tidak ditemukan: {path}\n"
            "Jalankan dulu: python scripts/run_preprocessing.py"
        )

    df = pd.read_csv(path, low_memory=False)
    logger.info(f"📥 Processed data loaded → shape: {df.shape}")
    return df
