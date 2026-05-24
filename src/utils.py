# src/utils.py
# ============================================================
# Helper kecil yang dipakai di banyak tempat.
# Taruh function generik di sini supaya tidak copy-paste.
# ============================================================

import logging
import pickle
from pathlib import Path

import joblib
import pandas as pd


# ── Logger ────────────────────────────────────────────────────

def setup_logger(name: str = "shopwise") -> logging.Logger:
    """
    Buat logger sederhana dengan format yang mudah dibaca.
    Panggil sekali di awal setiap module.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# ── Directory ─────────────────────────────────────────────────

def ensure_dir(path: Path) -> None:
    """Buat folder jika belum ada. Tidak error kalau sudah ada."""
    Path(path).mkdir(parents=True, exist_ok=True)


# ── Pickle helpers ────────────────────────────────────────────

def save_pickle(obj, path: Path) -> None:
    """
    Simpan object ke file .pkl menggunakan joblib.
    Lebih efisien dari pickle biasa untuk numpy array.
    """
    ensure_dir(Path(path).parent)
    joblib.dump(obj, path)
    logger = setup_logger()
    logger.info(f"Artifact disimpan → {path}")


def load_pickle(path: Path):
    """
    Load object dari file .pkl.
    Lempar FileNotFoundError dengan pesan yang jelas kalau file tidak ada.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact tidak ditemukan: {path}\n"
            "Pastikan sudah menjalankan: python scripts/run_training.py"
        )
    return joblib.load(path)


# ── Memory optimization ───────────────────────────────────────

def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Turunkan penggunaan RAM DataFrame dengan downcast tipe data.
    Berguna kalau dataset gabungan cukup besar.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type).startswith("int"):
                if c_min >= -128 and c_max <= 127:
                    df[col] = df[col].astype("int8")
                elif c_min >= -32768 and c_max <= 32767:
                    df[col] = df[col].astype("int16")
                elif c_min >= -2_147_483_648 and c_max <= 2_147_483_647:
                    df[col] = df[col].astype("int32")
            elif str(col_type).startswith("float"):
                df[col] = df[col].astype("float32")

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2

    if verbose:
        logger = setup_logger()
        logger.info(
            f"Memory usage: {start_mem:.2f} MB → {end_mem:.2f} MB "
            f"({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
        )
    return df
