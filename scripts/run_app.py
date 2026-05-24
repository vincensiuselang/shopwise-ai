# scripts/run_app.py
# ============================================================
# Informasi cara menjalankan Streamlit web app.
# ============================================================

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP  = ROOT / "app" / "streamlit_app.py"


def main():
    print("\n" + "=" * 55)
    print("🚀  ShopWise AI — Menjalankan Web App")
    print("=" * 55)
    print(f"\nApp path: {APP}")
    print("\nJalankan perintah berikut:")
    print(f"\n   streamlit run {APP}\n")
    print("Atau langsung dari root project:")
    print("\n   streamlit run app/streamlit_app.py\n")
    print("=" * 55 + "\n")

    # Tanya user apakah mau langsung launch
    try:
        answer = input("Launch sekarang? (y/n): ").strip().lower()
        if answer == "y":
            subprocess.run(
                [sys.executable, "-m", "streamlit", "run", str(APP)],
                check=True,
            )
    except KeyboardInterrupt:
        print("\n👋 Dibatalkan.")


if __name__ == "__main__":
    main()
