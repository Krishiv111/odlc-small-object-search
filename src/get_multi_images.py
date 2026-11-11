"""Download the extra stand-in images used for the generalization experiment
(train_rl_generalization.py / build_oracle_multi.py). All three are SAHI's
own public demo images -- different scenes on purpose (parking lot, desert
convoy, aerial marina) so training on two and holding out the third is an
actual test of generalizing across different-looking photos, not just
different crops of the same one.
"""
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data"
FILES = {
    "terrain2.png": f"{BASE}/terrain2.png",
    "obb_test_image.png": f"{BASE}/obb_test_image.png",
}
OUT_DIR = Path("data/multi")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        out_path = OUT_DIR / name
        if out_path.exists():
            print(f"{out_path} already exists, skipping")
            continue
        urllib.request.urlretrieve(url, out_path)
        print(f"Downloaded {out_path}")


if __name__ == "__main__":
    main()
