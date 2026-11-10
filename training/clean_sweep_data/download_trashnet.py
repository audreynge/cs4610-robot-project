"""Download TrashNet (classification images, one object per photo).

Used only for the crop classifier, never the detector.

    uv run download-trashnet --out data/raw/trashnet
"""

from __future__ import annotations

import argparse
from pathlib import Path

from clean_sweep_data.download import extract_zip, fetch

ARCHIVE_URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/raw/trashnet"))
    args = parser.parse_args()

    archive = args.out / "dataset-resized.zip"
    print(f"Downloading TrashNet (~43 MB) to {archive}")
    fetch(ARCHIVE_URL, archive)
    extract_zip(archive, args.out)
    archive.unlink()
    print(f"Extracted to {args.out / 'dataset-resized'}")


if __name__ == "__main__":
    main()
