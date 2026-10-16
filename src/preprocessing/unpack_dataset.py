"""
ISL Connect — Dataset Unpacker (Phase 4/5).

Safely extracts the RealSign Indian Sign Language Dataset from Dataset.zip
into dataset/raw/ preserving the Training / Testing / Validation / Letters structure.
"""

from pathlib import Path
import sys
import time
import zipfile
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ZIP_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "RealSign-Indian-Sign-Language-Dataset-main"
    / "Dataset.zip"
)
DEFAULT_RAW_DEST = PROJECT_ROOT / "dataset" / "raw"


def unpack_dataset(
    zip_path: Optional[Path] = None,
    dest_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> Dict[str, int]:
    """
    Extract Dataset.zip into destination directory.

    Returns:
    --------
    dict of split/folder -> count of files extracted
    """
    zip_p = Path(zip_path) if zip_path else DEFAULT_ZIP_PATH
    dest = Path(dest_dir) if dest_dir else DEFAULT_RAW_DEST

    if not zip_p.exists():
        raise FileNotFoundError(f"Dataset archive not found at: {zip_p}")

    dest.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    training_sample = dest / "Training" / "A"
    if training_sample.exists() and any(training_sample.iterdir()) and not overwrite:
        print(f"Dataset already extracted at {dest}. Skipping extraction.")
        return {"status": "already_extracted"}

    print(f"Opening archive: {zip_p} ({zip_p.stat().st_size / (1024*1024):.1f} MB)...")
    start_time = time.time()

    counts: Dict[str, int] = {}
    with zipfile.ZipFile(zip_p, "r") as z:
        members = z.infolist()
        total_files = len(members)
        print(f"Extracting {total_files} items to {dest}...")

        for idx, member in enumerate(members):
            z.extract(member, dest)
            parts = Path(member.filename).parts
            if parts:
                top_folder = parts[0]
                counts[top_folder] = counts.get(top_folder, 0) + 1

            if (idx + 1) % 5000 == 0 or idx + 1 == total_files:
                elapsed = time.time() - start_time
                print(f"  Extracted {idx + 1}/{total_files} files ({elapsed:.1f}s)...")

    elapsed_total = time.time() - start_time
    print(f"Extraction completed in {elapsed_total:.1f}s!")
    for k, v in sorted(counts.items()):
        print(f"  - {k}: {v} files")

    return counts


if __name__ == "__main__":
    unpack_dataset()
