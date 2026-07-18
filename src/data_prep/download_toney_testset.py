"""
download_toney_testset.py - Download Toney et al. test set from Zenodo

Downloads ONLY the test split CSV (~55 MB) from the 10 GB Zenodo archive
using HTTP range requests (no need to download the full zip).

Zenodo record: https://zenodo.org/records/13840776
Paper: Toney et al. PNAS 2025, "Graph neural networks for predicting
       metal-ligand coordination of transition metal complexes"

Target file: Data/Dataset Curation/Curated Datasets/ligand_data_final_test.csv
             (~6,616 ligands with SMILES + coordinating atom labels)

Usage:
    python download_toney_testset.py --output_dir data/toney
    python download_toney_testset.py --output_dir data/toney --also_holdout
"""

import os
import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ZENODO_RECORD = "13840776"
# Direct download URL for the single Data.zip file
ZIP_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files/Data.zip"

# Files to extract (paths inside the zip)
TARGET_FILES = {
    'test': 'Data/Dataset Curation/Curated Datasets/ligand_data_final_test.csv',
    'holdout': 'Data/Dataset Curation/Curated Datasets/ligand_data_final_holdout.csv',
    'readme': 'Data/readme.txt',
}


def download_with_remotezip(output_dir: str, files_to_extract: list):
    """Use remotezip to extract specific files via HTTP range requests."""
    try:
        from remotezip import RemoteZip
    except ImportError:
        logger.error("remotezip not installed. Install with: pip install remotezip")
        logger.info("Falling back to full download method...")
        download_with_requests(output_dir, files_to_extract)
        return

    logger.info(f"Connecting to {ZIP_URL}")
    logger.info("Using HTTP range requests (only downloading target files, not full 10GB)")

    with RemoteZip(ZIP_URL) as rz:
        # List available files
        all_files = rz.namelist()
        logger.info(f"Archive contains {len(all_files)} entries")

        for key in files_to_extract:
            target = TARGET_FILES[key]
            if target in all_files:
                info = rz.getinfo(target)
                size_mb = info.file_size / (1024 * 1024)
                logger.info(f"Extracting: {target} ({size_mb:.1f} MB)")

                data = rz.read(target)
                out_name = os.path.basename(target)
                out_path = os.path.join(output_dir, out_name)

                with open(out_path, 'wb') as f:
                    f.write(data)
                logger.info(f"  Saved to {out_path}")
            else:
                logger.warning(f"  NOT FOUND in archive: {target}")
                # Try partial match
                matches = [f for f in all_files if os.path.basename(target) in f]
                if matches:
                    logger.info(f"  Possible matches: {matches[:5]}")


def download_with_requests(output_dir: str, files_to_extract: list):
    """Fallback: download full zip and extract. Only use if remotezip fails."""
    import requests
    import zipfile
    import tempfile

    zip_path = os.path.join(output_dir, 'Data.zip')

    if not os.path.exists(zip_path):
        logger.info(f"Downloading full archive to {zip_path}")
        logger.info("This will download ~10 GB. Consider installing remotezip instead.")

        response = requests.get(ZIP_URL, stream=True)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))

        downloaded = 0
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    if downloaded % (100 * 1024 * 1024) < 8192 * 1024:
                        logger.info(f"  {downloaded / 1e9:.1f} / {total / 1e9:.1f} GB ({pct:.0f}%)")

    logger.info("Extracting target files from zip...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for key in files_to_extract:
            target = TARGET_FILES[key]
            try:
                data = zf.read(target)
                out_name = os.path.basename(target)
                out_path = os.path.join(output_dir, out_name)
                with open(out_path, 'wb') as f:
                    f.write(data)
                logger.info(f"  Extracted {out_path}")
            except KeyError:
                logger.warning(f"  {target} not found in zip")


def inspect_csv(csv_path: str):
    """Quick inspection of the downloaded CSV."""
    import csv

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        n_rows = sum(1 for _ in reader)

    logger.info(f"\n  File: {csv_path}")
    logger.info(f"  Columns ({len(header)}): {header}")
    logger.info(f"  Rows: {n_rows}")

    # Show first few rows
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for i, row in enumerate(reader):
            if i >= 3:
                break
            logger.info(f"  Row {i}: {row[:5]}...")


def main():
    parser = argparse.ArgumentParser(
        description='Download Toney et al. test set from Zenodo'
    )
    parser.add_argument('--output_dir', type=str, default='data/toney',
                        help='Output directory')
    parser.add_argument('--also_holdout', action='store_true',
                        help='Also download the holdout set (33.7 MB)')
    parser.add_argument('--full_download', action='store_true',
                        help='Force full zip download instead of remotezip')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = ['test', 'readme']
    if args.also_holdout:
        files.append('holdout')

    logger.info("=" * 60)
    logger.info("Downloading Toney et al. (PNAS 2025) test set")
    logger.info("=" * 60)
    logger.info(f"Zenodo: https://zenodo.org/records/{ZENODO_RECORD}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Files: {files}")

    if args.full_download:
        download_with_requests(args.output_dir, files)
    else:
        download_with_remotezip(args.output_dir, files)

    # Inspect downloaded files
    test_csv = os.path.join(args.output_dir, 'ligand_data_final_test.csv')
    if os.path.exists(test_csv):
        logger.info("\n" + "=" * 60)
        logger.info("TEST SET INSPECTION")
        logger.info("=" * 60)
        inspect_csv(test_csv)
        size_mb = os.path.getsize(test_csv) / (1024 * 1024)
        logger.info(f"\n  Size: {size_mb:.1f} MB")
        logger.info(f"  Ready for evaluation with eval_coordination_binding.py")
    else:
        logger.error(f"Test CSV not found at {test_csv}")


if __name__ == '__main__':
    main()
