"""
download_toney_trainset.py - Download Toney et al. TRAINING data from Zenodo

Downloads ONLY the training CSV(s) from the 10 GB Zenodo archive using HTTP
range requests (remotezip). Does NOT download the full archive.

Step 1: List archive contents, find training-related CSVs
Step 2: Download training CSV + any relevant metadata

Zenodo record: https://zenodo.org/records/13840776
Paper: Toney et al. PNAS 2025, "Graph neural networks for predicting
       metal-ligand coordination of transition metal complexes"

Usage:
    python download_toney_trainset.py --output_dir data/toney
    python download_toney_trainset.py --output_dir data/toney --list_only
"""

import os
import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ZENODO_RECORD = "13840776"
ZIP_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files/Data.zip"


def list_archive_contents():
    """List all files in the Zenodo archive via HTTP range requests."""
    from remotezip import RemoteZip

    logger.info(f"Connecting to {ZIP_URL}")
    logger.info("Reading archive index (small HTTP request, not full 10 GB)...")

    with RemoteZip(ZIP_URL) as rz:
        all_files = rz.infolist()

    logger.info(f"Archive contains {len(all_files)} entries\n")

    # Categorize files
    csvs = []
    for info in all_files:
        name = info.filename
        size_mb = info.file_size / (1024 * 1024)
        if name.endswith('.csv'):
            csvs.append((name, size_mb))
            logger.info(f"  CSV: {name} ({size_mb:.1f} MB)")
        elif name.endswith('.txt') or name.endswith('.md'):
            logger.info(f"  DOC: {name} ({size_mb:.1f} MB)")

    logger.info(f"\n--- All directories ---")
    dirs = sorted(set(os.path.dirname(f.filename) for f in all_files if f.filename))
    for d in dirs:
        if d:
            logger.info(f"  {d}/")

    return csvs


def find_training_files(csvs):
    """Identify which CSVs are likely training data."""
    training_files = []
    for name, size_mb in csvs:
        basename = os.path.basename(name).lower()
        # Look for train, full dataset, or the main curated dataset
        if any(kw in basename for kw in ['train', 'full', 'all_ligand', 'ligand_data_final.']):
            training_files.append((name, size_mb))
        # Also grab anything in Curated Datasets that's NOT test/holdout
        elif 'curated' in name.lower() and 'test' not in basename and 'holdout' not in basename:
            training_files.append((name, size_mb))

    return training_files


def download_files(output_dir, file_paths):
    """Download specific files from Zenodo archive via HTTP range requests."""
    from remotezip import RemoteZip

    logger.info(f"\nDownloading {len(file_paths)} files...")

    with RemoteZip(ZIP_URL) as rz:
        all_files = rz.namelist()

        for target_path in file_paths:
            if target_path in all_files:
                info = rz.getinfo(target_path)
                size_mb = info.file_size / (1024 * 1024)
                logger.info(f"  Extracting: {target_path} ({size_mb:.1f} MB)")

                data = rz.read(target_path)
                out_name = os.path.basename(target_path)
                out_path = os.path.join(output_dir, out_name)

                with open(out_path, 'wb') as f:
                    f.write(data)
                logger.info(f"    Saved to {out_path}")
            else:
                logger.warning(f"  NOT FOUND: {target_path}")
                # Partial match
                matches = [f for f in all_files if os.path.basename(target_path) in f]
                if matches:
                    logger.info(f"  Possible matches: {matches[:5]}")


def inspect_csv(csv_path):
    """Quick inspection of downloaded CSV."""
    import csv as csv_mod

    with open(csv_path, 'r') as f:
        reader = csv_mod.reader(f)
        header = next(reader)
        n_rows = sum(1 for _ in reader)

    size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    logger.info(f"\n  File: {csv_path}")
    logger.info(f"  Size: {size_mb:.1f} MB")
    logger.info(f"  Columns ({len(header)}): {header}")
    logger.info(f"  Rows: {n_rows}")

    # Show first 3 rows
    with open(csv_path, 'r') as f:
        reader = csv_mod.reader(f)
        next(reader)
        for i, row in enumerate(reader):
            if i >= 3:
                break
            logger.info(f"  Row {i}: {row[:6]}...")


def main():
    parser = argparse.ArgumentParser(
        description='Download Toney et al. training data from Zenodo'
    )
    parser.add_argument('--output_dir', type=str, default='data/toney',
                        help='Output directory')
    parser.add_argument('--list_only', action='store_true',
                        help='Only list archive contents, do not download')
    parser.add_argument('--files', nargs='*', default=None,
                        help='Specific file paths within the archive to download')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Toney et al. (PNAS 2025) Training Data Download")
    logger.info("=" * 60)
    logger.info(f"Zenodo: https://zenodo.org/records/{ZENODO_RECORD}")

    # Step 1: List archive contents
    csvs = list_archive_contents()

    if args.list_only:
        logger.info("\n--list_only mode: not downloading.")
        training = find_training_files(csvs)
        if training:
            logger.info("\nSuggested training files:")
            for name, size_mb in training:
                logger.info(f"  {name} ({size_mb:.1f} MB)")
            logger.info(f"\nRerun with: python {sys.argv[0]} --output_dir {args.output_dir} "
                        f"--files \"{training[0][0]}\"")
        return

    # Step 2: Determine which files to download
    if args.files:
        to_download = args.files
    else:
        training = find_training_files(csvs)
        if not training:
            logger.error("Could not auto-detect training files.")
            logger.error("Rerun with --list_only to see all files, then use --files")
            return
        to_download = [name for name, _ in training]
        logger.info(f"\nAuto-detected training files: {to_download}")

    # Also grab readme if not already present
    readme_path = os.path.join(args.output_dir, 'readme.txt')
    if not os.path.exists(readme_path):
        to_download.append('Data/readme.txt')

    # Step 3: Download
    download_files(args.output_dir, to_download)

    # Step 4: Inspect
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOADED FILES")
    logger.info("=" * 60)
    for f in os.listdir(args.output_dir):
        if f.endswith('.csv'):
            inspect_csv(os.path.join(args.output_dir, f))

    logger.info("\nDone. Next: adapt CSV and merge with stability constants for joint training.")


if __name__ == '__main__':
    main()
