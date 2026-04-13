"""
This script reorganizes the PAD-UFES-20 dataset from a downloaded archive into
class subfolders required for training with ImageDataGenerator.

Example usage:
  python reorganize_dataset.py --archive "C:\\...\\archive (1)" --output "dataset_organized"
"""
import argparse
import os
import shutil
import pandas as pd
from pathlib import Path

# Full disease names for each code
DISEASE_NAMES = {
    "NEV": "Melanocytic_Nevi",
    "BCC": "Basal_Cell_Carcinoma",
    "ACK": "Actinic_Keratosis",
    "SEK": "Seborrheic_Keratosis",
    "SCC": "Squamous_Cell_Carcinoma",
    "MEL": "Melanoma",
}


def main():
    parser = argparse.ArgumentParser(description='Reorganize PAD-UFES-20 dataset for training.')
    parser.add_argument('--archive', type=str, required=True, help='Path to the archive root folder containing imgs_part_* and metadata.csv')
    parser.add_argument('--output', type=str, default='dataset_organized', help='Output directory for reorganized dataset')
    args = parser.parse_args()

    archive_dir = Path(args.archive)
    if not archive_dir.exists():
        raise FileNotFoundError(f"Archive path not found: {archive_dir}")

    metadata_csv = archive_dir / 'metadata.csv'
    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata.csv not found in archive folder: {metadata_csv}")

    img_dirs = [
        archive_dir / 'imgs_part_1' / 'imgs_part_1',
        archive_dir / 'imgs_part_2' / 'imgs_part_2',
        archive_dir / 'imgs_part_3' / 'imgs_part_3',
    ]

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading metadata CSV from {metadata_csv}...")
    df = pd.read_csv(metadata_csv)
    print(f"Total records: {len(df)}")
    print(f"Diagnoses: {df['diagnostic'].unique()}")

    img_to_diag = dict(zip(df['img_id'], df['diagnostic']))

    for folder_name in DISEASE_NAMES.values():
        (output_dir / folder_name).mkdir(parents=True, exist_ok=True)

    copied = 0
    not_found = 0

    for img_dir in img_dirs:
        if not img_dir.exists():
            print(f"Skipping missing source folder: {img_dir}")
            continue
        for fname in os.listdir(img_dir):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            diag_code = img_to_diag.get(fname)
            if diag_code is None:
                not_found += 1
                continue
            folder_name = DISEASE_NAMES.get(diag_code)
            if folder_name is None:
                not_found += 1
                continue
            src = img_dir / fname
            dst = output_dir / folder_name / fname
            shutil.copy2(src, dst)
            copied += 1

    print(f"\nDone! Copied {copied} images to '{output_dir}'")
    print(f"Skipped/not matched: {not_found}")
    print("\nClass distribution:")
    for folder_name in sorted(os.listdir(output_dir)):
        count = len(os.listdir(output_dir / folder_name))
        print(f"  {folder_name}: {count} images")


if __name__ == '__main__':
    main()
