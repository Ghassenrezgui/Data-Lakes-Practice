import argparse
from pathlib import Path

import pandas as pd


def unpack_data(input_dir: str, output_file: str) -> None:
    """
    Combine multiple CSV files from a directory into a single CSV file.
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)

    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"Input directory does not exist: {input_path}")

    # 1) Récupérer tous les fichiers à lire :
    # - si input_dir contient des sous-dossiers (train/dev/test), on prend tous les fichiers récursivement
    # - sinon on prend juste les fichiers du dossier
    if any(p.is_dir() for p in input_path.iterdir()):
        files = sorted([p for p in input_path.rglob("*") if p.is_file()])
    else:
        files = sorted([p for p in input_path.iterdir() if p.is_file()])

    if not files:
        raise ValueError(f"No files found in {input_path}")

    # 2) Lire et concaténer
    dataframes = []
    for file_path in files:
        # optionnel: si tu veux vraiment filtrer .csv uniquement, décommente:
        # if file_path.suffix.lower() != ".csv":
        #     continue
        dataframes.append(pd.read_csv(file_path))

    if not dataframes:
        raise ValueError(f"No readable CSV files found in {input_path}")

    combined_df = pd.concat(dataframes, ignore_index=True)

    # 3) Sauvegarder
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unpack and combine CSV files.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    unpack_data(args.input_dir, args.output_file)
