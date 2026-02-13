import argparse
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_data(data_file: str, output_dir: str) -> None:
    """
    Preprocess raw protein sequence data for model training.

    Loads the raw data, cleans it, encodes labels, and splits it into
    train/validation/test sets with a custom strategy that handles
    extreme class imbalance (including classes with 1 sample).
    """
    data_path = Path(data_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1) Load
    df = pd.read_csv(data_path)

    # 2) Remove rows with missing values
    df = df.dropna()

    # 3) Encode family_accession -> integer label
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["family_accession"].astype(str))

    # 4) Custom split per class (robust to rare classes)
    rng = 42  # fixed seed for reproducibility

    train_parts = []
    val_parts = []
    test_parts = []

    # We split inside each class to guarantee no stratify errors
    for label, g in df.groupby("label"):
        g = g.sample(frac=1.0, random_state=rng).reset_index(drop=True)  # shuffle
        n = len(g)

        # Rules to handle small classes safely:
        if n == 1:
            # impossible to have in val/test without losing it from train
            train_parts.append(g)
            continue

        if n == 2:
            # put 1 in train, 1 in val (test will miss this class)
            train_parts.append(g.iloc[:1])
            val_parts.append(g.iloc[1:2])
            continue

        if n == 3:
            # 1 train, 1 val, 1 test
            train_parts.append(g.iloc[:1])
            val_parts.append(g.iloc[1:2])
            test_parts.append(g.iloc[2:3])
            continue

        if n == 4:
            # 2 train, 1 val, 1 test
            train_parts.append(g.iloc[:2])
            val_parts.append(g.iloc[2:3])
            test_parts.append(g.iloc[3:4])
            continue

        # n >= 5: approximate 80/10/10 but ensure at least 1 in val and test
        test_n = max(1, int(round(n * 0.10)))
        val_n = max(1, int(round(n * 0.10)))

        # Ensure we keep at least 1 for train
        if val_n + test_n > n - 1:
            # shrink val/test to fit
            overflow = (val_n + test_n) - (n - 1)
            # reduce val first, then test, but keep >=1
            while overflow > 0 and val_n > 1:
                val_n -= 1
                overflow -= 1
            while overflow > 0 and test_n > 1:
                test_n -= 1
                overflow -= 1

        train_n = n - val_n - test_n

        train_parts.append(g.iloc[:train_n])
        val_parts.append(g.iloc[train_n:train_n + val_n])
        test_parts.append(g.iloc[train_n + val_n:])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else df.iloc[:0].copy()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else df.iloc[:0].copy()

    # Shuffle final splits (optional but nice)
    train_df = train_df.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    val_df = val_df.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=rng).reset_index(drop=True)

    # 5) Save files
    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)

    # Report
    print("Preprocessing done.")
    print(f"Input rows (after dropna): {len(df):,}")
    print(f"Saved train rows: {len(train_df):,}  labels: {train_df['label'].nunique():,}")
    print(f"Saved val rows:   {len(val_df):,}    labels: {val_df['label'].nunique():,}")
    print(f"Saved test rows:  {len(test_df):,}   labels: {test_df['label'].nunique():,}")
    # How many labels are missing from val/test (expected for rare classes)
    missing_val = train_df['label'].nunique() - val_df['label'].nunique()
    missing_test = train_df['label'].nunique() - test_df['label'].nunique()
    print(f"Labels missing from val vs train: {missing_val}")
    print(f"Labels missing from test vs train: {missing_test}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Pfam data.")
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()
    preprocess_data(args.data_file, args.output_dir)