from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from txembed import DescriptionEmbeddingCache, TransactionPreprocessor


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Fit train-only preprocessing and cache CSV features")
    cleaned = repository / "data" / "dataset_features"
    parser.add_argument("--train-csv", type=Path, default=cleaned / "train_features.csv")
    parser.add_argument("--valid-csv", type=Path, default=cleaned / "valid_features.csv")
    parser.add_argument("--test-csv", type=Path, default=cleaned / "test_features.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "artifacts")
    parser.add_argument("--cache", type=Path, help="Defaults to OUTPUT_DIR/descriptions.sqlite")
    parser.add_argument("--device", help="Sentence Transformer device, e.g. cpu, cuda, or mps")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = DescriptionEmbeddingCache(
        args.cache or args.output_dir / "descriptions.sqlite", device=args.device
    )
    preprocessor = TransactionPreprocessor()

    train = pd.read_csv(args.train_csv)
    train_features = preprocessor.fit_transform(train, cache)
    preprocessor.save(args.output_dir / "preprocessor.json")
    train_features.save_npz(args.output_dir / "train.npz")

    for split, path in (("valid", args.valid_csv), ("test", args.test_csv)):
        if path is not None:
            preprocessor.transform(pd.read_csv(path), cache).save_npz(args.output_dir / f"{split}.npz")


if __name__ == "__main__":
    main()
