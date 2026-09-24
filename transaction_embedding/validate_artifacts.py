from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from txembed import (
    PreprocessedTransactions,
    TransactionEncoder,
    TransactionPreprocessor,
    TransactionSequenceDataset,
    collate_transaction_sequences,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate cached features and a model forward pass")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    preprocessor = TransactionPreprocessor.load(args.artifact_dir / "preprocessor.json")
    model = TransactionEncoder.from_preprocessor(preprocessor).eval()
    cardinalities = [
        preprocessor.categorical_cardinalities[name] for name in preprocessor.schema.categorical
    ]

    paths = sorted(args.artifact_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {args.artifact_dir}")

    for path in paths:
        rows = PreprocessedTransactions.load_npz(path)
        row_count = len(rows)
        assert rows.description_embeddings.shape == (row_count, 384)
        assert rows.categorical_indices.shape == (row_count, 5)
        assert rows.dense_features.shape == (row_count, preprocessor.dense_dimension)
        assert np.isfinite(rows.description_embeddings).all()
        assert np.isfinite(rows.dense_features).all()
        for column, cardinality in enumerate(cardinalities):
            indices = rows.categorical_indices[:, column]
            assert (indices >= 0).all() and (indices < cardinality).all()

        dataset = TransactionSequenceDataset(rows, max_length=args.max_length)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_transaction_sequences,
        )
        batch = next(iter(loader))
        with torch.inference_mode():
            output, padding_mask = model(batch)

        assert output.shape == (*padding_mask.shape, 128)
        assert padding_mask.dtype == torch.bool
        assert torch.isfinite(output).all()
        assert torch.count_nonzero(output[padding_mask]).item() == 0
        print(
            f"{path.stem}: OK | rows={row_count:,} | clients={len(dataset):,} | "
            f"sample_output={tuple(output.shape)}"
        )


if __name__ == "__main__":
    main()

