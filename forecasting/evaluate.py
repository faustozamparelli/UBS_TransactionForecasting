from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from txembed import (
    PreprocessedTransactions,
    TransactionPreprocessor,
    TransactionSequenceDataset,
)
from txforecast import TransactionForecastingModel
from txlabels import (
    LABEL_NAMES,
    TransactionForecastingDataset,
    collate_forecasting_batch,
    load_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report accuracy and macro F1 for a labeled split"
    )
    parser.add_argument(
        "--artifact-dir",
        required=True,
        type=Path,
        help="txembed prepare_csv.py output directory",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Trained model state dict (train.py's model.pt)",
    )
    parser.add_argument(
        "--split",
        required=True,
        help="npz stem inside --artifact-dir, e.g. train or valid",
    )
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    preprocessor = TransactionPreprocessor.load(args.artifact_dir / "preprocessor.json")
    model = TransactionForecastingModel.from_preprocessor(preprocessor).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    rows = PreprocessedTransactions.load_npz(args.artifact_dir / f"{args.split}.npz")
    sequences = TransactionSequenceDataset(rows, max_length=args.max_length)
    dataset = TransactionForecastingDataset(sequences, load_labels(args.labels))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_forecasting_batch,
    )

    predicted: list[int] = []
    actual: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.transactions)
            predicted.extend(logits.argmax(dim=-1).tolist())
            actual.extend(batch.labels.tolist())

    label_indices = list(range(len(LABEL_NAMES)))
    accuracy = sum(p == a for p, a in zip(predicted, actual)) / len(actual)
    macro_f1 = f1_score(
        actual, predicted, labels=label_indices, average="macro", zero_division=0
    )
    per_class_f1 = f1_score(
        actual, predicted, labels=label_indices, average=None, zero_division=0
    )

    print(
        f"{args.split}: n={len(actual)} accuracy={accuracy:.3f} macro_f1={macro_f1:.3f}"
    )
    for name, score in zip(LABEL_NAMES, per_class_f1):
        print(f"  {name:<10} f1={score:.3f}")


if __name__ == "__main__":
    main()
