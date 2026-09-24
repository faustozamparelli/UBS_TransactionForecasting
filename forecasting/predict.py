from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from txembed import (
    PreprocessedTransactions,
    TransactionPreprocessor,
    TransactionSequenceDataset,
    collate_transaction_sequences,
)
from txforecast import TransactionForecastingModel
from txlabels import LABEL_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict the next recurring merchant and write a submission CSV"
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
        "--split", default="test", help="npz stem inside --artifact-dir to predict on"
    )
    parser.add_argument(
        "--submission-template",
        required=True,
        type=Path,
        help="CSV whose client_id rows/order to fill in",
    )
    parser.add_argument("--output-csv", required=True, type=Path)
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
    loader = DataLoader(
        sequences,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_transaction_sequences,
    )

    predictions: dict[str, str] = {}
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            labels = [LABEL_NAMES[index] for index in logits.argmax(dim=-1).tolist()]
            batch_client_ids = sequences.client_ids[offset : offset + len(labels)]
            for client_id, label in zip(batch_client_ids, labels):
                predictions[client_id] = label
            offset += len(labels)

    template = pd.read_csv(args.submission_template)
    client_column, label_column = template.columns[0], template.columns[1]
    missing = set(template[client_column].astype(str)) - set(predictions)
    if missing:
        raise ValueError(
            f"No prediction for {len(missing)} client(s) in template, e.g. {sorted(missing)[:5]}"
        )

    template[label_column] = template[client_column].astype(str).map(predictions)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(template)} predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
