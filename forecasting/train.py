from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from txembed import (
    PreprocessedTransactions,
    TransactionPreprocessor,
    TransactionSequenceDataset,
)
from txforecast import TransactionForecastingModel
from txlabels import (
    TransactionForecastingDataset,
    collate_forecasting_batch,
    load_labels,
)


def run_epoch(
    model: TransactionForecastingModel,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    """One pass over ``loader``. Trains if ``optimizer`` is given, else evaluates."""
    model.train(optimizer is not None)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.transactions)
            loss = loss_fn(logits, batch.labels)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch.labels.shape[0]
            correct += int((logits.argmax(dim=-1) == batch.labels).sum())
            total += batch.labels.shape[0]

    return total_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the recurring-merchant forecasting model"
    )
    parser.add_argument(
        "--artifact-dir",
        required=True,
        type=Path,
        help="txembed prepare_csv.py output directory",
    )
    parser.add_argument("--train-labels", required=True, type=Path)
    parser.add_argument("--valid-labels", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Stop after this many epochs without improvement",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = TransactionPreprocessor.load(args.artifact_dir / "preprocessor.json")
    model = TransactionForecastingModel.from_preprocessor(preprocessor).to(device)

    def make_loader(split: str, labels_path: Path, shuffle: bool) -> DataLoader:
        rows = PreprocessedTransactions.load_npz(args.artifact_dir / f"{split}.npz")
        sequences = TransactionSequenceDataset(rows, max_length=args.max_length)
        dataset = TransactionForecastingDataset(sequences, load_labels(labels_path))
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=collate_forecasting_batch,
        )

    train_loader = make_loader("train", args.train_labels, shuffle=True)
    valid_loader = make_loader("valid", args.valid_labels, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_valid_loss = float("inf")
    epochs_without_improvement = 0
    checkpoint_path = args.output_dir / "model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, loss_fn, device, optimizer
        )
        valid_loss, valid_accuracy = run_epoch(
            model, valid_loader, loss_fn, device, optimizer=None
        )
        print(
            f"epoch {epoch:03d} | train loss {train_loss:.4f} acc {train_accuracy:.3f} | "
            f"valid loss {valid_loss:.4f} acc {valid_accuracy:.3f}"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Stopping early: no improvement for {args.patience} epochs")
                break

    print(
        f"Best validation loss {best_valid_loss:.4f}; checkpoint saved to {checkpoint_path}"
    )


if __name__ == "__main__":
    main()
