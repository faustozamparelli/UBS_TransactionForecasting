from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from txembed import (
    TransactionBatch,
    TransactionSequenceDataset,
    collate_transaction_sequences,
)
from txembed.batching import TransactionSequence


@dataclass
class ForecastingBatch:
    transactions: TransactionBatch
    labels: torch.Tensor  # [B], long

    def to(self, device: torch.device | str) -> ForecastingBatch:
        return ForecastingBatch(self.transactions.to(device), self.labels.to(device))


class TransactionForecastingDataset(Dataset[tuple[TransactionSequence, int]]):
    """Pairs each client's transaction sequence with its recurring-merchant label."""

    def __init__(
        self, sequences: TransactionSequenceDataset, labels: dict[str, int]
    ) -> None:
        missing = [
            client_id for client_id in sequences.client_ids if client_id not in labels
        ]
        if missing:
            raise ValueError(
                f"No label for {len(missing)} client(s), e.g. {missing[:5]}"
            )
        self.sequences = sequences
        self.labels = [labels[client_id] for client_id in sequences.client_ids]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[TransactionSequence, int]:
        return self.sequences[index], self.labels[index]


def collate_forecasting_batch(
    items: list[tuple[TransactionSequence, int]],
) -> ForecastingBatch:
    sequences, labels = zip(*items, strict=True)
    return ForecastingBatch(
        transactions=collate_transaction_sequences(list(sequences)),
        labels=torch.tensor(labels, dtype=torch.long),
    )
