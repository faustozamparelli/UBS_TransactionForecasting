from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from txembed import (
    DescriptionEmbeddingCache,
    TransactionEncoder,
    TransactionPreprocessor,
    TransactionSequenceDataset,
    collate_transaction_sequences,
)


class FakeEncoder:
    def __init__(self):
        self.calls = 0

    def eval(self):
        return self

    def parameters(self):
        return []

    def encode(self, texts, **_kwargs):
        self.calls += 1
        vectors = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
            vectors.append(np.random.default_rng(seed).standard_normal(384))
        return np.asarray(vectors, dtype=np.float32)


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "client_id": ["a", "b", "a"],
            "timestamp": ["2025-01-01T11:00:00Z", "2025-01-02T10:00:00Z", "2025-01-01T10:00:00Z"],
            "amount": [10.0, 20.0, 12.0],
            "mcc": [5411, 5812, 5411],
            "description": ["Market", "Coffee", "Market"],
            "candidate_family": ["groceries", "coffee", "groceries"],
            "is_recurring_candidate": [False, True, False],
            "type": ["card", "card", "card"],
            "direction": ["debit", "debit", "debit"],
            "currency": ["EUR", "EUR", "EUR"],
            "day_of_week": [2, 3, 2],
            "day_of_month": [1, 2, 1],
            "month": [1, 1, 1],
            "days_to_cutoff": [30, 29, 30],
            "days_since_prev_transaction": [1, np.nan, np.nan],
            "days_since_prev_same_family": [3, np.nan, np.nan],
            "count_same_family_before": [2, 0, 0],
            "median_interval_same_family": [3, np.nan, np.nan],
            "std_interval_same_family": [0.5, np.nan, np.nan],
            "median_amount_same_family": [11, np.nan, np.nan],
            "amount_vs_family_median": [0.91, np.nan, np.nan],
        }
    )


def test_end_to_end_shape_mask_unknown_and_artifact_roundtrip(tmp_path):
    fake_encoder = FakeEncoder()
    cache = DescriptionEmbeddingCache(
        tmp_path / "descriptions.sqlite", encoder_factory=lambda _name: fake_encoder
    )
    preprocessor = TransactionPreprocessor().fit(frame())
    artifact = tmp_path / "preprocessor.json"
    preprocessor.save(artifact)
    preprocessor = TransactionPreprocessor.load(artifact)

    validation = frame().copy()
    validation.loc[0, "currency"] = "GBP"  # unseen category -> reserved index zero
    features = preprocessor.transform(validation, cache)
    cached_market = cache.encode(["Market"])
    assert fake_encoder.calls == 1  # the second request came from SQLite
    assert np.array_equal(cached_market[0], features.description_embeddings[0])
    assert features.categorical_indices[0, 3] == 0
    assert features.dense_features.shape == (3, 23)

    features_path = tmp_path / "features.npz"
    features.save_npz(features_path)
    loaded_features = type(features).load_npz(features_path)
    assert np.array_equal(loaded_features.dense_features, features.dense_features)

    dataset = TransactionSequenceDataset(loaded_features)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_transaction_sequences)
    batch = next(iter(loader))
    model = TransactionEncoder.from_preprocessor(preprocessor).eval()
    with torch.inference_mode():
        embedded, padding_mask = model(batch)

    assert embedded.shape == (2, 2, 128)
    assert padding_mask.shape == (2, 2)
    assert padding_mask.sum().item() == 1
    assert torch.count_nonzero(embedded[padding_mask]).item() == 0
