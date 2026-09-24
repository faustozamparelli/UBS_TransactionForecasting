from __future__ import annotations

from typing import Any

import torch
from torch import nn
from txembed import TransactionBatch, TransactionEncoder, TransactionPreprocessor
from txlabels import LABEL_NAMES


class TransactionForecastingModel(nn.Module):
    """Predicts a client's next recurring-merchant family from their transaction history.

    A single self-attention layer lets transactions in the sequence inform each
    other, then a learned query attends over the (masked) sequence to produce
    one client-level vector for the classification head. Padding follows the
    encoder's convention: True in ``padding_mask`` means "ignore this position".
    """

    def __init__(
        self, encoder: TransactionEncoder, num_heads: int = 4, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.encoder = encoder
        embed_dim = TransactionEncoder.OUTPUT_DIM

        self.self_attention = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.self_attention_norm = nn.LayerNorm(embed_dim)
        self.self_attention_dropout = nn.Dropout(dropout)

        self.pool_query = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.pool_query)
        self.pool_attention = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.pool_norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(embed_dim, len(LABEL_NAMES))
        )

    @classmethod
    def from_preprocessor(
        cls, preprocessor: TransactionPreprocessor, **kwargs: Any
    ) -> TransactionForecastingModel:
        return cls(TransactionEncoder.from_preprocessor(preprocessor), **kwargs)

    def forward(self, batch: TransactionBatch) -> torch.Tensor:
        embeddings, padding_mask = self.encoder(batch)

        attended, _ = self.self_attention(
            embeddings,
            embeddings,
            embeddings,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        embeddings = self.self_attention_norm(
            embeddings + self.self_attention_dropout(attended)
        )
        # Padded positions must not leak into the pooling query's attention weights.
        embeddings = embeddings.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        query = self.pool_query.expand(embeddings.shape[0], -1, -1)
        pooled, _ = self.pool_attention(
            query,
            embeddings,
            embeddings,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        pooled = self.pool_norm(pooled.squeeze(1))

        return self.classifier(pooled)
