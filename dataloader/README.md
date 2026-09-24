# Dataloader

Loads recurring-merchant labels and pairs them with the `[T, 128]` transaction
sequences from [`transaction_embedding/`](../transaction_embedding/README.md), for
[`forecasting/`](../forecasting/README.md).

- `LABEL_NAMES` / `label_to_index`: the fixed 8-class vocabulary (`cloud, gym,
  insurance, mobile, music, software, streaming, none`). A label CSV containing any
  other value fails loudly rather than silently growing the vocabulary.
- `load_labels(path)`: reads a `client_id,target_next_recurring_merchant` CSV into
  `{client_id: label index}`.
- `TransactionForecastingDataset`: pairs a `txembed.TransactionSequenceDataset` with
  those labels, one entry per client. Raises if any client is missing a label.
- `collate_forecasting_batch` / `ForecastingBatch`: batches sequences and labels
  together for training and evaluation.
