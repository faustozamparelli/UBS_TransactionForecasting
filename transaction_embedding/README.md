# Transaction embedding

This directory converts the feature CSVs into ordered transaction sequences ready
for a future forecasting or anomaly-detection model. That later model is not part of
this implementation.

## What goes into one transaction

- **Description:** the cleaner's `clean_description` is passed to frozen
  `all-MiniLM-L6-v2`, producing a cached 384d text vector; a trainable layer reduces
  it to 64d.
- **Categories:** learned embeddings for recurring family, MCC, transaction type,
  currency, and direction.
- **Numbers:** amount and history values use `log1p` where appropriate, then training
  mean/std normalization.
- **Time:** weekday, day of month, and month use sine/cosine values so calendar cycles
  wrap around naturally.
- **Missing history:** unavailable history is filled with zero and paired with a flag
  so the model can tell “unknown” apart from a real zero.

The features are concatenated and projected to one 128d vector. `client_id` is never
embedded; it is used only to group transactions. `timestamp` is used only to sort
each client from oldest to newest.

```text
one transaction -> [128]
one client with T transactions -> [T, 128]
one padded client batch -> [B, T, 128] + padding mask
```

Shorter clients are padded inside a batch. `True` in the padding mask means “ignore
this empty position.” Shuffling a data loader shuffles clients, never the transaction
order inside a client. By default the full history is kept; setting `max_length`
keeps only the most recent transactions while preserving chronological order.

## Leakage protection

Category dictionaries and normalization statistics are fitted from train only and
saved in `preprocessor.json`. Validation and test can only reuse them. The supplied
splits have no clients in common.

## Build and validate

First rebuild the cleaned dataset from the repository root:

```bash
python dataset_cleaning/clean_dataset.py
```

Then, from this directory:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'

python prepare_csv.py

pytest -q
python validate_artifacts.py --artifact-dir artifacts
```

Descriptions are cached in SQLite and processed arrays are cached as compressed
`.npz` files. Rows in every artifact are deterministically ordered by client and
timestamp (with original row order breaking timestamp ties). The generated
`artifacts/` directory is ignored by Git.

The current dataset produces:

| Split | Transactions | Clients |
|---|---:|---:|
| Train | 147,459 | 2,000 |
| Validation | 73,898 | 1,000 |
| Test | 75,761 | 1,000 |

## Boundary of this work

This module prepares features, embeddings, chronological client sequences, and a
padding mask. It does not yet train the sequence model, create phishing alerts, or
forecast the next transaction. Those are the next stage of the project.
