# Transaction embedding

## The simple explanation

A model cannot directly understand text such as `coffee shop`, categories such as
`card_payment`, or a date such as Monday. This module turns every transaction into
**128 useful numbers**. Think of those numbers as a compact ID card describing the
transaction. Transactions belonging to one client are then placed in time order so
the next model can study their spending history.

Raw CSV files and generated arrays are intentionally not committed. They live in
ignored folders because they are large and can always be rebuilt.

## What is added for each transaction

| Input | What the code does | Why |
|---|---|---|
| `description` | Frozen `all-MiniLM-L6-v2`: text -> 384 numbers, then a learned `Linear(384, 64)` | Similar descriptions receive similar representations while our small layer adapts them to this task. |
| `candidate_family` | Learned 8d embedding | Compact representation of the candidate recurring family. |
| `mcc` | Learned 8d embedding | Represents the merchant category. |
| `type` | Learned 6d embedding | Represents transaction type. |
| `currency` | Learned 4d embedding | Represents currency. |
| `direction` | Learned 2d embedding | Represents money going in or out. |
| `is_recurring_candidate` | `0.0` or `1.0` | Preserves the upstream boolean signal. |
| Eight skewed numeric columns | `log1p`, then train-only standardization | Stops very large values from dominating. |
| `amount_vs_family_median` | Train-only standardization, without log | Preserves the continuous ratio. |
| Weekday, day of month, month | Sine + cosine pairs | Makes the end and start of a cycle close together; Sunday is near Monday and December is near January. |
| Missing history values | Fill with zero and add seven yes/no missing flags | Lets the model distinguish “no history exists” from a genuine numeric zero. |

All categorical dictionaries reserve index `0` for values that were missing or not
seen during training. The dense section contains 23 values: one boolean, nine
continuous values, six calendar values, and seven missing-history flags.

After concatenation the row has 115 values:

```text
description 64 + categories (8+8+6+4+2) + dense features 23 = 115
```

The final network is:

```text
Linear(115, 128) -> GELU -> LayerNorm -> Dropout(0.1)
-> Linear(128, 128) -> LayerNorm
```

## Leakage protection

Only `train_features.csv` is allowed to create category dictionaries and numeric
mean/standard-deviation values. Validation and test reuse the saved training values.
`client_id` only groups rows and `timestamp` only sorts them; neither is embedded.
The upstream history columns must be calculated using earlier transactions only.

The downstream sequence model must also use a **causal attention mask**. The padding
mask produced here only says which positions are empty; it does not stop a model
from looking at a later real transaction.

## Data location

The supplied archive is kept locally at `data/dataset_features.zip` and extracted to:

```text
data/dataset_features/
  train_features.csv
  valid_features.csv
  test_features.csv
```

Both paths are in `.gitignore`. The CSV includes a `fee` column, but it is ignored
because it was not part of the requested embedding specification.

## Build the real embeddings

Run these commands from `transaction_embedding/`:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'

python prepare_csv.py \
  --train-csv ../data/dataset_features/train_features.csv \
  --valid-csv ../data/dataset_features/valid_features.csv \
  --test-csv ../data/dataset_features/test_features.csv \
  --output-dir artifacts
```

The first run downloads `sentence-transformers/all-MiniLM-L6-v2`. Descriptions are
cached in `artifacts/descriptions.sqlite`, so later runs only encode new text.
`train.npz`, `valid.npz`, and `test.npz` cache the complete processed rows.

The actual run performed on this dataset produced:

| Split | Transactions | Clients | Cached row shape |
|---|---:|---:|---|
| Train | 147,459 | 2,000 | `(147459, 384)`, `(147459, 5)`, `(147459, 23)` |
| Validation | 73,898 | 1,000 | `(73898, 384)`, `(73898, 5)`, `(73898, 23)` |
| Test | 75,761 | 1,000 | `(75761, 384)`, `(75761, 5)`, `(75761, 23)` |

The three shapes are description vectors, categorical indices, and dense features.
Generated artifacts occupy about 216 MB and stay out of Git.

The final 128d vectors are **not** saved permanently because their projection and
category layers are trainable. They are produced by `TransactionEncoder` during
training or inference. Before that model is trained, its 128d output has the right
shape but contains randomly initialized learned projections; only the frozen 384d
description vectors and deterministic preprocessing are meaningful cached inputs.

## Validate everything

```bash
pytest -q
python validate_artifacts.py --artifact-dir artifacts
```

The first command tests preprocessing, unknown categories, cache reuse, artifact
loading, sequence padding, and model output without downloading a model. The second
checks every real row for correct dimensions and finite values, validates categorical
index ranges, and runs each split through the model.

## Use the model output

```python
from torch.utils.data import DataLoader
from txembed import (
    PreprocessedTransactions,
    TransactionEncoder,
    TransactionPreprocessor,
    TransactionSequenceDataset,
    collate_transaction_sequences,
)

preprocessor = TransactionPreprocessor.load("artifacts/preprocessor.json")
rows = PreprocessedTransactions.load_npz("artifacts/train.npz")
dataset = TransactionSequenceDataset(rows, max_length=256)
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_transaction_sequences,
)
model = TransactionEncoder.from_preprocessor(preprocessor)

for batch in loader:
    transaction_vectors, padding_mask = model(batch)
    # transaction_vectors: [B, T, 128]
    # padding_mask: [B, T], True means this position is empty padding
```

The implementation follows the current Context7 documentation for
`/huggingface/sentence-transformers` and `/pytorch/pytorch`.
