# Forecasting

Predicts each client's next recurring-merchant family from their transaction history,
using the `[T, 128]` sequences produced by [`transaction_embedding/`](../transaction_embedding/README.md).
Labels come from [`dataloader/`](../dataloader/README.md).

## Target

One of 8 classes, fixed by the task definition (not inferred from a label file):

```text
cloud, gym, insurance, mobile, music, software, streaming, none
```

`none` means the client has no predicted next recurring merchant.

## Model

`TransactionForecastingModel` (`txforecast/model.py`) wraps the transaction encoder
with a small attention head:

```text
[T, 128] client sequence + padding mask
        ↓
TransactionEncoder (txembed)              -- unchanged, already trained embedding
        ↓
single masked self-attention layer         -- lets transactions inform each other
        ↓
residual + LayerNorm + dropout
        ↓
learned-query attention pooling            -- one query vector attends over the
        ↓                                     sequence, ignoring padded positions
[128] client vector
        ↓
Linear(128, 8)
        ↓
logits over the 8 recurring-merchant classes
```

Padding follows the encoder's convention: `True` in the mask means "ignore this
position." A learned query (rather than mean pooling) lets the model weigh
transactions unevenly, e.g. favor ones that look like clear recurring payments.

## Process

1. Build embedding artifacts (`transaction_embedding/generate_*_embeddings.py`) —
   see that directory's README.
2. Train:

   ```bash
   python train.py \
     --artifact-dir ../transaction_embedding/artifacts/uncleaned \
     --train-labels ../data/dataset/train_labels.csv \
     --valid-labels ../data/dataset/valid_labels.csv \
     --output-dir artifacts_uncleaned
   ```

   Adam + cross-entropy, early stopping on validation loss (`--patience`, default 5
   epochs without improvement), checkpoint saved to `OUTPUT_DIR/model.pt` whenever
   validation loss improves.

3. Evaluate accuracy and macro F1 on a labeled split:

   ```bash
   python evaluate.py \
     --artifact-dir ../transaction_embedding/artifacts/uncleaned \
     --checkpoint artifacts_uncleaned/model.pt \
     --split valid \
     --labels ../data/dataset/valid_labels.csv
   ```

4. Predict on an unlabeled split and write a submission CSV in the same
   `client_id,predicted_next_recurring_merchant` format as `sample_submission.csv`:

   ```bash
   python predict.py \
     --artifact-dir ../transaction_embedding/artifacts/uncleaned \
     --checkpoint artifacts_uncleaned/model.pt \
     --split test \
     --submission-template ../data/dataset/sample_submission.csv \
     --output-csv ../data/dataset/submission.csv
   ```

## Results

Same model and training run, only the embedded description text differs
(`transaction_embedding/generate_cleaned_embeddings.py` vs
`generate_uncleaned_embeddings.py`):

| Description | Split | accuracy | macro F1 |
|---|---|---:|---:|
| cleaned (`clean_description`) | train | 0.543 | 0.495 |
| cleaned | valid | 0.377 | 0.254 |
| uncleaned (`description`) | train | 0.578 | 0.553 |
| **uncleaned** | **valid** | **0.503** | **0.440** |

The raw, uncleaned description text outperforms the cleaned one on every class. This
is the opposite of what `dataset_cleaning`'s text normalization was intended to
achieve, and is worth investigating separately. The current checkpoint and submission
file use the **uncleaned** embeddings.

`music` is the weakest class in both variants (valid F1 0.04 cleaned / 0.18
uncleaned) — there are few `music` examples in training.

## Boundary of this work

This covers the classification model, training, evaluation, and submission
generation. It does not cover anomaly detection or phishing alerting, which
[`README.md`](../README.md) lists as later goals.
