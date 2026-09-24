# Dataset cleaning

`clean_dataset.py` is the canonical raw-data cleaning and feature-engineering step.
It reads the three transaction JSONL splits from `data/dataset/`, retains every
original transaction and field, adds normalized text plus calendar and history
features, and orders rows by client and timestamp.

From the repository root, run:

```bash
python dataset_cleaning/clean_dataset.py
```

This writes the extracted CSVs to `data/dataset_features/` and rebuilds
`data/dataset_features.zip`. The archive also includes the train/validation labels
and sample submission file from the raw dataset directory.
