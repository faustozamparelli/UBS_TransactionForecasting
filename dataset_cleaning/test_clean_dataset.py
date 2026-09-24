import pandas as pd

from dataset_cleaning.clean_dataset import clean_transactions


def test_cleaner_retains_rows_and_fields_adds_features_and_orders() -> None:
    raw = pd.DataFrame(
        {
            "amount": [10.0, 20.0, 12.0],
            "client_id": ["b", "a", "a"],
            "currency": ["eur", "eur", "eur"],
            "description": ["coffee shop 123", "Cloud Backup ID9", "Cloud Backup ID8"],
            "direction": ["out", "out", "out"],
            "fee": [0.0, 0.1, 0.2],
            "mcc": ["5812", "5732", "5732"],
            "timestamp": [
                "2025-01-02T00:00:00Z",
                "2025-01-03T00:00:00Z",
                "2025-01-01T00:00:00Z",
            ],
            "type": ["card_payment", "card_payment", "card_payment"],
            "_source_order": [0, 1, 2],
        }
    )

    cleaned = clean_transactions(raw)

    assert len(cleaned) == len(raw)
    assert cleaned["client_id"].tolist() == ["a", "a", "b"]
    assert cleaned["fee"].tolist() == [0.2, 0.1, 0.0]
    assert cleaned["clean_description"].tolist() == ["cloud backup", "cloud backup", "coffee shop"]
    assert cleaned["candidate_family"].tolist() == ["cloud", "cloud", "none"]
    assert "is_recurring_candidate" not in cleaned
    assert cleaned.loc[1, "days_since_prev_same_family"] == 2.0
    assert "amount_vs_family_median" in cleaned
