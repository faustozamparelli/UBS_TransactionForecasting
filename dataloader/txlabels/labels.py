from __future__ import annotations

from pathlib import Path

import pandas as pd

# Fixed by the UBS task definition, not inferred from whatever appears in a given
# label file: a new value in a label CSV should fail loudly, not silently grow
# the vocabulary.
LABEL_NAMES: tuple[str, ...] = (
    "cloud",
    "gym",
    "insurance",
    "mobile",
    "music",
    "software",
    "streaming",
    "none",
)
LABEL_TO_INDEX: dict[str, int] = {name: index for index, name in enumerate(LABEL_NAMES)}


def label_to_index(label: str) -> int:
    try:
        return LABEL_TO_INDEX[label]
    except KeyError as exc:
        raise ValueError(
            f"Unknown recurring-merchant label {label!r}; expected one of {LABEL_NAMES}"
        ) from exc


def load_labels(
    path: str | Path,
    client_column: str = "client_id",
    label_column: str = "target_next_recurring_merchant",
) -> dict[str, int]:
    """Reads a client_id -> recurring-merchant label CSV into {client_id: label index}."""
    frame = pd.read_csv(path)
    missing_columns = {client_column, label_column} - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{path}: missing columns {sorted(missing_columns)}")
    return {
        str(client_id): label_to_index(str(label).strip())
        for client_id, label in zip(
            frame[client_column], frame[label_column], strict=True
        )
    }
