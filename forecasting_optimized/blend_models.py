from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from experiment import pair_probabilities, probabilities_in_label_order
from features import FAMILIES, LABELS, build_feature_tables


def as_distribution(recurring: np.ndarray) -> np.ndarray:
    none = np.clip(1.0 - recurring.max(axis=1, keepdims=True), 1e-6, 1.0)
    probabilities = np.column_stack([recurring, none])
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def macro_f1(actual: np.ndarray, scores: np.ndarray, offsets: np.ndarray | None = None) -> float:
    adjusted = np.log(np.clip(scores, 1e-7, 1.0))
    if offsets is not None:
        adjusted = adjusted + offsets
    return float(
        f1_score(
            actual,
            adjusted.argmax(axis=1),
            labels=np.arange(len(LABELS)),
            average="macro",
            zero_division=0,
        )
    )


def tune_offsets(actual: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, float]:
    offsets = np.zeros(len(LABELS), dtype=np.float64)
    best_score = macro_f1(actual, probabilities, offsets)
    for step in (0.4, 0.2, 0.1, 0.05, 0.025):
        improved = True
        while improved:
            improved = False
            for column in range(len(LABELS)):
                for direction in (-step, step):
                    candidate = offsets.copy()
                    candidate[column] += direction
                    score = macro_f1(actual, probabilities, candidate)
                    if score > best_score + 1e-9:
                        offsets, best_score = candidate, score
                        improved = True
    return offsets, best_score


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Blend temporal and recurrence models")
    parser.add_argument("--artifact-dir", type=Path, default=Path(__file__).parent / "artifacts")
    parser.add_argument(
        "--feature-csv",
        type=Path,
        default=repository / "data" / "dataset_features" / "valid_features.csv",
    )
    parser.add_argument(
        "--labels", type=Path, default=repository / "data" / "dataset" / "valid_labels.csv"
    )
    args = parser.parse_args()

    labels = pd.read_csv(args.labels, dtype={"client_id": str})
    target_map = {name: index for index, name in enumerate(LABELS)}
    target = labels["target_next_recurring_merchant"].map(target_map)
    target_by_client = pd.Series(target.to_numpy(), index=labels["client_id"])

    bundle = joblib.load(args.artifact_dir / "model.joblib")
    frame = pd.read_csv(args.feature_csv)
    wide, pairs = build_feature_tables(frame, labels["client_id"])
    wide = wide.reindex(columns=bundle["wide_columns"])
    pairs = pairs.reindex(columns=bundle["pair_columns"])
    catboost = probabilities_in_label_order(bundle["catboost_multiclass_model"], wide)
    pair = as_distribution(pair_probabilities(bundle["pair_model"], pairs).to_numpy())

    with np.load(args.artifact_dir / "family_valid_probabilities.npz") as family_data:
        family_by_client = {
            str(client_id): probabilities
            for client_id, probabilities in zip(
                family_data["client_ids"], family_data["probabilities"], strict=True
            )
        }
    family = as_distribution(np.stack([family_by_client[client_id] for client_id in wide.index]))

    actual = target_by_client.loc[wide.index].to_numpy()
    weights = {"catboost": 0.314, "pair": 0.601, "family": 0.085}
    probabilities = (
        weights["catboost"] * catboost
        + weights["pair"] * pair
        + weights["family"] * family
    )
    extra_weights = {
        "stacking": 0.272,
        "proxy_catboost": 0.181,
        "proxy_pair": 0.151,
        "xgboost_pair": 0.296,
        "ranking": 0.203,
        "embedding_pool": 0.479,
        "description_stream": 0.161,
    }
    extra_sources: dict[str, np.ndarray] = {}
    stacking_path = args.artifact_dir / "stacking_valid_probabilities.npz"
    if stacking_path.exists():
        with np.load(stacking_path) as data:
            by_client = {
                str(client_id): values
                for client_id, values in zip(
                    data["client_ids"], data["probabilities"], strict=True
                )
            }
        extra_sources["stacking"] = np.stack([by_client[client_id] for client_id in wide.index])
    proxy_path = args.artifact_dir / "proxy_valid_probabilities.npz"
    if proxy_path.exists():
        with np.load(proxy_path) as data:
            proxy_order = [str(value) for value in data["client_ids"]]
            positions = {client_id: index for index, client_id in enumerate(proxy_order)}
            extra_sources["proxy_catboost"] = np.stack(
                [data["catboost"][positions[client_id]] for client_id in wide.index]
            )
            extra_sources["proxy_pair"] = np.stack(
                [data["pair"][positions[client_id]] for client_id in wide.index]
            )
    xgboost_path = args.artifact_dir / "xgboost_valid_probabilities.npz"
    if xgboost_path.exists():
        with np.load(xgboost_path) as data:
            xgboost_by_client = {
                str(client_id): values
                for client_id, values in zip(
                    data["client_ids"], data["probabilities"], strict=True
                )
            }
        extra_sources["xgboost_pair"] = np.stack(
            [xgboost_by_client[client_id] for client_id in wide.index]
        )
    ranking_path = args.artifact_dir / "ranking_valid_probabilities.npz"
    if ranking_path.exists():
        with np.load(ranking_path) as data:
            ranking_by_client = {
                str(client_id): values
                for client_id, values in zip(
                    data["client_ids"], data["probabilities"], strict=True
                )
            }
        extra_sources["ranking"] = np.stack(
            [ranking_by_client[client_id] for client_id in wide.index]
        )
    embedding_path = args.artifact_dir / "embedding_pool_valid_probabilities.npz"
    if embedding_path.exists():
        with np.load(embedding_path) as data:
            embedding_by_client = {
                str(client_id): values
                for client_id, values in zip(
                    data["client_ids"], data["probabilities"], strict=True
                )
            }
        extra_sources["embedding_pool"] = np.stack(
            [embedding_by_client[client_id] for client_id in wide.index]
        )
    description_path = args.artifact_dir / "description_stream_valid_probabilities.npz"
    if description_path.exists():
        with np.load(description_path) as data:
            description_by_client = {
                str(client_id): values
                for client_id, values in zip(
                    data["client_ids"], data["probabilities"], strict=True
                )
            }
        extra_sources["description_stream"] = np.stack(
            [description_by_client[client_id] for client_id in wide.index]
        )

    missing_sources = sorted(set(extra_weights) - set(extra_sources))
    if missing_sources:
        raise FileNotFoundError(
            f"Missing validation probabilities for frozen sources: {missing_sources}"
        )
    for source_name, source_weight in extra_weights.items():
        probabilities = (
            (1 - source_weight) * probabilities
            + source_weight * extra_sources[source_name]
        )

    # Fixed class-specific adjustments chosen during development. They are
    # applied here as configured; negative values counter systematic errors.
    proposed_adjustments = {
        "stacking": {"gym": 0.10, "music": 0.125, "software": 0.40, "none": 0.225},
        "xgboost_pair": {
            "cloud": -0.15,
            "gym": 0.35,
            "insurance": -0.125,
            "mobile": -0.025,
            "software": -0.20,
            "streaming": -0.025,
        },
        "ranking": {"mobile": 0.025, "streaming": 0.05},
        "embedding_pool": {"cloud": 0.075, "mobile": 0.175, "streaming": 0.475},
        "description_stream": {"streaming": 0.05},
    }
    class_adjustments: list[dict[str, float | str]] = []
    for source_name, adjustments in proposed_adjustments.items():
        if source_name not in extra_sources:
            continue
        source_probabilities = extra_sources[source_name]
        for label, weight in adjustments.items():
            column = LABELS.index(label)
            probabilities[:, column] = (
                (1 - weight) * probabilities[:, column]
                + weight * source_probabilities[:, column]
            )
            probabilities = np.clip(probabilities, 1e-7, None)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            class_adjustments.append(
                {"source": source_name, "label": label, "weight": weight}
            )

    # Local calibration of the saved validation probability tables. Base models
    # are unchanged; this choice was tuned on validation and is not a test score.
    offsets = np.asarray([0.45, 0.397, 0.411, 0.435, 0.667, -0.007, 0.534, -0.4])
    predicted = (np.log(np.clip(probabilities, 1e-7, 1.0)) + offsets).argmax(axis=1)
    score = macro_f1(actual, probabilities, offsets)
    accuracy = accuracy_score(actual, predicted)
    print(f"best blended validation macro_f1={score:.4f}")
    print(f"validation accuracy={accuracy:.4f}")
    print(f"weights={weights}")
    print(f"extra sequential blend weights={extra_weights}")
    print(f"class-specific adjustments={class_adjustments}")
    print(f"class offsets={dict(zip(LABELS, offsets.round(3), strict=True))}")
    print(
        classification_report(
            actual,
            predicted,
            labels=np.arange(len(LABELS)),
            target_names=LABELS,
            digits=3,
            zero_division=0,
        )
    )
    configuration = {
        "macro_f1": score,
        "accuracy": accuracy,
        "weights": weights,
        "class_offsets": dict(zip(LABELS, offsets.tolist(), strict=True)),
        "extra_weights": extra_weights,
        "class_adjustments": class_adjustments,
    }
    (args.artifact_dir / "blend_config.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "client_id": wide.index,
            "actual": np.asarray(LABELS)[actual],
            "predicted": np.asarray(LABELS)[predicted],
        }
    ).to_csv(args.artifact_dir / "blended_validation_predictions.csv", index=False)
    np.savez_compressed(
        args.artifact_dir / "blended_valid_probabilities.npz",
        client_ids=np.asarray(wide.index, dtype=str),
        probabilities=probabilities,
        offsets=offsets,
    )


if __name__ == "__main__":
    main()
