from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler


def load_detection_mat(
    dataset_path: str | Path,
    pca_num: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load a same-source HSI detection MAT file and append full-image PCA.

    The file must contain ``data`` with shape ``(H, W, C)`` and ``map`` with
    shape ``(H, W)``. Background pixels are labeled 0 and target pixels are
    labeled 1. PCA is fitted on all spectra from the same image; no labels are
    used during normalization or PCA fitting.
    """
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Detection dataset does not exist: {path}")
    if pca_num < 1:
        raise ValueError("pca_num must be positive")

    mat = scipy.io.loadmat(str(path))
    missing = {"data", "map"} - set(mat)
    if missing:
        raise KeyError(f"Detection MAT file is missing variables: {sorted(missing)}")

    original = np.asarray(mat["data"], dtype=np.float32)
    target_map = np.asarray(mat["map"])
    if original.ndim != 3:
        raise ValueError(f"data must have shape (H, W, C), got {original.shape}")
    if target_map.shape != original.shape[:2]:
        raise ValueError(
            "map spatial shape must match data, got "
            f"{target_map.shape} and {original.shape[:2]}"
        )
    if not np.isfinite(original).all():
        raise ValueError("data contains NaN or infinite spectra")

    unique_labels = np.unique(target_map)
    if not np.isin(unique_labels, [0, 1]).all() or len(unique_labels) != 2:
        raise ValueError(
            "map must contain both 0 and 1 and no other labels, got "
            f"{unique_labels.tolist()}"
        )

    ground_truth = target_map.astype(np.int64, copy=True)
    image = original.copy()
    image -= image.min()
    image /= max(float(image.max()), np.finfo(np.float32).eps)
    image -= image.mean(axis=(0, 1), keepdims=True)

    pixels = image.reshape(-1, image.shape[-1])
    maximum_components = min(pixels.shape)
    if pca_num > maximum_components:
        raise ValueError(
            f"pca_num must be no greater than {maximum_components}, got {pca_num}"
        )
    pca = PCA(n_components=pca_num, whiten=True)
    pca_features = pca.fit_transform(pixels)
    pca_features = pca_features.reshape(image.shape[:2] + (pca_num,)).astype(
        np.float32, copy=False
    )
    processed_image = np.concatenate((image, pca_features), axis=-1)

    return original, processed_image, ground_truth, ["background", "target"]


def split_detection_gt(
    ground_truth: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create stratified train/validation/test masks from the same image."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be smaller than 1.0")

    labels = ground_truth.ravel()
    indices = np.flatnonzero((labels == 0) | (labels == 1))
    values = labels[indices]
    train_indices, remaining_indices, _, remaining_values = train_test_split(
        indices,
        values,
        train_size=train_ratio,
        random_state=seed,
        stratify=values,
    )
    validation_fraction = val_ratio / (1.0 - train_ratio)
    val_indices, test_indices = train_test_split(
        remaining_indices,
        train_size=validation_fraction,
        random_state=seed,
        stratify=remaining_values,
    )

    shape = ground_truth.shape
    train_gt = np.full(shape, -1, dtype=np.int64)
    validation_gt = np.full(shape, -1, dtype=np.int64)
    test_gt = np.full(shape, -1, dtype=np.int64)
    train_gt.ravel()[train_indices] = labels[train_indices]
    validation_gt.ravel()[val_indices] = labels[val_indices]
    test_gt.ravel()[test_indices] = labels[test_indices]
    return train_gt, validation_gt, test_gt


def class_weights_for_labels(labels: Iterable[int]) -> torch.Tensor:
    """Return inverse-frequency class weights normalized to sum to two."""
    values = np.asarray(labels, dtype=np.int64)
    values = values[(values == 0) | (values == 1)]
    if values.size == 0:
        raise ValueError("labels must contain valid 0/1 samples")

    counts = np.bincount(values, minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(
            f"both background and target are required, got counts {counts.tolist()}"
        )
    inverse_counts = 1.0 / counts
    weights = 2.0 * inverse_counts / inverse_counts.sum()
    return torch.tensor(weights, dtype=torch.float32)


def make_balanced_sampler(
    labels: Iterable[int],
    seed: int = 0,
) -> WeightedRandomSampler:
    """Sample background and target pixels with equal probability per epoch."""
    values = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(values, minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(
            f"both classes are required for balanced sampling, got counts {counts.tolist()}"
        )

    sample_weights = 1.0 / counts[values]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(values),
        replacement=True,
        generator=generator,
    )


def detection_metrics(
    scores: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | list[list[int]]]:
    """Compute thresholded and threshold-free binary detection metrics."""
    score_values = np.asarray(scores, dtype=np.float64).ravel()
    target_values = np.asarray(targets, dtype=np.int64).ravel()
    if score_values.shape != target_values.shape:
        raise ValueError(
            "score and target shapes differ: "
            f"{score_values.shape}, {target_values.shape}"
        )
    valid = (target_values == 0) | (target_values == 1)
    score_values = score_values[valid]
    target_values = target_values[valid]
    if len(np.unique(target_values)) != 2:
        raise ValueError("detection metrics require both background and target samples")

    predictions = (score_values >= threshold).astype(np.int64)
    true_positive = int(np.sum((predictions == 1) & (target_values == 1)))
    false_positive = int(np.sum((predictions == 1) & (target_values == 0)))
    true_negative = int(np.sum((predictions == 0) & (target_values == 0)))
    false_negative = int(np.sum((predictions == 0) & (target_values == 1)))

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    probability_detection = true_positive / (true_positive + false_negative)
    false_alarm_rate = false_positive / (false_positive + true_negative)
    f1 = 2.0 * precision * probability_detection / (precision + probability_detection) if precision + probability_detection else 0.0

    order = np.argsort(-score_values, kind="stable")
    sorted_targets = target_values[order]
    cumulative_tp = np.cumsum(sorted_targets == 1)
    cumulative_fp = np.cumsum(sorted_targets == 0)
    background_count = float(np.sum(target_values == 0))
    target_count = float(np.sum(target_values == 1))
    false_alarms = cumulative_fp / background_count
    detections = cumulative_tp / target_count

    operating_points: dict[str, float] = {}
    for far_limit, name in ((1e-2, "1e-2"), (1e-3, "1e-3"), (1e-4, "1e-4")):
        valid_points = false_alarms <= far_limit + 1e-15
        operating_points[f"pd_at_far_{name}"] = (
            float(np.max(detections[valid_points])) if np.any(valid_points) else 0.0
        )

    return {
        "threshold": float(threshold),
        "tp": true_positive,
        "fp": false_positive,
        "tn": true_negative,
        "fn": false_negative,
        "precision": float(precision),
        "pd": float(probability_detection),
        "recall": float(probability_detection),
        "far": float(false_alarm_rate),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(target_values, score_values)),
        "pr_auc": float(average_precision_score(target_values, score_values)),
        **operating_points,
        "confusion_matrix": [[true_negative, false_positive], [false_negative, true_positive]],
    }


def select_best_threshold(scores: np.ndarray, targets: np.ndarray) -> float:
    """Select the target-probability threshold with maximum validation F1."""
    score_values = np.asarray(scores, dtype=np.float64).ravel()
    target_values = np.asarray(targets, dtype=np.int64).ravel()
    valid = (target_values == 0) | (target_values == 1)
    score_values = score_values[valid]
    target_values = target_values[valid]
    if len(np.unique(target_values)) != 2:
        raise ValueError("threshold selection requires both background and target samples")

    candidates = np.unique(np.concatenate(([0.0], score_values, [1.0])))
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold in candidates:
        result = detection_metrics(score_values, target_values, threshold=float(threshold))
        if result["f1"] > best_f1 + 1e-12:  # type: ignore[operator]
            best_f1 = float(result["f1"])
            best_threshold = float(threshold)
    return best_threshold
