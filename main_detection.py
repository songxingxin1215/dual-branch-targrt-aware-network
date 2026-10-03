from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from models.hsi_detector import TargetAwareHSIDetector
from utils.dataset import HSIDataset
from utils.detection import (
    class_weights_for_labels,
    detection_metrics,
    load_detection_mat,
    make_balanced_sampler,
    select_best_threshold,
    split_detection_gt,
)
from utils.detection_runtime import evaluate_detector, predict_probability_map, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised HSI target detection on a data/map MAT file")
    parser.add_argument("--dataset_path", type=str, default="dataset.mat")
    parser.add_argument("--output_dir", type=str, default="Results_Detection")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0 or cpu")
    parser.add_argument("--patch_size", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--inference_batch_size", type=int, default=128)
    parser.add_argument("--train_ratio", type=float, default=0.10)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--pca_num", type=int, default=32)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=202201)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_png", action="store_true")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name.lower() == "cpu":
        return torch.device("cpu")
    if device_name.isdigit():
        device_name = f"cuda:{device_name}"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_png_outputs(directory: Path, probability_map: np.ndarray, prediction: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    plt.imsave(directory / "probability_map.png", probability_map, cmap="inferno", vmin=0.0, vmax=1.0)
    plt.imsave(directory / "prediction.png", prediction.astype(np.uint8), cmap="gray", vmin=0, vmax=1)


def summarize_runs(results: list[dict]) -> dict:
    summary = {}
    keys = ["precision", "pd", "far", "f1", "roc_auc", "pr_auc"]
    for key in keys:
        values = np.asarray([result[key] for result in results], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)) if len(values) > 1 else 0.0,
        }
    return summary


def main() -> None:
    opts = parse_args()
    if opts.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if opts.num_runs < 1:
        raise ValueError("num_runs must be at least 1")
    if not 0.0 < opts.train_ratio or not 0.0 < opts.val_ratio or opts.train_ratio + opts.val_ratio >= 1.0:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1")

    device = resolve_device(opts.device)
    output_root = Path(opts.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    original, image, gt, _ = load_detection_mat(opts.dataset_path, pca_num=opts.pca_num)
    class_counts = {int(label): int(np.sum(gt == label)) for label in (0, 1)}
    print(f"dataset: {opts.dataset_path}")
    print(f"image: {image.shape}, target/background ratio: 1:{class_counts[0] / class_counts[1]:.2f}")
    print(f"features: {original.shape[-1]} raw bands + {opts.pca_num} full-image PCA bands")

    all_results = []
    for run in range(opts.num_runs):
        seed = opts.seed + run
        set_seeds(seed)
        train_gt, val_gt, test_gt = split_detection_gt(gt, opts.train_ratio, opts.val_ratio, seed)
        run_dir = output_root / f"run_{run}"
        run_dir.mkdir(parents=True, exist_ok=True)

        train_set = HSIDataset(image, train_gt, patch_size=opts.patch_size, data_aug=True)
        val_set = HSIDataset(image, val_gt, patch_size=opts.patch_size, data_aug=False)
        sampler = make_balanced_sampler(train_set.labels, seed=seed)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=opts.batch_size,
            sampler=sampler,
            num_workers=opts.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=len(train_set) > opts.batch_size,
        )
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=opts.inference_batch_size,
            shuffle=False,
            num_workers=opts.num_workers,
            pin_memory=device.type == "cuda",
        )

        weights = class_weights_for_labels(train_set.labels)
        model = TargetAwareHSIDetector(
            num_bands=original.shape[-1],
            pca_channels=opts.pca_num,
            patch_size=opts.patch_size,
        ).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=opts.learning_rate, weight_decay=opts.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opts.epochs)

        best_score = -math.inf
        checkpoint_path = run_dir / "model_best.pth"
        for epoch in range(1, opts.epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            scores, targets = evaluate_detector(model, val_loader, device)
            validation = detection_metrics(scores, targets, threshold=0.5)
            score = validation["pr_auc"] + validation["f1"] * 1e-6
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), checkpoint_path)
            scheduler.step()
            if epoch == 1 or epoch % 10 == 0 or epoch == opts.epochs:
                print(
                    f"run {run + 1}/{opts.num_runs} epoch {epoch:03d}/{opts.epochs} "
                    f"loss={loss:.6f} val_pr_auc={validation['pr_auc']:.4f} "
                    f"val_f1={validation['f1']:.4f} val_far={validation['far']:.4f}"
                )

        if not checkpoint_path.exists():
            torch.save(model.state_dict(), checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        val_scores, val_targets = evaluate_detector(model, val_loader, device)
        threshold = select_best_threshold(val_scores, val_targets)

        probability_map = predict_probability_map(
            model,
            image,
            patch_size=opts.patch_size,
            device=device,
            batch_size=opts.inference_batch_size,
        )
        test_mask = test_gt >= 0
        result = detection_metrics(probability_map[test_mask], test_gt[test_mask], threshold=threshold)
        prediction = (probability_map >= threshold).astype(np.uint8)

        np.save(run_dir / "probability_map.npy", probability_map)
        np.save(run_dir / "prediction.npy", prediction)
        np.save(run_dir / "test_mask.npy", test_mask)
        torch.save(
            {
                "model_state": model.state_dict(),
                "threshold": threshold,
                "patch_size": opts.patch_size,
                "pca_num": opts.pca_num,
                "num_bands": original.shape[-1],
            },
            run_dir / "detector_bundle.pth",
        )
        if opts.save_png:
            save_png_outputs(run_dir, probability_map, prediction)

        result["run"] = run
        result["seed"] = seed
        result["train_background"] = int(np.sum(train_gt == 0))
        result["train_target"] = int(np.sum(train_gt == 1))
        result["val_background"] = int(np.sum(val_gt == 0))
        result["val_target"] = int(np.sum(val_gt == 1))
        result["test_background"] = int(np.sum(test_gt == 0))
        result["test_target"] = int(np.sum(test_gt == 1))
        with open(run_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        all_results.append(result)
        print(
            f"run {run + 1} test: ROC-AUC={result['roc_auc']:.4f} PR-AUC={result['pr_auc']:.4f} "
            f"PD={result['pd']:.4f} FAR={result['far']:.4f} F1={result['f1']:.4f} threshold={threshold:.4f}"
        )

        del model, criterion, optimizer, scheduler, train_set, val_set, train_loader, val_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = summarize_runs(all_results)
    with open(output_root / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("aggregated results:")
    for key, values in summary.items():
        print(f"  {key}: {values['mean']:.4f} +/- {values['std']:.4f}")


if __name__ == "__main__":
    main()
