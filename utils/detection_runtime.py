from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch


ModelLike = Callable[[torch.Tensor], torch.Tensor]


def _extract_logits(output: torch.Tensor) -> torch.Tensor:
    if output.ndim != 2 or output.shape[1] != 2:
        raise ValueError(
            f"detector must output logits with shape (B, 2), got {tuple(output.shape)}"
        )
    return output


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch and return mean weighted-cross-entropy loss."""
    model.train()
    losses: list[float] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = _extract_logits(model(images))
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.inference_mode()
def evaluate_detector(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return target probabilities and labels for all loader samples."""
    model.eval()
    scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = _extract_logits(model(images))
        probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        scores.append(probabilities)
        targets.append(labels.numpy())

    if not scores:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.concatenate(scores), np.concatenate(targets).astype(np.int64)


@torch.inference_mode()
def predict_probability_map(
    model: ModelLike,
    image: np.ndarray,
    patch_size: int,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    """Run reflect-padded, patch-wise inference for every image pixel."""
    if image.ndim != 3:
        raise ValueError(f"image must have shape (H, W, C), got {image.shape}")
    if patch_size < 3 or patch_size % 2 == 0:
        raise ValueError(f"patch_size must be an odd integer >= 3, got {patch_size}")
    if patch_size > min(image.shape[0], image.shape[1]):
        raise ValueError("patch_size cannot exceed image height or width")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    if hasattr(model, "eval"):
        model.eval()

    height, width = image.shape[:2]
    pad = patch_size // 2
    padded = np.pad(
        image,
        ((pad, pad), (pad, pad), (0, 0)),
        mode="reflect",
    )
    probabilities = np.zeros((height, width), dtype=np.float32)
    patches: list[np.ndarray] = []
    coordinates: list[tuple[int, int]] = []

    def infer_batch() -> None:
        batch = np.stack(patches, axis=0).transpose(0, 3, 1, 2)[:, np.newaxis]
        inputs = torch.from_numpy(np.ascontiguousarray(batch)).to(device)
        logits = _extract_logits(model(inputs))
        target_probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        for (row, column), probability in zip(coordinates, target_probabilities):
            probabilities[row, column] = probability
        patches.clear()
        coordinates.clear()

    for row in range(height):
        for column in range(width):
            patches.append(padded[row : row + patch_size, column : column + patch_size])
            coordinates.append((row, column))
            if len(patches) == batch_size:
                infer_batch()
    if patches:
        infer_batch()

    return probabilities
