from __future__ import annotations

import numpy as np
import torch


class HSIDataset(torch.utils.data.Dataset):
    """Return reflect-padded spectral patches and their center labels."""

    def __init__(
        self,
        image: np.ndarray,
        ground_truth: np.ndarray,
        patch_size: int,
        data_aug: bool = True,
    ):
        if image.ndim != 3:
            raise ValueError(f"image must have shape (H, W, C), got {image.shape}")
        if ground_truth.shape != image.shape[:2]:
            raise ValueError(
                "ground_truth spatial shape must match image, got "
                f"{ground_truth.shape} and {image.shape[:2]}"
            )
        if patch_size < 3 or patch_size % 2 == 0:
            raise ValueError("patch_size must be an odd integer of at least 3")
        if patch_size > min(image.shape[0], image.shape[1]):
            raise ValueError("patch_size cannot exceed image height or width")

        self.patch_size = patch_size
        self.data_aug = data_aug
        self.pad = patch_size // 2
        self.data = np.pad(
            image,
            ((self.pad, self.pad), (self.pad, self.pad), (0, 0)),
            mode="reflect",
        )

        rows, columns = np.nonzero(ground_truth >= 0)
        self.indices = np.column_stack(
            (rows + self.pad, columns + self.pad)
        ).astype(np.int64)
        self.labels = ground_truth[rows, columns].astype(np.int64)

    def _augment(self, patch: np.ndarray) -> np.ndarray:
        if np.random.random() <= 0.5:
            return patch

        transform = np.random.randint(5)
        if transform == 0:
            return np.fliplr(patch)
        if transform == 1:
            return np.flipud(patch)
        return np.rot90(patch, k=transform - 1)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row, column = self.indices[index]
        top = row - self.pad
        left = column - self.pad
        patch = self.data[
            top : top + self.patch_size,
            left : left + self.patch_size,
        ]

        if self.data_aug:
            patch = self._augment(patch)

        patch = np.ascontiguousarray(patch.transpose(2, 0, 1), dtype=np.float32)
        inputs = torch.from_numpy(patch).unsqueeze(0)
        label = torch.tensor(self.labels[index], dtype=torch.int64)
        return inputs, label
