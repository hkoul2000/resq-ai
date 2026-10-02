"""Synthetic dataset for smoke testing when real data is unavailable.

Generates realistic synthetic multi-modal flood data matching the
Sen1Floods11Dataset interface, allowing the full pipeline to run
without downloading any real data.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SyntheticFloodDataset(Dataset):
    """Synthetic dataset simulating multi-modal flood data.

    Generates deterministic synthetic data for smoke testing, matching
    the output format of Sen1Floods11Dataset and MultiModalFloodDataset.

    Returns dict with keys:
        - sar: (2, H, W) SAR backscatter
        - optical: (13, H, W) optical bands
        - geo: (C_geo, H, W) terrain features
        - rainfall: (seq_len, 1) rainfall time series
        - label: (H, W) binary flood mask
        - valid_mask: (H, W) valid pixel mask
        - modality_mask: (5,) modality availability [sar, optical, dem, lc, rain]
        - region: str event region name
        - chip_id: str chip identifier
    """

    # Synthetic event names for realistic testing
    REGIONS = [
        "India-Kerala", "India-Bihar", "Bangladesh-Dhaka",
        "Bolivia-Beni", "USA-Houston", "Spain-Valencia",
        "Nigeria-Lagos", "Pakistan-Sindh",
    ]

    def __init__(
        self,
        split: str = "train",
        num_samples: int = 32,
        crop_size: int = 64,
        sar_channels: int = 2,
        optical_channels: int = 13,
        geo_channels: int = 6,
        rainfall_seq_len: int = 30,
        modality_dropout: float = 0.0,
        seed: int = 42,
        config: dict | None = None,
    ) -> None:
        super().__init__()
        self.split = split
        self.num_samples = num_samples
        self.crop_size = crop_size
        self.sar_channels = sar_channels
        self.optical_channels = optical_channels
        self.geo_channels = geo_channels
        self.rainfall_seq_len = rainfall_seq_len
        self.modality_dropout = modality_dropout if split == "train" else 0.0

        # Override from config if provided
        if config is not None:
            data_cfg = config.get("data", {})
            model_cfg = config.get("model", {})
            self.crop_size = data_cfg.get("crop_size", crop_size)
            self.sar_channels = model_cfg.get("sar_channels", sar_channels)
            self.optical_channels = model_cfg.get("optical_channels", optical_channels)
            self.geo_channels = model_cfg.get("geo_channels", geo_channels)
            self.rainfall_seq_len = model_cfg.get("rainfall_seq_len", rainfall_seq_len)
            self.modality_dropout = data_cfg.get("modality_dropout", modality_dropout)
            seed = config.get("project", {}).get("seed", seed)

            # Sample counts from config
            num_key = {"train": "num_train", "valid": "num_val", "val": "num_val", "test": "num_test"}
            self.num_samples = data_cfg.get(num_key.get(split, "num_train"), num_samples)

        # Deterministic seeding per split
        split_offset = {"train": 0, "valid": 100, "val": 100, "test": 200, "calibration": 300}
        self.seed = seed + split_offset.get(split, 0)
        self.rng = np.random.default_rng(self.seed)

        # Pre-generate region assignments
        self._region_assignments = [
            self.REGIONS[i % len(self.REGIONS)] for i in range(self.num_samples)
        ]

        logger.info(
            f"SyntheticFloodDataset [{split}]: {self.num_samples} samples, "
            f"size={self.crop_size}, seed={self.seed}"
        )

    def __len__(self) -> int:
        return self.num_samples

    def _generate_flood_label(self, h: int, w: int) -> np.ndarray:
        """Generate synthetic flood label with circular/irregular regions."""
        label = np.zeros((h, w), dtype=np.float32)
        num_blobs = self.rng.integers(1, 5)

        for _ in range(num_blobs):
            # Random elliptical flood region
            max_r = min(h, w) // 4
            min_r = max(3, max_r // 4)
            rx = self.rng.integers(min_r, max(min_r + 1, max_r))
            ry = self.rng.integers(min_r, max(min_r + 1, max_r))
            cx = self.rng.integers(rx, max(rx + 1, w - rx))
            cy = self.rng.integers(ry, max(ry + 1, h - ry))

            y, x = np.ogrid[:h, :w]
            mask = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0
            label[mask] = 1.0

        return label

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Use index-specific seed for reproducibility
        rng = np.random.default_rng(self.seed + idx)
        h, w = self.crop_size, self.crop_size

        # Generate SAR-like data (log-scale backscatter)
        sar = rng.normal(-15.0, 5.0, (self.sar_channels, h, w)).astype(np.float32)

        # Generate optical-like data (reflectance values)
        optical = np.clip(
            rng.normal(1500.0, 500.0, (self.optical_channels, h, w)),
            0, 10000
        ).astype(np.float32)

        # Generate geo features (elevation, slope, HAND, TWI, dist_drainage, landcover)
        geo = np.zeros((self.geo_channels, h, w), dtype=np.float32)
        geo[0] = rng.normal(100.0, 50.0, (h, w))   # elevation
        geo[1] = np.abs(rng.normal(5.0, 3.0, (h, w)))  # slope
        geo[2] = np.abs(rng.normal(10.0, 8.0, (h, w)))  # HAND
        geo[3] = rng.normal(8.0, 3.0, (h, w))  # TWI
        geo[4] = np.abs(rng.normal(500.0, 300.0, (h, w)))  # dist to drainage
        if self.geo_channels > 5:
            geo[5] = rng.integers(0, 11, (h, w)).astype(np.float32)  # landcover class

        # Generate rainfall sequence
        rainfall = np.abs(
            rng.normal(5.0, 10.0, (self.rainfall_seq_len, 1))
        ).astype(np.float32)

        # Generate flood label
        label = self._generate_flood_label(h, w)
        valid_mask = np.ones((h, w), dtype=np.float32)

        # Modality availability mask
        modality_mask = np.ones(5, dtype=np.float32)

        # Apply modality dropout during training
        if self.modality_dropout > 0:
            for i in range(5):
                if rng.random() < self.modality_dropout:
                    modality_mask[i] = 0.0
                    if i == 0:
                        sar = np.zeros_like(sar)
                    elif i == 1:
                        optical = np.zeros_like(optical)
                    elif i in (2, 3):
                        geo = np.zeros_like(geo)
                    elif i == 4:
                        rainfall = np.zeros_like(rainfall)

        region = self._region_assignments[idx]
        chip_id = f"{idx:04d}"

        return {
            "sar": torch.from_numpy(sar),
            "optical": torch.from_numpy(optical),
            "geo": torch.from_numpy(geo),
            "rainfall": torch.from_numpy(rainfall),
            "label": torch.from_numpy(label),
            "valid_mask": torch.from_numpy(valid_mask),
            "modality_mask": torch.from_numpy(modality_mask),
            "region": region,
            "chip_id": chip_id,
        }


def create_synthetic_dataloaders(
    config: dict,
    batch_size: int = 4,
    num_workers: int = 0,
) -> dict[str, torch.utils.data.DataLoader]:
    """Create train/val/test DataLoaders with synthetic data.

    Args:
        config: Configuration dictionary
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        Dictionary with 'train', 'valid', 'test' DataLoaders
    """
    data_cfg = config.get("data", {})
    batch_size = data_cfg.get("batch_size", batch_size)

    def collate_fn(batch: list[dict]) -> dict[str, Any]:
        """Custom collate that handles string fields."""
        result = {}
        for key in batch[0]:
            if isinstance(batch[0][key], torch.Tensor):
                result[key] = torch.stack([b[key] for b in batch])
            elif isinstance(batch[0][key], str):
                result[key] = [b[key] for b in batch]
            else:
                result[key] = [b[key] for b in batch]
        return result

    dataloaders = {}
    for split, num_key in [("train", "num_train"), ("valid", "num_val"), ("test", "num_test")]:
        num_samples = data_cfg.get(num_key, 32 if split == "train" else 8)
        ds = SyntheticFloodDataset(
            split=split,
            num_samples=num_samples,
            config=config,
        )
        dataloaders[split] = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=(split == "train"),
        )

    return dataloaders
