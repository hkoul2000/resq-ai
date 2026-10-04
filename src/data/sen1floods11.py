"""Sen1Floods11 dataset module.

Downloads and processes the Sen1Floods11 dataset from the public Google Cloud bucket.
Provides PyTorch datasets with multi-modal inputs and modality availability masks.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Sen1Floods11 event metadata: region -> country mapping
EVENT_METADATA: dict[str, dict[str, Any]] = {
    "Bangladesh": {"country": "Bangladesh", "continent": "Asia", "india_subcontinent": True},
    "Bolivia": {"country": "Bolivia", "continent": "South America", "india_subcontinent": False},
    "Cambodia": {"country": "Cambodia", "continent": "Asia", "india_subcontinent": False},
    "Ghana": {"country": "Ghana", "continent": "Africa", "india_subcontinent": False},
    "India-ثorth": {"country": "India", "continent": "Asia", "india_subcontinent": True},
    "India-South": {"country": "India", "continent": "Asia", "india_subcontinent": True},
    "Kerala": {"country": "India", "continent": "Asia", "india_subcontinent": True},
    "Mekong": {"country": "Vietnam", "continent": "Asia", "india_subcontinent": False},
    "Myanmar": {"country": "Myanmar", "continent": "Asia", "india_subcontinent": True},
    "Nigeria": {"country": "Nigeria", "continent": "Africa", "india_subcontinent": False},
    "Pakistan": {"country": "Pakistan", "continent": "Asia", "india_subcontinent": True},
    "Paraguay": {"country": "Paraguay", "continent": "South America", "india_subcontinent": False},
    "Somalia": {"country": "Somalia", "continent": "Africa", "india_subcontinent": False},
    "Spain": {"country": "Spain", "continent": "Europe", "india_subcontinent": False},
    "Sri-Lanka": {"country": "Sri Lanka", "continent": "Asia", "india_subcontinent": True},
    "USA": {"country": "USA", "continent": "North America", "india_subcontinent": False},
}

# GCS bucket paths
GCS_BUCKET = "gs://sen1floods11"
GCS_HTTP_BASE = "https://storage.googleapis.com/sen1floods11"

# Split files in the dataset
SPLIT_FILES = {
    "train": "flood_train_data.csv",
    "valid": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
    "bolivia": "flood_bolivia_data.csv",
}


class Sen1Floods11Catalog:
    """Manages the Sen1Floods11 dataset catalog and file paths."""

    def __init__(self, root: str | Path, download: bool = True) -> None:
        self.root = Path(root) / "sen1floods11"
        self.root.mkdir(parents=True, exist_ok=True)

        self.s1_dir = self.root / "v1.1" / "data" / "flood_events" / "HandLabeled" / "S1Hand"
        self.s2_dir = self.root / "v1.1" / "data" / "flood_events" / "HandLabeled" / "S2Hand"
        self.label_dir = self.root / "v1.1" / "data" / "flood_events" / "HandLabeled" / "LabelHand"
        self.split_dir = self.root / "v1.1" / "splits" / "flood_handlabeled"

        if download:
            self._download_split_files()

    def _download_split_files(self) -> None:
        """Download split CSV files if not present."""
        import requests

        self.split_dir.mkdir(parents=True, exist_ok=True)
        for split_name, filename in SPLIT_FILES.items():
            local_path = self.split_dir / filename
            if local_path.exists():
                continue
            url = f"{GCS_HTTP_BASE}/v1.1/splits/flood_handlabeled/{filename}"
            logger.info(f"Downloading split file: {filename}")
            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                local_path.write_text(resp.text)
                logger.info(f"  Saved to {local_path}")
            except Exception as e:
                logger.warning(f"  Failed to download {filename}: {e}")

    def get_split_chips(self, split: str) -> list[dict[str, Path]]:
        """Get list of chip paths for a given split.

        Returns list of dicts with keys: s1, s2, label, region, chip_id
        """
        split_file = self.split_dir / SPLIT_FILES.get(split, f"flood_{split}_data.csv")
        if not split_file.exists():
            logger.warning(f"Split file not found: {split_file}")
            return []

        chips = []
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Format: S1Hand/region_chipid.tif or similar
                parts = line.split(",")
                s1_path_str = parts[0].strip() if len(parts) > 0 else ""
                label_path_str = parts[1].strip() if len(parts) > 1 else ""

                if not s1_path_str:
                    continue

                # Extract region and chip_id from filename
                basename = Path(s1_path_str).stem
                region = "_".join(basename.split("_")[:-1])
                chip_id = basename.split("_")[-1]

                chip = {
                    "s1": self.s1_dir / Path(s1_path_str).name,
                    "s2": self.s2_dir / Path(s1_path_str).name.replace("S1Hand", "S2Hand"),
                    "label": self.label_dir / Path(label_path_str).name if label_path_str else None,
                    "region": region,
                    "chip_id": chip_id,
                }
                chips.append(chip)

        return chips

    def get_leave_event_out_splits(
        self, held_out_event: str
    ) -> tuple[list[dict], list[dict]]:
        """Create leave-one-event-out split."""
        all_chips = []
        for split in ["train", "valid", "test"]:
            all_chips.extend(self.get_split_chips(split))

        train_chips = [c for c in all_chips if c["region"] != held_out_event]
        test_chips = [c for c in all_chips if c["region"] == held_out_event]
        return train_chips, test_chips

    def download_chip(self, chip: dict[str, Any]) -> bool:
        """Download a single chip's S1, S2, and label files."""
        import requests

        success = True
        for key in ["s1", "s2", "label"]:
            path = chip.get(key)
            if path is None or path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)

            # Construct GCS URL
            rel_path = path.relative_to(self.root)
            url = f"{GCS_HTTP_BASE}/{rel_path.as_posix()}"

            try:
                resp = requests.get(url, timeout=120, stream=True)
                resp.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
            except Exception as e:
                logger.warning(f"Failed to download {url}: {e}")
                success = False

        return success


class Sen1Floods11Dataset(Dataset):
    """PyTorch Dataset for Sen1Floods11 with multi-modal support.

    Supports SAR (S1), optical (S2), and label data.
    Returns modality availability masks for handling missing data.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        modalities: list[str] | None = None,
        crop_size: int = 256,
        augment: bool = False,
        modality_dropout: float = 0.0,
        normalize: bool = True,
        download: bool = True,
        smoke: bool = False,
        max_chips: int | None = None,
        transform: Any = None,
        chips: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.modalities = modalities or ["sar", "optical"]
        self.crop_size = crop_size
        self.augment = augment and split == "train"
        self.modality_dropout = modality_dropout if split == "train" else 0.0
        self.normalize = normalize
        self.transform = transform
        self.smoke = smoke

        # Initialize catalog
        self.catalog = Sen1Floods11Catalog(root, download=download)

        # Get chip list
        if chips is not None:
            self.chips = list(chips)
        elif split == "bolivia":
            self.chips = self.catalog.get_split_chips("bolivia")
        else:
            self.chips = self.catalog.get_split_chips(split)

        # Limit for smoke testing
        if smoke and max_chips is None:
            max_chips = 20
        if max_chips is not None:
            self.chips = self.chips[:max_chips]

        # In non-smoke mode, enforce real data presence (NO silent fallbacks)
        if not self.smoke:
            if len(self.chips) == 0:
                raise FileNotFoundError(
                    f"No Sen1Floods11 chips found for split '{split}' in {self.root}. "
                    "Real dataset files must be downloaded before running full experiments. "
                    "Download the dataset using Sen1Floods11Catalog or run with --smoke for CPU testing."
                )
            sample_chip = self.chips[0]
            if sample_chip.get("s1") and not sample_chip["s1"].exists():
                raise FileNotFoundError(
                    f"Sen1Floods11 imagery file does not exist on disk: {sample_chip['s1']}. "
                    "Please download the official GeoTIFFs or run with --smoke for CPU testing."
                )

        # Normalization statistics (from Sen1Floods11 paper)
        self.s1_mean = np.array([-12.54, -20.19], dtype=np.float32)
        self.s1_std = np.array([5.25, 5.73], dtype=np.float32)
        self.s2_mean = np.array(
            [1226.0, 1137.0, 1139.0, 1350.0, 1932.0, 2520.0,
             2834.0, 2971.0, 3054.0, 2160.0, 1854.0, 1542.0, 1116.0],
            dtype=np.float32,
        )
        self.s2_std = np.array(
            [741.0, 740.0, 870.0, 849.0, 863.0, 1124.0,
             1275.0, 1345.0, 1408.0, 960.0, 910.0, 850.0, 757.0],
            dtype=np.float32,
        )

        logger.info(
            f"Sen1Floods11 {split}: {len(self.chips)} chips, "
            f"modalities={self.modalities}, crop={crop_size}, augment={self.augment}"
        )

    def __len__(self) -> int:
        return len(self.chips)

    def _load_tif(self, path: Path, is_sar: bool = False) -> np.ndarray | None:
        """Load a GeoTIFF file, masking nodata and unphysical values."""
        if path is None or not path.exists():
            return None
        try:
            import rasterio

            with rasterio.open(path) as src:
                data = src.read().astype(np.float32)
                nodata_val = src.nodata

            # Replace tagged nodata sentinel with NaN
            if nodata_val is not None:
                data[data == nodata_val] = np.nan

            # For SAR: mask unphysical values and untagged -9999 nodata sentinels.
            # Sentinel-1 GRD in dB typically spans [-35, +5] dB; clamp to [-50, 25] dB.
            if is_sar:
                data[data <= -9000.0] = np.nan
                data[(data < -50.0) | (data > 25.0)] = np.nan

            return data
        except Exception as e:
            logger.debug(f"Failed to load {path}: {e}")
            return None

    def _impute_sar(self, sar_data: np.ndarray) -> np.ndarray:
        """Impute NaN/Inf pixels in SAR with the channel's valid mean."""
        sar_clean = sar_data.copy()
        for c in range(sar_clean.shape[0]):
            band = sar_clean[c]
            valid_mask = np.isfinite(band)
            if np.any(valid_mask):
                fill_val = float(np.mean(band[valid_mask]))
            else:
                fill_val = float(self.s1_mean[c])
            sar_clean[c] = np.where(valid_mask, band, fill_val)
        return sar_clean

    def _random_crop(self, *arrays: np.ndarray | None) -> list[np.ndarray | None]:
        """Apply consistent random crop to all arrays during training only."""
        # Find first non-None array to get dimensions
        ref = None
        for arr in arrays:
            if arr is not None:
                ref = arr
                break
        if ref is None:
            return list(arrays)

        h, w = ref.shape[-2:]
        if h <= self.crop_size and w <= self.crop_size:
            return list(arrays)

        y = np.random.randint(0, max(1, h - self.crop_size))
        x = np.random.randint(0, max(1, w - self.crop_size))

        result = []
        for arr in arrays:
            if arr is None:
                result.append(None)
            else:
                result.append(arr[..., y : y + self.crop_size, x : x + self.crop_size])
        return result

    def _augment(self, *arrays: np.ndarray | None) -> list[np.ndarray | None]:
        """Apply random augmentations (flips, rotations)."""
        if not self.augment:
            return list(arrays)

        # Random horizontal flip
        if np.random.random() > 0.5:
            arrays = tuple(
                np.flip(arr, axis=-1).copy() if arr is not None else None for arr in arrays
            )

        # Random vertical flip
        if np.random.random() > 0.5:
            arrays = tuple(
                np.flip(arr, axis=-2).copy() if arr is not None else None for arr in arrays
            )

        # Random 90-degree rotation
        k = np.random.randint(0, 4)
        if k > 0:
            arrays = tuple(
                np.rot90(arr, k=k, axes=(-2, -1)).copy() if arr is not None else None
                for arr in arrays
            )

        return list(arrays)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chip = self.chips[idx]

        # Load data with SAR-specific nodata masking
        s1_data = self._load_tif(chip["s1"], is_sar=True)   # (2, H, W) - VV, VH
        s2_data = self._load_tif(chip["s2"], is_sar=False)  # (13, H, W) - 13 bands
        label_data = self._load_tif(chip.get("label"), is_sar=False)  # (1, H, W)

        # Apply random crop ONLY during training; evaluation uses full chips deterministically
        if self.split == "train":
            s1_data, s2_data, label_data = self._random_crop(s1_data, s2_data, label_data)
            # Augmentation (train only)
            s1_data, s2_data, label_data = self._augment(s1_data, s2_data, label_data)

        # Determine spatial dimensions
        ref_arr = s1_data if s1_data is not None else (s2_data if s2_data is not None else label_data)
        h, w = ref_arr.shape[-2:] if ref_arr is not None else (self.crop_size, self.crop_size)

        # Build output dict
        sample: dict[str, Any] = {
            "region": chip["region"],
            "chip_id": chip["chip_id"],
        }

        # Modality availability mask
        modality_mask = {}

        # SAR (Sentinel-1)
        if "sar" in self.modalities:
            if s1_data is not None:
                # 1. Impute nodata/NaN with channel valid mean prior to normalization
                s1_data = self._impute_sar(s1_data)
                # 2. Per-channel normalization
                if self.normalize:
                    s1_data = (s1_data - self.s1_mean[:, None, None]) / (
                        self.s1_std[:, None, None] + 1e-8
                    )
                # 3. Post-normalization nan_to_num safety guard
                s1_data = np.nan_to_num(s1_data, nan=0.0, posinf=5.0, neginf=-5.0)
                # Modality dropout
                if self.modality_dropout > 0 and np.random.random() < self.modality_dropout:
                    s1_data = np.zeros_like(s1_data)
                    modality_mask["sar"] = False
                else:
                    modality_mask["sar"] = True
                sample["sar"] = torch.from_numpy(s1_data.copy())
            else:
                sample["sar"] = torch.zeros(2, h, w)
                modality_mask["sar"] = False

        # Optical (Sentinel-2)
        if "optical" in self.modalities:
            if s2_data is not None:
                # Impute any optical NaN with median / zero
                s2_data = np.nan_to_num(s2_data, nan=0.0)
                if self.normalize:
                    s2_data = (s2_data - self.s2_mean[:, None, None]) / (
                        self.s2_std[:, None, None] + 1e-8
                    )
                s2_data = np.nan_to_num(s2_data, nan=0.0, posinf=10.0, neginf=-10.0)
                if self.modality_dropout > 0 and np.random.random() < self.modality_dropout:
                    s2_data = np.zeros_like(s2_data)
                    modality_mask["optical"] = False
                else:
                    modality_mask["optical"] = True
                sample["optical"] = torch.from_numpy(s2_data.copy())
            else:
                sample["optical"] = torch.zeros(13, h, w)
                modality_mask["optical"] = False

        # Label: Sen1Floods11 uses 1=flood, 0=dry, -1=nodata
        if label_data is not None:
            valid_mask = ((label_data >= 0) & np.isfinite(label_data)).astype(np.float32)
            label = (label_data > 0).astype(np.float32) * valid_mask
            sample["label"] = torch.from_numpy(label.copy()).squeeze(0)
            sample["valid_mask"] = torch.from_numpy(valid_mask.copy()).squeeze(0)
        else:
            sample["label"] = torch.zeros(h, w)
            sample["valid_mask"] = torch.zeros(h, w)

        # Modality availability tensor
        mod_names = ["sar", "optical", "dem", "landcover", "rainfall"]
        mask_vec = torch.tensor(
            [modality_mask.get(m, False) for m in mod_names], dtype=torch.float32
        )
        sample["modality_mask"] = mask_vec

        return sample


class MultiModalFloodDataset(Dataset):
    """Extended dataset that integrates DEM, land cover, and rainfall with Sen1Floods11.

    This wraps Sen1Floods11Dataset and adds terrain, land cover, and rainfall features.
    """

    def __init__(
        self,
        sen1floods_dataset: Sen1Floods11Dataset,
        dem_dir: str | Path | None = None,
        landcover_dir: str | Path | None = None,
        rainfall_dir: str | Path | None = None,
        geo_channels: int = 6,
        rainfall_seq_len: int = 30,
    ) -> None:
        self.base = sen1floods_dataset
        self.dem_dir = Path(dem_dir) if dem_dir else None
        self.landcover_dir = Path(landcover_dir) if landcover_dir else None
        self.rainfall_dir = Path(rainfall_dir) if rainfall_dir else None
        self.geo_channels = geo_channels
        self.rainfall_seq_len = rainfall_seq_len

    def __len__(self) -> int:
        return len(self.base)

    def _load_terrain(self, region: str, chip_id: str) -> torch.Tensor | None:
        """Load pre-computed terrain features (elevation, slope, HAND, TWI, dist_drainage)."""
        if self.dem_dir is None:
            return None
        path = self.dem_dir / f"{region}_{chip_id}_terrain.npy"
        if not path.exists():
            return None
        data = np.load(path).astype(np.float32)
        return torch.from_numpy(data)

    def _load_landcover(self, region: str, chip_id: str) -> torch.Tensor | None:
        """Load ESA WorldCover land cover classification."""
        if self.landcover_dir is None:
            return None
        path = self.landcover_dir / f"{region}_{chip_id}_lc.npy"
        if not path.exists():
            return None
        data = np.load(path).astype(np.float32)
        return torch.from_numpy(data)

    def _load_rainfall(self, region: str, chip_id: str) -> torch.Tensor | None:
        """Load CHIRPS rainfall time series (30-day antecedent)."""
        if self.rainfall_dir is None:
            return None
        path = self.rainfall_dir / f"{region}_{chip_id}_rain.npy"
        if not path.exists():
            return None
        data = np.load(path).astype(np.float32)
        # Shape: (seq_len,) or (seq_len, features)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        return torch.from_numpy(data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.base[idx]
        region = sample["region"]
        chip_id = sample["chip_id"]
        # Dynamic spatial dimensions (matches 256 on train crop, 512 on val/test)
        ref_t = sample.get("sar") if "sar" in sample else sample.get("optical")
        h, w = ref_t.shape[-2:] if ref_t is not None else (self.base.crop_size, self.base.crop_size)

        # Add terrain features
        terrain = self._load_terrain(region, chip_id)
        if terrain is not None:
            sample["geo"] = terrain
            sample["modality_mask"][2] = 1.0  # dem available
        else:
            sample["geo"] = torch.zeros(self.geo_channels, h, w)

        # Add land cover
        lc = self._load_landcover(region, chip_id)
        if lc is not None:
            # Combine with geo features or keep separate
            sample["modality_mask"][3] = 1.0  # landcover available
            # Append to geo tensor
            if terrain is not None:
                sample["geo"] = torch.cat([sample["geo"], lc], dim=0)
            else:
                sample["geo"] = lc

        # Add rainfall
        rainfall = self._load_rainfall(region, chip_id)
        if rainfall is not None:
            sample["rainfall"] = rainfall
            sample["modality_mask"][4] = 1.0  # rainfall available
        else:
            sample["rainfall"] = torch.zeros(self.rainfall_seq_len, 1)

        return sample


def create_dataloaders(
    config: dict,
    split_type: str = "official",
    held_out_event: Optional[str] = None,
    dem_dir: Optional[str | Path] = None,
    landcover_dir: Optional[str | Path] = None,
    rainfall_dir: Optional[str | Path] = None,
    smoke: bool = False,
) -> dict[str, torch.utils.data.DataLoader]:
    """Create train/val/test dataloaders from config using real Sen1Floods11 data.

    Args:
        config: Configuration dictionary
        split_type: 'official' or 'leave_event_out'
        held_out_event: Event name for leave-event-out validation (e.g. 'Bolivia', 'Ghana')
        dem_dir: Directory containing preprocessed DEM/terrain npy files
        landcover_dir: Directory containing WorldCover npy files
        rainfall_dir: Directory containing CHIRPS rainfall npy files
        smoke: If True, allow small subset for quick smoke checks

    Returns:
        Dictionary of DataLoaders with keys 'train', 'valid', 'test'
    """
    data_cfg = config.get("data", {})
    root = data_cfg.get("root", "data/")
    crop_size = data_cfg.get("crop_size", 256)
    batch_size = data_cfg.get("batch_size", 16)
    num_workers = data_cfg.get("num_workers", 4)
    modalities = data_cfg.get("modalities", ["sar", "optical"])
    modality_dropout = data_cfg.get("modality_dropout", 0.2)

    if dem_dir is None:
        dem_dir = data_cfg.get("dem_dir")
    if landcover_dir is None:
        landcover_dir = data_cfg.get("landcover_dir")
    if rainfall_dir is None:
        rainfall_dir = data_cfg.get("rainfall_dir")

    if smoke:
        crop_size = config.get("smoke", {}).get("crop_size", 64)
        batch_size = config.get("smoke", {}).get("batch_size", 4)
        num_workers = config.get("smoke", {}).get("num_workers", 0)

    split_chips_map: dict[str, list[dict[str, Any]] | None] = {
        "train": None,
        "valid": None,
        "test": None,
    }

    if split_type == "leave_event_out":
        if not held_out_event:
            held_out_event = "Bolivia"
        catalog = Sen1Floods11Catalog(root, download=True)
        train_pool, test_chips = catalog.get_leave_event_out_splits(held_out_event)
        n_val = max(1, int(len(train_pool) * 0.15))
        split_chips_map["train"] = train_pool[:-n_val]
        split_chips_map["valid"] = train_pool[-n_val:]
        split_chips_map["test"] = test_chips
        logger.info(
            f"Leave-event-out split for '{held_out_event}': "
            f"train={len(split_chips_map['train'])}, "
            f"val={len(split_chips_map['valid'])}, "
            f"test={len(split_chips_map['test'])}"
        )

    dataloaders = {}

    for split in ["train", "valid", "test"]:
        base_ds = Sen1Floods11Dataset(
            root=root,
            split=split,
            modalities=modalities,
            crop_size=crop_size,
            augment=(split == "train"),
            modality_dropout=modality_dropout if split == "train" else 0.0,
            normalize=True,
            smoke=smoke,
            chips=split_chips_map[split],
        )

        # Wrap with multi-modal features (DEM, Landcover, Rainfall)
        multi_ds = MultiModalFloodDataset(
            sen1floods_dataset=base_ds,
            dem_dir=dem_dir,
            landcover_dir=landcover_dir,
            rainfall_dir=rainfall_dir,
            geo_channels=config.get("model", {}).get("geo_channels", 6),
            rainfall_seq_len=config.get("model", {}).get("rainfall_seq_len", 30),
        )

        dataloaders[split] = torch.utils.data.DataLoader(
            multi_ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train" and len(multi_ds) > batch_size),
        )

    return dataloaders
