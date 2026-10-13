"""Monthly cached encoder features, grouped by chip for annual supervision."""

import json
import os
from pathlib import Path
import random

import numpy as np
from skimage import io
from torch.utils.data import Dataset


def load_feature_manifest(directory):
    path = Path(directory) / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "128x128": [32, 128, 128], "64x64": [64, 64, 64],
            "32x32": [96, 32, 32], "16x16": [224, 16, 16], "8x8": [640, 8, 8],
        }
        if (manifest["format_version"] != 1 or manifest["dtype"] != "float32"
                or manifest["features"] != "encoder_before_temporal_attention"
                or manifest["output_shapes"] != expected
                or manifest["preprocessing"]["input_shape"] != [15, 256, 256]):
            raise ValueError("expected the five float32 EfficientNetV2-L feature scales")
        if not manifest.get("checkpoint_sha256"):
            raise ValueError("missing encoder checkpoint SHA-256")
        return manifest
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid feature manifest {path}: {exc}") from exc


def ordered_shapes(manifest):
    return sorted(manifest["output_shapes"].items(), key=lambda item: item[1][1], reverse=True)


def validate_feature_coverage(directory, chip_ids, manifest):
    """Check directory inventories once, rather than stat every required feature."""
    complete = None
    for scale, _ in ordered_shapes(manifest):
        folder = Path(directory) / scale
        if not folder.is_dir():
            raise ValueError(f"Missing feature resolution directory: {folder}")
        with os.scandir(folder) as entries:
            names = {entry.name[:-4] for entry in entries if entry.name.endswith(".npy") and entry.is_file()}
        complete = names if complete is None else complete & names
    for chip_id in chip_ids:
        for month in range(12):
            sample_id = f"{chip_id}_{month:02d}"
            if sample_id not in complete:
                raise ValueError(
                    f"Missing cached features for {sample_id}. All five scales for all 12 "
                    "months are required per chip. Finish extraction or provide a "
                    "--train-df containing only fully extracted chips."
                )


def month_dropout(features, mask, probability=0.3):
    dropped = np.random.rand(len(mask)) < probability
    if dropped.all():
        dropped[np.random.randint(len(mask))] = False
    for scale in features:
        scale[dropped] = 0
    return np.logical_or(mask, dropped)


class FeatureDS(Dataset):
    def __init__(self, df, dir_features, dir_labels, augs=False, manifest=None):
        self.df = df
        self.dir_features = Path(dir_features)
        self.dir_labels = Path(dir_labels)
        self.augs = augs
        self.manifest = manifest or load_feature_manifest(self.dir_features)
        self.scales = ordered_shapes(self.manifest)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        chip_id = str(self.df.iloc[index].chip_id)
        features = []
        for scale, shape in self.scales:
            months = []
            for month in range(12):
                path = self.dir_features / scale / f"{chip_id}_{month:02d}.npy"
                try:
                    array = np.load(path, allow_pickle=False)
                    if (not isinstance(array, np.ndarray) or array.shape != tuple(shape)
                            or array.dtype != np.float32 or not np.isfinite(array).all()):
                        raise ValueError(f"expected finite float32 array of shape {tuple(shape)}")
                    months.append(array)
                except (OSError, ValueError, EOFError) as exc:
                    raise ValueError(f"Invalid cached features {path}: {exc}") from exc
            features.append(np.stack(months))
        mask = np.zeros(12, dtype=bool)
        label_path = self.dir_labels / f"{chip_id}_agbm.tif"
        try:
            target = io.imread(label_path).astype(np.float32)
            if target.shape != (256, 256) or not np.isfinite(target).all():
                raise ValueError("expected finite 256x256 biomass label")
        except (OSError, ValueError) as exc:
            raise ValueError(f"Invalid label {label_path}: {exc}") from exc
        # Cached features do not support the original image geometry augmentations.
        if self.augs and random.random() > 0.5:
            mask = month_dropout(features, mask)
        return features, mask, target
