"""Export the five frozen encoder feature maps for each S1/S2 chip-month pair."""

import argparse
import hashlib
import importlib
import io as bytes_io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager

import numpy as np
from skimage import io
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import IMG_SIZE, s1_min, s1_mm, s2_max


SCALES = {
    "128x128": (32, 128, 128),
    "64x64": (64, 64, 64),
    "32x32": (96, 32, 32),
    "16x16": (224, 16, 16),
    "8x8": (640, 8, 8),
}
SAMPLE_BYTES = sum(int(np.prod(shape)) * 4 for shape in SCALES.values())
BACKBONE = "tf_efficientnetv2_l.in21k"


def discover_samples(directory, limit=100):
    """Select chip/month pairs before considering existing outputs."""
    samples = []
    pattern = re.compile(r"^(.+)_S1_(\d{2})\.tif$")
    with os.scandir(directory) as entries:
        for entry in entries:
            match = pattern.fullmatch(entry.name)
            if match and entry.is_file():
                if not 0 <= int(match[2]) < 12:
                    raise ValueError(f"Invalid month in {entry.path}")
                samples.append((f"{match[1]}_{match[2]}", entry.name))
    samples.sort(key=lambda item: item[1])
    if not samples:
        raise ValueError(f"No chip_S1_MM.tif files found in {directory}")
    return samples[:limit] if limit else samples


def read_image(path, channels):
    try:
        image = io.imread(path)
        expected = IMG_SIZE + (channels,)
        if image.shape != expected:
            raise ValueError(f"expected shape {expected}, got {image.shape}")
        if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
            raise ValueError("image must contain finite numeric values")
        return image
    except Exception as exc:
        raise ValueError(f"Cannot read input {path}: {exc}") from exc


def read_pair(directory, s1_filename):
    s1_path = directory / s1_filename
    image_s1 = read_image(s1_path, 4)
    missing = image_s1 == -9999
    image_s1 = (image_s1.astype(np.float32) - s1_min) / s1_mm
    image_s1 = np.where(missing, 0, image_s1)
    chip, month = s1_filename.rsplit("_S1_", 1)
    s2_path = directory / f"{chip}_S2_{month}"
    if s2_path.exists():
        image_s2 = read_image(s2_path, 11).astype(np.float32) / s2_max
    else:
        image_s2 = np.zeros(IMG_SIZE + (11,), dtype=np.float32)
    image = np.concatenate((image_s1, image_s2), axis=2)
    return np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)


class ChipMonthDataset(Dataset):
    def __init__(self, directory, samples):
        self.directory = directory
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_id, filename = self.samples[index]
        return sample_id, torch.from_numpy(read_pair(self.directory, filename))


@contextmanager
def legacy_timm_aliases():
    """Unpickle original classes without permanently changing module aliases."""
    aliases = {
        "timm.models.efficientnet_blocks": "timm.models._efficientnet_blocks",
        "timm.models.layers.conv2d_same": "timm.layers.conv2d_same",
        "timm.models.layers.norm_act": "timm.layers.norm_act",
    }
    previous = {name: sys.modules.get(name) for name in aliases}
    try:
        for old, new in aliases.items():
            sys.modules[old] = importlib.import_module(new)
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_encoder(checkpoint, device):
    import timm
    import models  # Registers the classes serialized in the trusted checkpoint.

    with legacy_timm_aliases():
        model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(model, models.UnetVFLOW):
        raise ValueError("Expected a complete UnetVFLOW checkpoint (modelo_best.pth)")
    encoder = timm.create_model(
        BACKBONE, in_chans=15, pretrained=False, num_classes=0,
        features_only=True, out_indices=(0, 1, 2, 3, 4),
    )
    encoder.load_state_dict(model.encoder.model.state_dict(), strict=True)
    del model
    return encoder.float().eval().requires_grad_(False).to(
        device=device, memory_format=torch.channels_last,
    )


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_manifest(checkpoint, directory):
    return {
        "format_version": 1,
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "input_directory": str(directory),
        "backbone": BACKBONE,
        "features": "encoder_before_temporal_attention",
        "dtype": "float32",
        "compute": {"autocast": False, "tf32": False, "eval": True},
        "output_shapes": {name: list(shape) for name, shape in SCALES.items()},
        "preprocessing": {
            "input_shape": [15, *IMG_SIZE],
            "s1_min": s1_min.tolist(), "s1_range": s1_mm.tolist(),
            "s1_nodata": -9999, "s1_nodata_fill": 0,
            "s2_max": s2_max.tolist(), "missing_s2_fill": 0,
            "band_order": "S1 followed by S2", "augmentations": False,
        },
    }


def validate_manifest(output_dir, expected, allow_source_relocation=False):
    path = output_dir / "manifest.json"
    if path.exists():
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise ValueError(f"Cannot read {path}: {exc}") from exc
        if allow_source_relocation and isinstance(actual, dict):
            identity = lambda manifest: {key: value for key, value in manifest.items() if key != "input_directory"}
            compatible = identity(actual) == identity(expected)
        else:
            compatible = actual == expected
        if not compatible:
            raise ValueError(
                f"Incompatible manifest in {output_dir}; use a separate output directory. "
                "--overwrite does not permit mixing extraction settings. If only the "
                "input path changed for the same TIFFs, use --allow-source-relocation."
            )
        if actual.get("input_directory") != expected.get("input_directory"):
            print(f"Resuming relocated source: {actual.get('input_directory')} -> {expected.get('input_directory')}", flush=True)
    elif output_dir.exists() and any(output_dir.rglob("*.npy")):
        raise ValueError(f"Existing NPY files in {output_dir} have no manifest")


def valid_sample(output_dir, sample_id):
    for scale, shape in SCALES.items():
        path = output_dir / scale / f"{sample_id}.npy"
        try:
            # Read only the NPY header and file size. Some Drive/FUSE mounts do
            # not support mmap; loading entire arrays would waste remote reads.
            with path.open("rb") as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    actual_shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    actual_shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
                else:
                    return False
                if actual_shape != shape or dtype != np.dtype(np.float32):
                    return False
                expected_size = handle.tell() + int(np.prod(shape)) * dtype.itemsize
                if os.fstat(handle.fileno()).st_size != expected_size:
                    return False
        except (OSError, ValueError, EOFError):
            return False
    return True


def existing_sample_ids(output_dir):
    """Find candidates using directory listings, avoiding one stat per missing sample."""
    complete = None
    for scale in SCALES:
        directory = output_dir / scale
        if not directory.is_dir():
            return set()
        with os.scandir(directory) as entries:
            names = {
                entry.name[:-4] for entry in entries
                if entry.name.endswith(".npy") and entry.is_file()
            }
        complete = names if complete is None else complete & names
        if not complete:
            return set()
    return complete


def allocation_unit_size(directory):
    """Return filesystem allocation granularity, including exFAT cluster sizes."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        volume_path = ctypes.create_unicode_buffer(32768)
        get_volume = kernel.GetVolumePathNameW
        get_volume.argtypes = (wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD)
        get_volume.restype = wintypes.BOOL
        if not get_volume(str(directory), volume_path, len(volume_path)):
            raise ctypes.WinError(ctypes.get_last_error())
        get_space = kernel.GetDiskFreeSpaceW
        get_space.argtypes = (wintypes.LPCWSTR,) + (ctypes.POINTER(wintypes.DWORD),) * 4
        get_space.restype = wintypes.BOOL
        values = [wintypes.DWORD() for _ in range(4)]
        if not get_space(volume_path.value, *(ctypes.byref(value) for value in values)):
            raise ctypes.WinError(ctypes.get_last_error())
        return values[0].value * values[1].value
    info = os.statvfs(directory)
    return info.f_frsize or info.f_bsize


def allocated_sample_bytes(allocation_unit):
    if allocation_unit < 1:
        raise ValueError("Filesystem allocation unit must be positive")
    total = 0
    for shape in SCALES.values():
        header = bytes_io.BytesIO()
        np.lib.format.write_array_header_1_0(header, {
            "descr": np.lib.format.dtype_to_descr(np.dtype(np.float32)),
            "fortran_order": False, "shape": shape,
        })
        file_bytes = int(np.prod(shape)) * 4 + header.tell()
        total += ((file_bytes + allocation_unit - 1) // allocation_unit) * allocation_unit
    return total


def check_disk_space(output_dir, sample_count):
    existing_parent = output_dir
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    free = shutil.disk_usage(existing_parent).free
    allocation_unit = allocation_unit_size(existing_parent)
    # Round EACH file to full clusters: a 128-byte header can add a whole cluster.
    per_sample = allocated_sample_bytes(allocation_unit)
    required = sample_count * per_sample
    # Include directory metadata and the atomic-write temporary file.
    if sample_count:
        required += 64 * 1024 * 1024
    if free < required:
        raise ValueError(
            f"Insufficient disk space: {sample_count:,} remaining samples need "
            f"{required / 2**30:.2f} GiB; {free / 2**30:.2f} GiB available. "
            f"Allocation unit: {allocation_unit:,} bytes; "
            f"each sample occupies {per_sample / 2**20:.2f} MiB on disk. "
            "Reduce --max-samples or choose another --output-dir."
        )
    return required, free


def atomic_write(path, writer):
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_sample(output_dir, sample_id, features):
    if len(features) != len(SCALES):
        raise ValueError(f"Expected five encoder outputs for {sample_id}")
    for array, (scale, shape) in zip(features, SCALES.items()):
        if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"Invalid encoder output for {sample_id}, scale {scale}")
    for array, scale in zip(features, SCALES):
        path = output_dir / scale / f"{sample_id}.npy"
        atomic_write(path, lambda handle: np.save(handle, array, allow_pickle=False))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted complete modelo_best.pth")
    parser.add_argument("--train-features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu")
    parser.add_argument("--max-samples", type=int, default=100, help="Chip/month samples; 0 means all")
    parser.add_argument("--overwrite", action="store_true", help="Recompute selected samples with matching manifest")
    parser.add_argument("--allow-source-relocation", action="store_true",
                        help="Resume the same dataset after its input path changes; all other manifest fields must match")
    parser.add_argument("--skip-disk-space-check", action="store_true",
                        help="Skip local free-space preflight for remote mounts; verify remote storage quota yourself")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0 or args.max_samples < 0:
        parser.error("batch size must be positive; workers and max samples must be nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    checkpoint = args.checkpoint.resolve()
    directory = args.train_features_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint not found: {checkpoint}")
    if not directory.is_dir():
        raise ValueError(f"Input directory not found: {directory}")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path is not a directory: {output_dir}")
    samples = discover_samples(directory, args.max_samples)
    manifest = make_manifest(checkpoint, directory)
    validate_manifest(output_dir, manifest, args.allow_source_relocation)
    existing = existing_sample_ids(output_dir) if not args.overwrite else set()
    pending = samples if args.overwrite else [
        sample for sample in tqdm(samples, desc="Checking outputs")
        if sample[0] not in existing or not valid_sample(output_dir, sample[0])
    ]
    if args.skip_disk_space_check:
        required, free = len(pending) * allocated_sample_bytes(1), None
    else:
        required, free = check_disk_space(output_dir, len(pending))
    print(f"Selected {len(samples):,}; skipped {len(samples)-len(pending):,}; remaining {len(pending):,}", flush=True)
    if free is None:
        print(f"Remaining NPY file contents: {required/2**30:.2f} GiB. "
              "Free-space preflight skipped; ensure your remote account has enough quota.", flush=True)
    else:
        print(f"Required {required/2**30:.2f} GiB; available {free/2**30:.2f} GiB", flush=True)
    if not pending:
        print("All selected samples are complete.", flush=True)
        return
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA extraction are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; select --device cpu or install CUDA-enabled PyTorch")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder = load_encoder(checkpoint, device)
    for scale in SCALES:
        (output_dir / scale).mkdir(parents=True, exist_ok=True)
    atomic_write(output_dir / "manifest.json", lambda handle: handle.write(
        (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    ))
    loader = DataLoader(
        ChipMonthDataset(directory, pending), batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False, pin_memory=device.type == "cuda",
    )
    with torch.inference_mode(), tqdm(total=len(pending), desc="Extracting chip/months") as progress:
        for sample_ids, images in loader:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            features = [feature.float().cpu().numpy() for feature in encoder(images)]
            for index, sample_id in enumerate(sample_ids):
                save_sample(output_dir, sample_id, [feature[index] for feature in features])
                progress.update(1)
    print(f"Finished: {len(samples):,} selected samples, {len(samples)*5:,} feature files in {output_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Extraction failed: {exc}", file=sys.stderr)
        sys.exit(1)
