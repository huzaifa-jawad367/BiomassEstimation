"""Copy a local feature shard into a shared cache, optionally clearing verified files."""

import argparse
import hashlib
import json
from pathlib import Path

from tqdm import tqdm

from extract_features import SCALES, atomic_write, checkpoint_sha256, validate_manifest, valid_sample


def check_layout(root):
    for scale in SCALES:
        directory = root / scale
        if directory.exists() and directory.resolve().parent != root:
            raise ValueError(f"Resolution directory escapes cache root: {directory}")


def shard_ids(source):
    names = set()
    for scale in SCALES:
        directory = source / scale
        if directory.is_dir():
            names.update(path.stem for path in directory.glob("*.npy") if path.is_file())
    return sorted(names)


def copy_verified(source, target):
    digest = hashlib.sha256()

    def writer(handle):
        with source.open("rb") as incoming:
            for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
                digest.update(chunk)
                handle.write(chunk)

    atomic_write(target, writer)
    if checkpoint_sha256(target) != digest.hexdigest():
        raise OSError(f"Copied file failed SHA-256 verification: {target}")


def merge_shard(source, output, remove_source=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Source and output must be separate, non-nested cache directories")
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1 or manifest.get("output_shapes") != {
        name: list(shape) for name, shape in SCALES.items()
    } or manifest.get("dtype") != "float32" or not manifest.get("checkpoint_sha256"):
        raise ValueError(f"Invalid shard manifest: {manifest_path}")
    check_layout(source)
    check_layout(output)
    validate_manifest(output, manifest, allow_source_relocation=True)
    for scale in SCALES:
        (output / scale).mkdir(parents=True, exist_ok=True)
    if not (output / "manifest.json").exists():
        atomic_write(output / "manifest.json", lambda handle: handle.write(
            (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
        ))
    ids = shard_ids(source)
    copied = 0
    for sample_id in tqdm(ids, desc="Merging verified chip/months"):
        paths = [(source / scale / f"{sample_id}.npy", output / scale / f"{sample_id}.npy")
                 for scale in SCALES]
        for incoming, target in paths:
            if incoming.is_symlink() or incoming.resolve().parent.parent != source:
                raise ValueError(f"Source file escapes cache root: {incoming}")
            if target.is_symlink():
                raise ValueError(f"Destination feature must not be a symlink: {target}")
        if not valid_sample(source, sample_id):
            # Recover an interrupted cleanup: some local files may already have
            # been removed after the complete destination was verified.
            remaining = [(incoming, target) for incoming, target in paths if incoming.exists()]
            if (remove_source and valid_sample(output, sample_id)
                    and all(checkpoint_sha256(incoming) == checkpoint_sha256(target)
                            for incoming, target in remaining)):
                for incoming, _ in remaining:
                    incoming.unlink()
                continue
            raise ValueError(f"Incomplete or invalid local sample {sample_id}; rerun extraction before merging")
        # Compare existing files too: removal never discards a local file until
        # the matching destination has been verified byte for byte.
        for incoming, target in paths:
            if target.is_file() and checkpoint_sha256(incoming) == checkpoint_sha256(target):
                continue
            copy_verified(incoming, target)
            copied += 1
        if not valid_sample(output, sample_id):
            raise OSError(f"Merged sample failed validation: {sample_id}")
        if remove_source:
            for incoming, _ in paths:
                incoming.unlink()
    print(f"Merged {len(ids):,} samples; copied/repaired {copied:,} files into {output}.", flush=True)
    if remove_source:
        print("Verified local NPY files removed; manifest and resolution folders retained for resume.", flush=True)
    return len(ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--remove-source", action="store_true",
                        help="Remove each local sample only after all five destination files are verified")
    args = parser.parse_args()
    merge_shard(args.source_dir, args.output_dir, args.remove_source)


if __name__ == "__main__":
    main()
