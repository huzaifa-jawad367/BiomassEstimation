"""Stage raw TIFFs locally, extract locally, then publish verified features to Drive."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import stat

from tqdm import tqdm

import extract_features as extraction
from merge_feature_shard import copy_verified, merge_shard


def write_json(path, value):
    extraction.atomic_write(path, lambda handle: handle.write(
        (json.dumps(value, indent=2) + "\n").encode("utf-8")
    ))


def separate_paths(first, second):
    return first != second and not first.is_relative_to(second) and not second.is_relative_to(first)


def stage_root(work_dir, num_shards, shard_id):
    return Path(work_dir).resolve() / f"shard_{shard_id:03d}_of_{num_shards:03d}"


def load_plan(path):
    path = Path(path).resolve()
    plan = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent
    if plan.get("version") != 1:
        raise ValueError(f"Unsupported staging plan: {path}")
    for key, name in (("local_inputs", "train_features"), ("local_outputs", "extracted_features")):
        if Path(plan[key]).resolve() != root / name or (root / name).is_symlink():
            raise ValueError(f"Plan {key} must stay inside its staging directory")
    for key in ("source_inputs", "drive_outputs"):
        if not separate_paths(root, Path(plan[key]).resolve()):
            raise ValueError("Staging directory must not overlap Drive/source directories")
    for record in plan["inputs"]:
        if Path(record["name"]).name != record["name"] or not record["name"].endswith(".tif"):
            raise ValueError("Invalid input filename in staging plan")
    return path, plan


def source_records(source, sample):
    sample_id, filename = sample
    chip, month = sample_id.rsplit("_", 1)
    records = []
    for name, optional in ((filename, False), (f"{chip}_S2_{month}.tif", True)):
        path = source / name
        try:
            info = path.stat()
        except FileNotFoundError:
            if optional:
                continue
            raise
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Input is not a TIFF file: {path}")
        records.append({"name": name, "size": info.st_size, "sha256": None})
    return records


def prepare(checkpoint, source, drive_output, work_dir, num_shards=6, shard_id=1,
            copy_workers=4, reserve_gib=8, max_samples=0):
    checkpoint, source, drive_output = (Path(path).resolve() for path in (checkpoint, source, drive_output))
    root = stage_root(work_dir, num_shards, shard_id)
    if copy_workers < 1 or reserve_gib < 0 or max_samples < 0:
        raise ValueError("Copy workers must be positive; reserve and sample limit must be nonnegative")
    if not checkpoint.is_file() or not source.is_dir():
        raise ValueError("Checkpoint and source TIFF directory must exist")
    if not separate_paths(root, source) or not separate_paths(root, drive_output):
        raise ValueError("Staging directory must be separate from source and Drive outputs")
    if root.is_symlink():
        raise ValueError("Staging root must not be a symlink")
    local_inputs, local_outputs = root / "train_features", root / "extracted_features"
    for directory in (local_inputs, local_outputs):
        if directory.is_symlink():
            raise ValueError(f"Staging subdirectory must not be a symlink: {directory}")
    selected = extraction.select_shard(extraction.discover_samples(source, 0), num_shards, shard_id)
    if max_samples:
        selected = selected[:max_samples]
    manifest = extraction.make_manifest(checkpoint, source)
    extraction.validate_manifest(drive_output, manifest, allow_source_relocation=True)
    local_manifest = dict(manifest, input_directory=str(local_inputs))
    extraction.validate_manifest(local_outputs, local_manifest, allow_source_relocation=True)
    identity = {
        "version": 1, "checkpoint": str(checkpoint), "checkpoint_sha256": manifest["checkpoint_sha256"],
        "source_inputs": str(source), "drive_outputs": str(drive_output),
        "local_inputs": str(local_inputs), "local_outputs": str(local_outputs),
        "num_shards": num_shards, "shard_id": shard_id,
        "selected_samples": [sample_id for sample_id, _ in selected],
    }
    plan_path = root / "plan.json"
    if plan_path.exists():
        _, previous = load_plan(plan_path)
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("Existing staging plan differs; keep the same configuration or use a new work directory")
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError(f"Refusing to reuse a nonempty staging directory without a plan: {root}")
        previous = {"inputs": []}
    remote_ids = extraction.existing_sample_ids(drive_output)
    local_ids = extraction.existing_sample_ids(local_outputs)
    pending = []
    for sample in tqdm(selected, desc="Checking published/staged features"):
        sample_id = sample[0]
        if sample_id in remote_ids and extraction.valid_sample(drive_output, sample_id):
            continue
        if sample_id in local_ids and extraction.valid_sample(local_outputs, sample_id):
            continue
        pending.append(sample)
    records = {record["name"]: dict(record) for record in previous["inputs"]}
    copies = []
    with ThreadPoolExecutor(max_workers=copy_workers) as pool:
        for group in tqdm(pool.map(lambda sample: source_records(source, sample), pending),
                          total=len(pending), desc="Sizing shard TIFFs"):
            for record in group:
                destination = local_inputs / record["name"]
                old = records.get(record["name"])
                if destination.is_symlink():
                    raise ValueError(f"Staged input must not be a symlink: {destination}")
                if destination.exists() and old is None:
                    raise ValueError(f"Refusing to overwrite an unowned staging file: {destination}")
                ready = (old and old.get("sha256") and old["size"] == record["size"]
                         and destination.is_file() and destination.stat().st_size == old["size"]
                         and extraction.checkpoint_sha256(destination) == old["sha256"])
                if not ready:
                    records[record["name"]] = record
                    copies.append(record)
    ancestor = root
    while not ancestor.exists():
        ancestor = ancestor.parent
    unit = extraction.allocation_unit_size(ancestor)
    input_bytes = sum(((record["size"] + unit - 1) // unit) * unit for record in copies)
    output_bytes = len(pending) * extraction.allocated_sample_bytes(unit)
    required = input_bytes + output_bytes + int(reserve_gib * 2**30)
    free = shutil.disk_usage(ancestor).free
    print(f"Shard {shard_id}/{num_shards}: {len(selected):,} selected, {len(pending):,} need extraction.", flush=True)
    print(f"Additional TIFFs {input_bytes/2**30:.2f} GiB + feature outputs {output_bytes/2**30:.2f} GiB "
          f"+ reserve {reserve_gib:.2f} GiB = {required/2**30:.2f} GiB; free {free/2**30:.2f} GiB.", flush=True)
    if required > free:
        raise ValueError("Insufficient local space for inputs AND outputs. Increase --num-shards "
                         "using a new work directory, or free verified local staging data.")
    local_inputs.mkdir(parents=True, exist_ok=True)
    for scale in extraction.SCALES:
        (local_outputs / scale).mkdir(parents=True, exist_ok=True)
    write_json(local_outputs / "manifest.json", local_manifest)
    plan = dict(identity, inputs=list(records.values()), pending_samples=[sample[0] for sample in pending],
                prepared=False, extracted=False, published=False)
    write_json(plan_path, plan)
    write_json(root / "samples.json", plan["pending_samples"])

    def copy_one(record):
        destination = local_inputs / record["name"]
        digest = copy_verified(source / record["name"], destination)
        return record["name"], digest

    try:
        with ThreadPoolExecutor(max_workers=copy_workers) as pool:
            for index, (name, digest) in enumerate(tqdm(pool.map(copy_one, copies), total=len(copies),
                                                       desc="Copying and verifying TIFFs locally"), 1):
                records[name]["sha256"] = digest
                if index % 256 == 0:
                    plan["inputs"] = list(records.values())
                    write_json(plan_path, plan)
        plan["prepared"] = True
    finally:
        plan["inputs"] = list(records.values())
        write_json(plan_path, plan)
    print(f"Prepared local inputs. Next run extract --plan {plan_path}", flush=True)
    return plan_path


def extract_local(plan_path, batch_size=32, num_workers=2, device="cuda"):
    plan_path, plan = load_plan(plan_path)
    if not plan.get("prepared"):
        raise ValueError("Run prepare successfully before extraction")
    if extraction.checkpoint_sha256(Path(plan["checkpoint"])) != plan["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed since preparation")
    needed = set(plan["pending_samples"])
    if needed:
        # The input folder already contains the chosen shard. Do NOT shard it
        # again, and do not consult Drive while the GPU is extracting features.
        extraction.main([
            "--checkpoint", plan["checkpoint"], "--train-features-dir", plan["local_inputs"],
            "--output-dir", plan["local_outputs"], "--sample-list", str(plan_path.parent / "samples.json"),
            "--max-samples", "0", "--batch-size", str(batch_size), "--num-workers", str(num_workers),
            "--device", device,
        ])
        if not all(extraction.valid_sample(Path(plan["local_outputs"]), sample_id) for sample_id in needed):
            raise ValueError("Local extraction did not produce every requested sample")
    else:
        print("All selected samples were already extracted; no GPU work needed.", flush=True)
    plan["extracted"] = True
    write_json(plan_path, plan)


def publish(plan_path, remove_local=False):
    plan_path, plan = load_plan(plan_path)
    source, target = Path(plan["local_outputs"]), Path(plan["drive_outputs"])
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    extraction.validate_manifest(target, manifest, allow_source_relocation=True)
    for sample_id in plan["pending_samples"]:
        if not extraction.valid_sample(source, sample_id) and not extraction.valid_sample(target, sample_id):
            raise ValueError(f"Sample {sample_id} is not extracted yet; run extract before merge")
    merge_shard(source, target, remove_source=remove_local)
    for sample_id in tqdm(plan["selected_samples"], desc="Verifying published shard"):
        if not extraction.valid_sample(target, sample_id):
            raise ValueError(f"Published shard incomplete at {sample_id}; retaining local TIFFs")
    if remove_local:
        inputs = Path(plan["local_inputs"])
        # Delete only plan-owned local copies, never recurse or touch Drive TIFFs.
        for record in plan["inputs"]:
            path = inputs / record["name"]
            if path.is_symlink() or path.resolve().parent != inputs:
                raise ValueError(f"Local cleanup path escapes staging directory: {path}")
            path.unlink(missing_ok=True)
        plan["prepared"] = False
        plan["extracted"] = False
    plan["published"] = True
    write_json(plan_path, plan)
    print(f"Shard published to {target}. " + ("Verified local TIFF/NPY copies cleared." if remove_local else "Local copies retained."), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    staging = commands.add_parser("prepare", help="Copy only this shard's pending S1/S2 TIFFs from Drive")
    staging.add_argument("--checkpoint", type=Path, required=True)
    staging.add_argument("--train-features-dir", type=Path, required=True)
    staging.add_argument("--drive-output-dir", type=Path, required=True)
    staging.add_argument("--work-dir", type=Path, default=Path("/content/biomass_feature_work"))
    staging.add_argument("--num-shards", type=int, default=6)
    staging.add_argument("--shard-id", type=int, default=1)
    staging.add_argument("--copy-workers", type=int, default=4)
    staging.add_argument("--reserve-gib", type=float, default=8)
    staging.add_argument("--max-samples", type=int, default=0)
    extracting = commands.add_parser("extract", help="Run GPU extraction using only staged local files")
    extracting.add_argument("--plan", type=Path, required=True)
    extracting.add_argument("--batch-size", type=int, default=32)
    extracting.add_argument("--num-workers", type=int, default=2)
    extracting.add_argument("--device", default="cuda")
    merging = commands.add_parser("merge", help="Publish to the shared Drive directory and verify the shard")
    merging.add_argument("--plan", type=Path, required=True)
    merging.add_argument("--remove-local", action="store_true", help="Clear owned local TIFFs and NPYs after successful publication")
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        args["source"] = args.pop("train_features_dir")
        args["drive_output"] = args.pop("drive_output_dir")
        prepare(**args)
    elif command == "extract":
        args["plan_path"] = args.pop("plan")
        extract_local(**args)
    else:
        args["plan_path"] = args.pop("plan")
        publish(**args)


if __name__ == "__main__":
    main()
