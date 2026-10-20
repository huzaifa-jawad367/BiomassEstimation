"""Repair the Drive TIFF directory from a split ZIP, without staging all TIFFs."""

import argparse
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import zipfile
import zlib


PATTERN = re.compile(r"^(.+)_S([12])_(\d{2})\.tif$")


def members(archive, expected_s1):
    result = {}
    for info in archive.infolist():
        name = PurePosixPath(info.filename).name
        match = PATTERN.fullmatch(name)
        if info.is_dir() or not match:
            continue
        if name in result or not 0 <= int(match[3]) < 12:
            raise ValueError(f"Duplicate filename or invalid month in archive: {name}")
        result[name] = info
    count = sum('_S1_' in name for name in result)
    if not result or (expected_s1 and count != expected_s1):
        raise ValueError(f"Archive contains {count:,} S1 samples; expected {expected_s1:,}. No TIFFs written.")
    return result


def matches(path, info):
    if path.is_symlink():
        raise ValueError(f"Refusing a symlink TIFF destination: {path}")
    if not path.is_file() or path.stat().st_size != info.file_size:
        return False
    crc = 0
    with path.open('rb') as incoming:
        for chunk in iter(lambda: incoming.read(1024 * 1024), b''):
            crc = zlib.crc32(chunk, crc)
    return crc == info.CRC


def restore(archive_path, output, expected_s1=104268):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    with zipfile.ZipFile(archive_path) as archive:
        inventory = members(archive, expected_s1)
        print(f"Archive: {len(inventory):,} S1/S2 TIFFs. Checking existing files by size and CRC.", flush=True)
        for index, (name, info) in enumerate(inventory.items(), 1):
            target = output / name
            if matches(target, info):
                skipped += 1
            else:
                descriptor, temporary = tempfile.mkstemp(prefix=f'.{name}.', suffix='.tmp', dir=output)
                try:
                    with os.fdopen(descriptor, 'wb') as outgoing, archive.open(info) as incoming:
                        # Reading to EOF also validates the ZIP member CRC.
                        shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
                    os.replace(temporary, target)
                    if not matches(target, info):
                        raise OSError(f"Destination verification failed: {target}")
                    copied += 1
                finally:
                    Path(temporary).unlink(missing_ok=True)
            if index % 250 == 0 or index == len(inventory):
                print(f"{index:,}/{len(inventory):,}: restored {copied:,}; verified existing {skipped:,}", flush=True)
        actual = {p.name for p in output.iterdir() if p.is_file() and PATTERN.fullmatch(p.name)}
        if actual != set(inventory):
            raise ValueError("TIFF directory contains unexpected files; resolve these before sharding: "
                             + str(sorted(actual - set(inventory))[:10]))
    print(f"Verified complete input inventory: {len(inventory):,} TIFFs.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--local-dir', type=Path, default=Path('/content/biomass_archive_restore'))
    parser.add_argument('--expected-s1', type=int, default=104268)
    parser.add_argument('--reserve-gib', type=float, default=8)
    args = parser.parse_args()
    archive = args.archive.resolve()
    if not archive.is_file() or args.reserve_gib < 0 or args.expected_s1 < 0:
        raise ValueError('Require an existing .zip archive and nonnegative reserve/count')
    parts = sorted(archive.parent.glob(archive.stem + '.z[0-9][0-9]'))
    if parts:
        expected = [archive.with_suffix(f'.z{i:02d}') for i in range(1, len(parts) + 1)]
        if parts != expected:
            raise ValueError('Split archive parts are missing or not contiguous from .z01')
        if not shutil.which('zip'):
            raise ValueError('Install Info-ZIP first: apt-get install -y zip')
        args.local_dir.mkdir(parents=True, exist_ok=True)
        # Rebuild a temporary combined archive on every run; never reuse an
        # interrupted conversion. Only compressed data lives on local disk.
        required = sum(p.stat().st_size for p in parts + [archive]) + int(args.reserve_gib * 2**30)
        if required > shutil.disk_usage(args.local_dir).free:
            raise ValueError(f'Insufficient local space to combine archive: need {required/2**30:.2f} GiB')
        with tempfile.TemporaryDirectory(prefix='restore_', dir=args.local_dir) as directory:
            combined = Path(directory) / 'combined.zip'
            print(f'Combining {len(parts) + 1} archive parts locally.', flush=True)
            subprocess.run(['zip', '-s', '0', str(archive), '--out', str(combined)], check=True, stdin=subprocess.DEVNULL)
            restore(combined, args.output_dir, args.expected_s1)
    else:
        restore(archive, args.output_dir, args.expected_s1)


if __name__ == '__main__':
    main()
