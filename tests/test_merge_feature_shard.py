import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import extract_features as extraction
import merge_feature_shard as merge


class MergeShardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.output = self.root / "merged"
        checkpoint = self.root / "checkpoint.pth"
        checkpoint.write_bytes(b"checkpoint")
        self.manifest = extraction.make_manifest(checkpoint, self.root)

    def create_shard(self, name, sample_id):
        source = self.root / name
        source.mkdir()
        (source / "manifest.json").write_text(json.dumps(self.manifest))
        for scale, shape in extraction.SCALES.items():
            (source / scale).mkdir()
            np.save(source / scale / f"{sample_id}.npy", np.zeros(shape, dtype=np.float32))
        return source

    def test_three_shards_merge_into_same_flat_directory_and_resume(self):
        for shard in (1, 2, 3):
            source = self.create_shard(f"shard{shard}", f"chip{shard}_00")
            merge.merge_shard(source, self.output, remove_source=True)
            self.assertFalse(list(source.rglob("*.npy")))
            self.assertTrue(extraction.valid_sample(self.output, f"chip{shard}_00"))
            merge.merge_shard(source, self.output, remove_source=True)
        self.assertEqual(len(list(self.output.rglob("*.npy"))), 15)
        self.assertFalse(list(self.output.glob("shard*")))

    def test_copy_failure_preserves_local_files(self):
        source = self.create_shard("source", "chip_00")
        with patch.object(merge, "copy_verified", side_effect=OSError("Drive quota exceeded")):
            with self.assertRaises(OSError):
                merge.merge_shard(source, self.output, remove_source=True)
        self.assertTrue(extraction.valid_sample(source, "chip_00"))

    def test_hash_verification_failure_never_removes_local_files(self):
        source = self.create_shard("source", "chip_00")
        with patch.object(merge, "checkpoint_sha256", return_value="bad-destination-hash"):
            with self.assertRaisesRegex(OSError, "SHA-256"):
                merge.merge_shard(source, self.output, remove_source=True)
        self.assertTrue(extraction.valid_sample(source, "chip_00"))

    def test_manifest_mismatch_preserves_files_and_rejects_merge(self):
        source = self.create_shard("source", "chip_00")
        self.output.mkdir()
        (self.output / "manifest.json").write_text(json.dumps(dict(self.manifest, checkpoint_sha256="different")))
        with self.assertRaisesRegex(ValueError, "Incompatible manifest"):
            merge.merge_shard(source, self.output, remove_source=True)
        self.assertTrue(extraction.valid_sample(source, "chip_00"))

    def test_interrupted_local_cleanup_can_resume(self):
        source = self.create_shard("source", "chip_00")
        merge.merge_shard(source, self.output)
        (source / "128x128" / "chip_00.npy").unlink()
        merge.merge_shard(source, self.output, remove_source=True)
        self.assertFalse(list(source.rglob("*.npy")))
        self.assertTrue(extraction.valid_sample(self.output, "chip_00"))

    def test_nested_or_same_paths_are_rejected(self):
        source = self.create_shard("source", "chip_00")
        for output in (source, source / "nested", self.root):
            with self.assertRaisesRegex(ValueError, "non-nested"):
                merge.merge_shard(source, output, remove_source=True)
        self.assertTrue(extraction.valid_sample(source, "chip_00"))


if __name__ == "__main__":
    unittest.main()
