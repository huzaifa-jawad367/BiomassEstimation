import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import extract_features as extraction


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_pair_normalization_and_nodata(self):
        s1 = np.broadcast_to(extraction.s1_min + extraction.s1_mm, (256, 256, 4)).copy()
        s1[0, 0, 0] = -9999
        s2 = np.broadcast_to(extraction.s2_max, (256, 256, 11)).copy()
        (self.root / "chip_S2_00.tif").touch()
        with patch.object(extraction.io, "imread", side_effect=[s1, s2]):
            image = extraction.read_pair(self.root, "chip_S1_00.tif")
        self.assertEqual(image.shape, (15, 256, 256))
        self.assertEqual(image.dtype, np.float32)
        self.assertTrue(image.flags.c_contiguous)
        self.assertEqual(image[0, 0, 0], 0)
        self.assertEqual(image[:, 1:, :].min(), 1)
        np.testing.assert_array_equal(image[4:], 1)

    def test_missing_s2_is_zero_filled(self):
        s1 = np.broadcast_to(extraction.s1_min, (256, 256, 4)).copy()
        with patch.object(extraction.io, "imread", return_value=s1) as reader:
            image = extraction.read_pair(self.root, "chip_S1_02.tif")
        reader.assert_called_once()
        np.testing.assert_array_equal(image, 0)

    def test_malformed_image_identifies_source(self):
        with patch.object(extraction.io, "imread", return_value=np.zeros((10, 10, 4))):
            with self.assertRaisesRegex(ValueError, "chip_S1_00.tif.*expected shape"):
                extraction.read_pair(self.root, "chip_S1_00.tif")

    def test_selection_is_sorted_and_counts_months(self):
        for name in ("b_S1_01.tif", "a_S2_00.tif", "a_S1_01.tif", "a_S1_00.tif"):
            (self.root / name).touch()
        self.assertEqual(extraction.discover_samples(self.root, 2), [
            ("a_00", "a_S1_00.tif"), ("a_01", "a_S1_01.tif"),
        ])
        self.assertEqual(len(extraction.discover_samples(self.root, 0)), 3)

    def test_saved_arrays_are_numeric_and_incomplete_sample_is_repaired(self):
        features = [np.zeros(shape, dtype=np.float32) for shape in extraction.SCALES.values()]
        for scale in extraction.SCALES:
            (self.root / scale).mkdir()
        extraction.save_sample(self.root, "chip_00", features)
        self.assertTrue(extraction.valid_sample(self.root, "chip_00"))
        self.assertEqual(extraction.existing_sample_ids(self.root), {"chip_00"})
        missing = self.root / "8x8" / "chip_00.npy"
        missing.write_bytes(b"truncated")
        self.assertFalse(extraction.valid_sample(self.root, "chip_00"))
        extraction.save_sample(self.root, "chip_00", features)
        self.assertTrue(extraction.valid_sample(self.root, "chip_00"))
        array = np.load(missing, allow_pickle=False)
        self.assertEqual(array.dtype, np.float32)
        self.assertFalse(list(self.root.rglob("*.tmp")))

    def test_missing_resolution_excludes_sample_without_opening_arrays(self):
        for scale in list(extraction.SCALES)[:-1]:
            (self.root / scale).mkdir()
            (self.root / scale / "chip_00.npy").touch()
        self.assertEqual(extraction.existing_sample_ids(self.root), set())

    def test_wrong_dtype_is_not_complete(self):
        features = [np.zeros(shape, dtype=np.float32) for shape in extraction.SCALES.values()]
        for scale in extraction.SCALES:
            (self.root / scale).mkdir()
        extraction.save_sample(self.root, "chip_00", features)
        np.save(self.root / "8x8" / "chip_00.npy", features[-1].astype(np.float16))
        self.assertFalse(extraction.valid_sample(self.root, "chip_00"))

    def test_manifest_mismatch_and_untracked_outputs_are_rejected(self):
        manifest = {"checkpoint_sha256": "abc", "dtype": "float32"}
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        extraction.validate_manifest(self.root, manifest)
        with self.assertRaisesRegex(ValueError, "Incompatible manifest"):
            extraction.validate_manifest(self.root, dict(manifest, checkpoint_sha256="other"))
        (self.root / "manifest.json").unlink()
        np.save(self.root / "orphan.npy", np.zeros(1))
        with self.assertRaisesRegex(ValueError, "no manifest"):
            extraction.validate_manifest(self.root, manifest)

    def test_disk_space_checks_pending_samples_before_creation(self):
        with patch.object(extraction.shutil, "disk_usage", return_value=SimpleNamespace(free=223 * 2**30)):
            with self.assertRaisesRegex(ValueError, "Insufficient disk space"):
                extraction.check_disk_space(self.root / "new", 104268)
            extraction.check_disk_space(self.root / "new", 100)
        self.assertFalse((self.root / "new").exists())

    def test_large_exfat_clusters_include_npy_header_rounding(self):
        # Two maps are exact MiB multiples before their headers. They need an
        # additional cluster, giving 3+2+1+1+1 MiB rather than 3.75 MiB per sample.
        self.assertEqual(extraction.allocated_sample_bytes(2**20), 8 * 2**20)
        self.assertEqual(extraction.allocated_sample_bytes(4096), extraction.SAMPLE_BYTES + 5 * 4096)
        with patch.object(extraction, "allocation_unit_size", return_value=2**20), \
                patch.object(extraction.shutil, "disk_usage", return_value=SimpleNamespace(free=int(381.43 * 2**30))):
            with self.assertRaisesRegex(ValueError, "802.32 GiB.*8.00 MiB"):
                extraction.check_disk_space(self.root, 102689)

    def test_zero_pending_samples_need_no_extra_space(self):
        with patch.object(extraction, "allocation_unit_size", return_value=2**20), \
                patch.object(extraction.shutil, "disk_usage", return_value=SimpleNamespace(free=0)):
            required, free = extraction.check_disk_space(self.root, 0)
        self.assertEqual((required, free), (0, 0))

    def test_atomic_write_preserves_previous_file_on_failure(self):
        path = self.root / "output.npy"
        path.write_bytes(b"previous")
        def fail(handle):
            handle.write(b"partial")
            raise OSError("write failed")
        with self.assertRaises(OSError):
            extraction.atomic_write(path, fail)
        self.assertEqual(path.read_bytes(), b"previous")
        self.assertFalse(list(self.root.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
