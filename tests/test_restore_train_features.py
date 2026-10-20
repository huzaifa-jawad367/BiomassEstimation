from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from restore_train_features import restore


class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / 'train.zip'
        self.output = self.root / 'inputs'
        with zipfile.ZipFile(self.archive, 'w') as archive:
            archive.writestr('Volumes/data/train_features/chip_S1_00.tif', b'sentinel1')
            archive.writestr('Volumes/data/train_features/chip_S2_00.tif', b'sentinel2')
            archive.writestr('../../unrelated.txt', b'ignore')

    def test_restore_repair_same_size_corruption_and_skip_valid(self):
        restore(self.archive, self.output, expected_s1=1)
        valid = self.output / 'chip_S1_00.tif'
        original_time = valid.stat().st_mtime_ns
        (self.output / 'chip_S2_00.tif').write_bytes(b'corrupt!!')
        restore(self.archive, self.output, expected_s1=1)
        self.assertEqual(valid.stat().st_mtime_ns, original_time)
        self.assertEqual((self.output / 'chip_S2_00.tif').read_bytes(), b'sentinel2')
        self.assertEqual(len(list(self.output.iterdir())), 2)
        self.assertFalse((self.root.parent / 'unrelated.txt').exists())

    def test_wrong_inventory_stops_before_writing_tiffs(self):
        with self.assertRaisesRegex(ValueError, 'expected 104,268'):
            restore(self.archive, self.output)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_duplicate_flattened_name_rejected_before_writing(self):
        with zipfile.ZipFile(self.archive, 'a') as archive:
            archive.writestr('other/chip_S1_00.tif', b'conflict')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            restore(self.archive, self.output, expected_s1=1)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_unexpected_existing_samples_rejected(self):
        self.output.mkdir()
        (self.output / 'other_S1_00.tif').write_bytes(b'keep')
        with self.assertRaisesRegex(ValueError, 'unexpected'):
            restore(self.archive, self.output, expected_s1=1)
        self.assertEqual((self.output / 'other_S1_00.tif').read_bytes(), b'keep')


if __name__ == '__main__':
    unittest.main()
