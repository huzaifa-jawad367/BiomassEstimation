import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import colab_feature_workflow as workflow
import extract_features as extraction


class FakeEncoder:
    def __call__(self, images):
        return [torch.zeros(images.shape[0], *shape, device=images.device) for shape in extraction.SCALES.values()]


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "drive" / "train_features"
        self.remote = self.root / "drive" / "extracted_features"
        self.work = self.root / "local"
        self.source.mkdir(parents=True)
        self.checkpoint = self.root / "checkpoint.pth"
        self.checkpoint.write_bytes(b"trusted checkpoint")
        for chip in range(6):
            for month in range(2):
                (self.source / f"chip{chip}_S1_{month:02d}.tif").write_bytes(b"S1" * 512)
                if month == 0:
                    (self.source / f"chip{chip}_S2_{month:02d}.tif").write_bytes(b"S2" * 512)

    def prepare(self, **options):
        return workflow.prepare(self.checkpoint, self.source, self.remote, self.work,
                                num_shards=6, shard_id=2, reserve_gib=0, copy_workers=1, **options)

    def extract(self, plan):
        def read_local(path):
            self.assertTrue(Path(path).resolve().is_relative_to(self.work))
            return np.zeros((256, 256, 4 if "_S1_" in Path(path).name else 11), dtype=np.float32)
        with patch.object(extraction, "load_encoder", return_value=FakeEncoder()), \
                patch.object(extraction.io, "imread", side_effect=read_local):
            workflow.extract_local(plan, batch_size=2, num_workers=0, device="cpu")

    def test_full_local_read_extract_publish_cleanup_flow(self):
        path = self.prepare()
        _, plan = workflow.load_plan(path)
        inputs = Path(plan["local_inputs"])
        self.assertEqual({p.name for p in inputs.glob("*.tif")},
                         {"chip1_S1_00.tif", "chip1_S2_00.tif", "chip1_S1_01.tif"})
        for record in plan["inputs"]:
            self.assertEqual(extraction.checkpoint_sha256(inputs / record["name"]), record["sha256"])
        self.extract(path)
        workflow.publish(path, remove_local=True)
        self.assertEqual(len(list(self.remote.rglob("*.npy"))), 10)
        self.assertFalse(list(inputs.glob("*.tif")))
        self.assertFalse(list(Path(plan["local_outputs"]).rglob("*.npy")))
        self.assertEqual(len(list(self.source.glob("*.tif"))), 18)

    def test_combined_input_output_budget_checked_before_copy(self):
        output_bytes = 2 * extraction.allocated_sample_bytes(4096)
        with patch.object(extraction, "allocation_unit_size", return_value=4096), \
                patch.object(workflow.shutil, "disk_usage", return_value=SimpleNamespace(free=output_bytes + 4096)), \
                patch.object(workflow, "copy_verified", side_effect=AssertionError("Must preflight before copying")):
            with self.assertRaisesRegex(ValueError, "inputs AND outputs"):
                self.prepare()
        self.assertFalse(self.work.exists())

    def test_preparation_resumes_after_copy_failure(self):
        original = workflow.copy_verified
        calls = []
        def interrupted(source, output):
            calls.append(source.name)
            if len(calls) == 2:
                raise OSError("Drive input interrupted")
            return original(source, output)
        with patch.object(workflow, "copy_verified", side_effect=interrupted):
            with self.assertRaises(OSError):
                self.prepare()
        path = self.prepare()
        _, plan = workflow.load_plan(path)
        self.assertTrue(plan["prepared"])
        self.assertTrue(all(record["sha256"] for record in plan["inputs"]))

    def test_already_published_shard_needs_no_copies_or_gpu_work(self):
        path = self.prepare()
        self.extract(path)
        workflow.publish(path, remove_local=True)
        with patch.object(workflow, "copy_verified", side_effect=AssertionError("Already published")):
            path = self.prepare()
        with patch.object(extraction, "load_encoder", side_effect=AssertionError("Already published")):
            workflow.extract_local(path)
        workflow.publish(path, remove_local=True)

    def test_merge_failure_retains_local_tiffs(self):
        path = self.prepare()
        self.extract(path)
        with patch.object(workflow, "merge_shard", side_effect=OSError("Drive quota exceeded")):
            with self.assertRaises(OSError):
                workflow.publish(path, remove_local=True)
        _, plan = workflow.load_plan(path)
        self.assertEqual(len(list(Path(plan["local_inputs"]).glob("*.tif"))), 3)

    def test_cleanup_plan_cannot_point_at_original_drive_inputs(self):
        path = self.prepare()
        plan = json.loads(path.read_text())
        plan["local_inputs"] = str(self.source)
        path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "inside its staging"):
            workflow.publish(path, remove_local=True)
        self.assertEqual(len(list(self.source.glob("*.tif"))), 18)

    def test_published_samples_not_reextracted_from_leftover_local_tiffs(self):
        path = self.prepare()
        self.extract(path)
        _, plan = workflow.load_plan(path)
        local = Path(plan["local_outputs"])
        # Publish only month 00, mimicking an interrupted merge.
        manifest = json.loads((local / "manifest.json").read_text())
        for scale in extraction.SCALES:
            (self.remote / scale).mkdir(parents=True)
            workflow.copy_verified(local / scale / "chip1_00.npy", self.remote / scale / "chip1_00.npy")
        (self.remote / "manifest.json").write_text(json.dumps(manifest))
        for array in local.rglob("*.npy"):
            array.unlink()
        path = self.prepare()
        _, plan = workflow.load_plan(path)
        self.assertEqual(plan["pending_samples"], ["chip1_01"])
        self.assertEqual(len(list(Path(plan["local_inputs"]).glob("*.tif"))), 3)
        self.extract(path)
        self.assertFalse(list(local.glob("*/chip1_00.npy")))
        self.assertEqual(len(list(local.glob("*/chip1_01.npy"))), 5)
        workflow.publish(path, remove_local=True)


if __name__ == "__main__":
    unittest.main()
