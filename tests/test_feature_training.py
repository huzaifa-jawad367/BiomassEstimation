import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import dataset
import extract_features
import feature_dataset
import models
import train


CHANNELS = [32, 64, 96, 224, 640]


def model_args(feature_mode=True):
    return SimpleNamespace(
        feature_mode=feature_mode, feature_channels=CHANNELS,
        backbone="tf_efficientnetv2_l.in21k", in_channels=15,
        dec_channels=[8, 8, 8, 8, 8], dec_attn_type=None, n_classes=1,
    )


def feature_batch():
    return [torch.randn(1, 2, channels, 16 // 2**i, 16 // 2**i)
            for i, channels in enumerate(CHANNELS)]


class FakeEncoder(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.out_channels = [15] + CHANNELS
        self.register_buffer("marker", torch.tensor(1.0))
        self.features = features

    def forward(self, images):
        return [value.flatten(0, 1) for value in self.features]


class FeatureTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        checkpoint = self.root / "source.pth"
        checkpoint.write_bytes(b"source")
        self.manifest = extract_features.make_manifest(checkpoint, self.root)
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    def test_cli_feature_directory_required_and_image_default(self):
        self.assertFalse(train.parse_args([]).feature_mode)
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            train.parse_args(["--feature-mode"])
        args = train.parse_args(["--feature_mode", "--train-features-dir", str(self.root)])
        self.assertTrue(args.feature_mode)

    def test_feature_model_has_no_encoder_and_updates_attention_decoder(self):
        with patch.object(models, "TimmEncoder", side_effect=AssertionError("Encoder constructed")):
            model = models.UnetVFLOW(model_args())
        self.assertIsNone(model.encoder)
        self.assertFalse(any(name.startswith("encoder.") for name in model.state_dict()))
        features = feature_batch()
        mask = torch.tensor([[False, False]])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        before = model.attn[0].attn[0].weight.detach().clone()
        output = model(features, mask)
        self.assertEqual(output.shape, (1, 1, 32, 32))
        output.square().mean().backward()
        for module in (model.attn, model.decoder, model.segmentation_head):
            self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) for p in module.parameters()))
        self.assertTrue(all(value.grad is None for value in features))
        optimizer.step()
        self.assertFalse(torch.equal(before, model.attn[0].attn[0].weight))

    def test_image_and_cached_entry_points_agree(self):
        features = feature_batch()
        with patch.object(models, "TimmEncoder", return_value=FakeEncoder(features)):
            image_model = models.UnetVFLOW(model_args(False)).eval()
        feature_model = models.UnetVFLOW(model_args()).eval()
        state = {k: v for k, v in image_model.state_dict().items() if not k.startswith("encoder.")}
        feature_model.load_state_dict(state, strict=True)
        mask = torch.tensor([[False, False]])
        with torch.no_grad():
            image_result = image_model(torch.zeros(1, 2, 15, 32, 32), mask)
            feature_result = feature_model(features, mask)
        torch.testing.assert_close(image_result, feature_result, rtol=0, atol=0)

    def test_masked_month_does_not_affect_prediction(self):
        model = models.UnetVFLOW(model_args()).eval()
        features = feature_batch()
        changed = [value.clone() for value in features]
        for value in changed:
            value[:, 1] += 10000
        mask = torch.tensor([[False, True]])
        with torch.no_grad():
            torch.testing.assert_close(model(features, mask), model(changed, mask), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "at least one"):
            model(features, torch.ones_like(mask))

    def test_dataset_groups_months_without_reading_images(self):
        shapes = dict(feature_dataset.ordered_shapes(self.manifest))
        def load(path, **kwargs):
            self.assertFalse(kwargs["allow_pickle"])
            month = int(Path(path).stem.rsplit("_", 1)[1])
            return np.full(shapes[Path(path).parent.name], month, dtype=np.float32)
        data = feature_dataset.FeatureDS(pd.DataFrame({"chip_id": ["chip"]}), self.root, self.root)
        with patch.object(feature_dataset.np, "load", side_effect=load), \
                patch.object(feature_dataset.io, "imread", return_value=np.zeros((256, 256), dtype=np.float32)):
            features, mask, target = data[0]
        self.assertEqual(len(features), 5)
        self.assertEqual(features[0].shape, (12, 32, 128, 128))
        self.assertEqual(features[-1][11, 0, 0, 0], 11)
        self.assertFalse(mask.any())
        self.assertEqual(target.dtype, np.float32)

    def test_dropout_keeps_one_month_and_zeros_all_scales(self):
        values = [np.ones((12, 2, 2, 2), dtype=np.float32) for _ in range(5)]
        with patch.object(np.random, "rand", return_value=np.zeros(12)), \
                patch.object(np.random, "randint", return_value=3):
            mask = feature_dataset.month_dropout(values, np.zeros(12, dtype=bool))
        self.assertEqual(np.count_nonzero(~mask), 1)
        for value in values:
            self.assertTrue((value[mask] == 0).all())
            self.assertTrue((value[3] == 1).all())

    def test_coverage_requires_every_month_and_scale(self):
        for scale, _ in feature_dataset.ordered_shapes(self.manifest):
            (self.root / scale).mkdir()
            for month in range(12):
                (self.root / scale / f"chip_{month:02d}.npy").touch()
        feature_dataset.validate_feature_coverage(self.root, ["chip"], self.manifest)
        (self.root / "8x8" / "chip_11.npy").unlink()
        with self.assertRaisesRegex(ValueError, "chip_11.*12"):
            feature_dataset.validate_feature_coverage(self.root, ["chip"], self.manifest)

    def test_build_model_uses_manifest_channels_without_encoder(self):
        args = train.parse_args(["--feature-mode", "--train-features-dir", str(self.root)])
        with patch.object(models, "TimmEncoder", side_effect=AssertionError("Encoder constructed")):
            model = train.build_model(args)
        self.assertEqual(args.feature_channels, CHANNELS)
        self.assertIsNone(model.encoder)

    def test_image_dataset_path_remains_default(self):
        args = train.parse_args([])
        frame = pd.DataFrame({"chip_id": ["chip"]})
        training, validation = train.build_datasets(frame, frame, frame.iloc[:0], args)
        self.assertIsInstance(training, dataset.DS)
        self.assertIsInstance(validation, dataset.DS)

    def test_feature_validation_uses_one_prediction_without_tta(self):
        class CountingModel(nn.Module):
            feature_mode = True
            def __init__(self):
                super().__init__()
                self.calls = 0
            def forward(self, features, mask):
                self.calls += 1
                return torch.ones((1, 1, 4, 4))
        model = CountingModel()
        loader = [(feature_batch(), torch.zeros(1, 2, dtype=torch.bool), torch.zeros(1, 4, 4))]
        with patch.object(torch.Tensor, "cuda", lambda value, *a, **k: value), \
                patch.object(torch.cuda, "synchronize"):
            loss, count, metrics = train.epoch_step_val(loader, "test", model, train.RMSE(), local_rank=1)
        self.assertEqual(model.calls, 1)
        self.assertEqual((loss, count, metrics), (1.0, 1, {"tta_1": 1.0}))

    def test_image_checkpoint_initializes_feature_heads_and_resume_checks_provenance(self):
        features = feature_batch()
        with patch.object(models, "TimmEncoder", return_value=FakeEncoder(features)):
            image_model = models.UnetVFLOW(model_args(False))
        feature_model = models.UnetVFLOW(model_args())
        path = self.root / "weights.pth"
        torch.save({"state_dict": image_model.state_dict(), "args": model_args(False)}, path)
        args = model_args()
        args.load, args.resume, args.feature_manifest = str(path), False, self.manifest
        train.load_training_checkpoint(feature_model, args)
        torch.testing.assert_close(feature_model.segmentation_head.weight, image_model.segmentation_head.weight)
        args.resume = True
        with self.assertRaisesRegex(ValueError, "between image and feature"):
            train.load_training_checkpoint(feature_model, args)
        saved = copy.deepcopy(args)
        torch.save({"state_dict": feature_model.state_dict(), "args": saved}, path)
        train.load_training_checkpoint(feature_model, args)
        args.feature_manifest = dict(self.manifest, input_directory="/content/relocated")
        train.load_training_checkpoint(feature_model, args)
        args.feature_manifest = dict(self.manifest, checkpoint_sha256="different")
        with self.assertRaisesRegex(ValueError, "different encoder"):
            train.load_training_checkpoint(feature_model, args)


if __name__ == "__main__":
    unittest.main()
