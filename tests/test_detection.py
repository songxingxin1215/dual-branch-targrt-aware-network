import sys
import unittest
import tempfile
from pathlib import Path

import numpy as np
import scipy.io
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.detection import (
    class_weights_for_labels,
    detection_metrics,
    load_detection_mat,
    make_balanced_sampler,
    select_best_threshold,
    split_detection_gt,
)
from utils.detection_runtime import evaluate_detector, predict_probability_map, train_one_epoch
from models.hsi_detector import TargetAwareHSIDetector
from utils.dataset import HSIDataset


class TestDetectionData(unittest.TestCase):
    def test_split_keeps_binary_background_and_target(self):
        gt = np.zeros((10, 10), dtype=np.int8)
        gt[:2, :] = 1
        train, val, test = split_detection_gt(gt, 0.2, 0.2, seed=7)

        self.assertEqual(np.unique(train).tolist(), [-1, 0, 1])
        self.assertEqual(np.unique(val).tolist(), [-1, 0, 1])
        self.assertEqual(np.unique(test).tolist(), [-1, 0, 1])
        self.assertEqual((np.sum(train >= 0) + np.sum(val >= 0) + np.sum(test >= 0)), 100)
        self.assertEqual(np.sum(train == 1), 4)
        self.assertEqual(np.sum(val == 1), 4)
        self.assertEqual(np.sum(test == 1), 12)

    def test_class_weights_reflect_imbalance(self):
        labels = np.array([0] * 170 + [1])
        weights = class_weights_for_labels(labels)

        self.assertEqual(weights.shape, (2,))
        self.assertAlmostEqual(weights.sum().item(), 2.0, places=6)
        self.assertAlmostEqual(weights[1].item() / weights[0].item(), 170.0, places=5)

    def test_balanced_sampler_draws_both_classes(self):
        labels = np.array([0] * 170 + [1])
        sampler = make_balanced_sampler(labels, seed=17)

        sampled = np.array([labels[i] for i in iter(sampler)])
        self.assertEqual(len(sampled), len(labels))
        self.assertEqual(np.unique(sampled).tolist(), [0, 1])
        self.assertGreater(np.sum(sampled == 1), 20)

    def test_hsi_dataset_returns_center_labeled_reflect_padded_patch(self):
        image = np.arange(5 * 5 * 2, dtype=np.float32).reshape(5, 5, 2) / 100.0
        ground_truth = np.full((5, 5), -1, dtype=np.int64)
        ground_truth[2, 2] = 1
        dataset = HSIDataset(image, ground_truth, patch_size=3, data_aug=False)

        patch, label = dataset[0]

        self.assertEqual(patch.shape, (1, 2, 3, 3))
        self.assertEqual(label.item(), 1)
        np.testing.assert_allclose(patch[0, :, 1, 1].numpy(), image[2, 2], atol=1e-7)


class TestDetectionMetrics(unittest.TestCase):
    def test_metrics_report_detection_statistics(self):
        target = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.4, 0.8, 0.35, 0.6, 0.75, 0.9])
        result = detection_metrics(scores, target, threshold=0.5)

        self.assertEqual(result["tp"], 3)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["tn"], 3)
        self.assertAlmostEqual(result["precision"], 3 / 4)
        self.assertAlmostEqual(result["pd"], 3 / 4)
        self.assertAlmostEqual(result["far"], 1 / 4)
        self.assertAlmostEqual(result["f1"], 2 * 0.75 * 0.75 / 1.5)
        self.assertGreater(result["roc_auc"], 0.5)
        self.assertGreater(result["pr_auc"], 0.5)

    def test_best_threshold_maximizes_f1(self):
        target = np.array([0, 0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
        threshold = select_best_threshold(scores, target)

        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)
        best = detection_metrics(scores, target, threshold=threshold)
        for candidate in np.linspace(0.05, 0.95, 19):
            result = detection_metrics(scores, target, threshold=candidate)
            self.assertGreaterEqual(best["f1"] + 1e-12, result["f1"])


class TestDetectionMat(unittest.TestCase):
    def test_load_detection_mat_preserves_binary_map_and_appends_pca(self):
        rng = np.random.default_rng(3)
        data = rng.normal(size=(4, 5, 3)).astype(np.float32)
        target_map = np.zeros((4, 5), dtype=np.uint8)
        target_map[1:3, 1:4] = 1

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mini.mat"
            scipy.io.savemat(str(path), {"data": data, "map": target_map})
            original, processed, gt, labels = load_detection_mat(str(path), pca_num=2)

        self.assertEqual(original.shape, (4, 5, 3))
        self.assertEqual(processed.shape, (4, 5, 5))
        self.assertEqual(gt.dtype, np.int64)
        np.testing.assert_array_equal(gt, target_map)
        self.assertEqual(labels, ["background", "target"])
        self.assertTrue(np.isfinite(processed).all())


class TestDetectorModel(unittest.TestCase):
    def test_full_model_has_verified_size_and_fixed_architecture(self):
        torch.manual_seed(0)
        model = TargetAwareHSIDetector(num_bands=89, pca_channels=32, patch_size=9)

        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        conv_macs = sum(
            module.weight.numel() * 9 * 9
            for module in model.modules()
            if isinstance(module, torch.nn.Conv2d)
        )
        linear_macs = sum(
            module.weight.numel()
            for module in model.modules()
            if isinstance(module, torch.nn.Linear)
        )

        self.assertEqual(parameter_count, 55_603)
        self.assertEqual(conv_macs + linear_macs, 4_230_960)
        with self.assertRaises(TypeError):
            TargetAwareHSIDetector(num_bands=89, pca_channels=32, gate_mode="ones")

    def test_model_outputs_two_logits(self):
        torch.manual_seed(0)
        model = TargetAwareHSIDetector(num_bands=89, pca_channels=32, patch_size=9)
        x = torch.randn(2, 1, 121, 9, 9)
        y = model(x)

        self.assertEqual(y.shape, (2, 2))
        probabilities = torch.softmax(y, dim=1)
        self.assertTrue(torch.all((probabilities >= 0) & (probabilities <= 1)).item())


class TestRuntimeInference(unittest.TestCase):
    def test_probability_map_scores_every_center_pixel(self):
        class CenterScoreModel:
            def __call__(self, patches):
                center = patches[:, 0, -1, patches.shape[-2] // 2, patches.shape[-1] // 2]
                background = torch.zeros_like(center)
                target_logits = torch.log(center / (1.0 - center))
                return torch.stack((background, target_logits), dim=1)

        image = np.arange(4 * 5 * 2, dtype=np.float32).reshape(4, 5, 2) / 100.0
        result = predict_probability_map(
            CenterScoreModel(),
            image,
            patch_size=3,
            device=torch.device("cpu"),
            batch_size=4,
        )

        self.assertEqual(result.shape, (4, 5))
        expected = image[:, :, -1]
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_train_and_evaluate_round_trip(self):
        torch.manual_seed(1)
        inputs = torch.randn(4, 1, 3, 3, 3)
        labels = torch.tensor([0, 0, 1, 1])
        dataset = torch.utils.data.TensorDataset(inputs, labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(27, 2),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        loss = train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"))
        scores, targets = evaluate_detector(model, loader, torch.device("cpu"))

        self.assertTrue(np.isfinite(loss))
        self.assertEqual(scores.shape, (4,))
        self.assertEqual(targets.tolist(), labels.tolist())
        self.assertTrue(np.all((scores >= 0.0) & (scores <= 1.0)))


if __name__ == "__main__":
    unittest.main()
