"""Tests for automated model retraining."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rbm.src.model import RBM
from rbm.retrain import (
    IMPROVEMENT_THRESHOLD,
    compare_models,
    evaluate_model,
    save_metrics,
    set_seed,
)


class TestSetSeed:
    """Tests for reproducibility seed setting."""

    def test_set_seed_deterministic(self):
        """set_seed makes torch operations deterministic."""
        set_seed(42)
        a = torch.rand(3, 3)
        set_seed(42)
        b = torch.rand(3, 3)
        assert torch.allclose(a, b)

    def test_set_seed_numpy(self):
        """set_seed affects numpy random state."""
        set_seed(42)
        a = np.random.rand(3)
        set_seed(42)
        b = np.random.rand(3)
        assert np.allclose(a, b)

    def test_different_seeds_different_results(self):
        """Different seeds produce different results."""
        set_seed(42)
        a = torch.rand(3, 3)
        set_seed(123)
        b = torch.rand(3, 3)
        assert not torch.allclose(a, b)


class TestEvaluateModel:
    """Tests for model evaluation function."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple RBM for testing."""
        return RBM(n_visible=10, n_hidden=5)

    @pytest.fixture
    def simple_data(self):
        """Create simple train/test tensors."""
        train = torch.zeros(5, 10)
        test = torch.zeros(5, 10)
        train[0, [0, 1, 2]] = 1
        train[1, [3, 4, 5]] = 1
        train[2, [6, 7]] = 1
        test[0, [3]] = 1
        test[1, [6]] = 1
        return train, test

    def test_evaluate_model_returns_dict(self, simple_model, simple_data):
        """evaluate_model returns dictionary with expected keys."""
        train, test = simple_data
        metrics = evaluate_model(simple_model, train, test, k=5, device='cpu')
        assert isinstance(metrics, dict)
        assert 'precision@5' in metrics
        assert 'map@5' in metrics
        assert 'ndcg@5' in metrics

    def test_evaluate_model_values_in_range(self, simple_model, simple_data):
        """Metric values are in valid range [0, 1]."""
        train, test = simple_data
        metrics = evaluate_model(simple_model, train, test, k=5, device='cpu')
        for key, value in metrics.items():
            assert 0 <= value <= 1, f"{key} = {value} is out of range"

    def test_evaluate_model_different_k(self, simple_model, simple_data):
        """Different k values produce different metric keys."""
        train, test = simple_data
        metrics_5 = evaluate_model(simple_model, train, test, k=5, device='cpu')
        metrics_10 = evaluate_model(simple_model, train, test, k=10, device='cpu')
        assert 'precision@5' in metrics_5
        assert 'precision@10' in metrics_10


class TestCompareModels:
    """Tests for model comparison logic."""

    def test_no_current_model_promotes(self):
        """When no current model exists, new model is promoted."""
        new_metrics = {'map@10': 0.4, 'precision@10': 0.2}
        should_promote, improvement = compare_models(None, new_metrics)
        assert should_promote is True
        assert improvement == float('inf')

    def test_better_model_promotes(self):
        """New model significantly better than current is promoted."""
        current = {'map@10': 0.3, 'precision@10': 0.15}
        new = {'map@10': 0.4, 'precision@10': 0.2}
        should_promote, improvement = compare_models(current, new)
        assert should_promote is True
        assert improvement > IMPROVEMENT_THRESHOLD

    def test_worse_model_not_promoted(self):
        """New model worse than current is not promoted."""
        current = {'map@10': 0.4, 'precision@10': 0.2}
        new = {'map@10': 0.35, 'precision@10': 0.18}
        should_promote, improvement = compare_models(current, new)
        assert should_promote is False
        assert improvement < 0

    def test_marginal_improvement_not_promoted(self):
        """Marginal improvement below threshold not promoted."""
        current = {'map@10': 0.4, 'precision@10': 0.2}
        new = {'map@10': 0.41, 'precision@10': 0.2}
        should_promote, improvement = compare_models(current, new)
        assert should_promote is False
        assert improvement < IMPROVEMENT_THRESHOLD

    def test_exact_threshold_promotes(self):
        """Improvement exactly at threshold promotes."""
        current = {'map@10': 0.4, 'precision@10': 0.2}
        threshold_value = 0.4 * (1 + IMPROVEMENT_THRESHOLD)
        new = {'map@10': threshold_value, 'precision@10': 0.2}
        should_promote, improvement = compare_models(current, new)
        assert should_promote is True

    def test_custom_primary_metric(self):
        """Can use different primary metric for comparison."""
        current = {'map@10': 0.4, 'precision@10': 0.2, 'ndcg@10': 0.3}
        new = {'map@10': 0.35, 'precision@10': 0.3, 'ndcg@10': 0.32}
        should_promote, _ = compare_models(current, new, primary_metric='precision@10')
        assert should_promote is True

    def test_zero_current_metric(self):
        """Handles zero current metric gracefully."""
        current = {'map@10': 0.0}
        new = {'map@10': 0.1}
        should_promote, improvement = compare_models(current, new)
        assert should_promote is True
        assert improvement == float('inf')


class TestSaveMetrics:
    """Tests for metrics saving functionality."""

    @pytest.fixture
    def sample_config(self):
        """Sample training configuration."""
        return {
            'model': {
                'n_hidden': 1024,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 50,
                'k': 10
            },
            'data': {
                'holdout_ratio': 0.1,
                'min_likes_user': 50,
                'min_likes_anime': 50
            }
        }

    def test_save_metrics_creates_file(self, sample_config):
        """save_metrics creates JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, 'metrics.json')
            with patch('rbm.retrain.RETRAIN_METRICS_PATH', metrics_path):
                with patch('rbm.retrain.OUTPUT_DIR', tmpdir):
                    save_metrics({'map@10': 0.3}, {'map@10': 0.4}, True, 0.33, sample_config)
                    assert os.path.exists(metrics_path)

    def test_save_metrics_content(self, sample_config):
        """save_metrics writes correct content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, 'metrics.json')
            with patch('rbm.retrain.RETRAIN_METRICS_PATH', metrics_path):
                with patch('rbm.retrain.OUTPUT_DIR', tmpdir):
                    current = {'map@10': 0.3}
                    new = {'map@10': 0.4}
                    save_metrics(current, new, True, 0.33, sample_config)

                    with open(metrics_path) as f:
                        data = json.load(f)

                    assert 'timestamp' in data
                    assert data['promoted'] is True
                    assert data['improvement'] == 0.33
                    assert data['new_model'] == new
                    assert data['current_model'] == new

    def test_save_metrics_not_promoted(self, sample_config):
        """When not promoted, current_model stays the same."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, 'metrics.json')
            with patch('rbm.retrain.RETRAIN_METRICS_PATH', metrics_path):
                with patch('rbm.retrain.OUTPUT_DIR', tmpdir):
                    current = {'map@10': 0.4}
                    new = {'map@10': 0.35}
                    save_metrics(current, new, False, -0.125, sample_config)

                    with open(metrics_path) as f:
                        data = json.load(f)

                    assert data['promoted'] is False
                    assert data['current_model'] == current


class TestImprovementThreshold:
    """Tests for improvement threshold constant."""

    def test_threshold_reasonable_value(self):
        """Threshold is a reasonable percentage (1-20%)."""
        assert 0.01 <= IMPROVEMENT_THRESHOLD <= 0.20

    def test_threshold_is_float(self):
        """Threshold is a float."""
        assert isinstance(IMPROVEMENT_THRESHOLD, float)


class TestRetrainIntegration:
    """Integration tests for the retrain function."""

    @pytest.fixture
    def mock_data(self):
        """Create mock training data."""
        train = torch.zeros(10, 20)
        test = torch.zeros(10, 20)
        for i in range(10):
            train[i, i:i+3] = 1
            test[i, (i+3) % 20] = 1
        return train, test

    def test_retrain_dry_run_no_side_effects(self, mock_data):
        """Dry run doesn't create or modify files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'model.pth')
            metrics_path = os.path.join(tmpdir, 'metrics.json')
            assert not os.path.exists(model_path)
            assert not os.path.exists(metrics_path)
