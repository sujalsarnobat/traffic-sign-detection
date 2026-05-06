"""
Unit tests for Traffic Sign Detection project.
Run: pytest tests/ -v
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from class_names import CLASS_NAMES, NUM_CLASSES
from data_loader import preprocess, preprocess_single_image
from model import build_model, compile_model


# ── Class names ────────────────────────────────────────────────────────────────

def test_num_classes():
    assert NUM_CLASSES == 43

def test_class_names_keys():
    assert set(CLASS_NAMES.keys()) == set(range(43))

def test_class_names_values_unique():
    assert len(set(CLASS_NAMES.values())) == 43


# ── Preprocessing ──────────────────────────────────────────────────────────────

def test_preprocess_normalises():
    imgs = np.array([[[255, 128, 0]]], dtype=np.uint8).reshape(1, 1, 1, 3)
    result = preprocess(imgs)
    assert result.max() <= 1.0
    assert result.min() >= 0.0
    assert result.dtype == np.float32

def test_preprocess_single_image_shape():
    dummy_bgr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = preprocess_single_image(dummy_bgr)
    assert result.shape == (1, 32, 32, 3)
    assert result.dtype == np.float32
    assert result.max() <= 1.0

def test_preprocess_single_image_different_sizes():
    for size in [(50, 50), (128, 128), (640, 480)]:
        dummy = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        result = preprocess_single_image(dummy)
        assert result.shape == (1, 32, 32, 3)


# ── Model ──────────────────────────────────────────────────────────────────────

def test_model_builds():
    model = build_model(num_classes=43)
    assert model is not None
    assert model.name == "TrafficSignCNN"

def test_model_output_shape():
    model = build_model(num_classes=43)
    model = compile_model(model)
    dummy_input = np.random.rand(4, 32, 32, 3).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)
    assert output.shape == (4, 43)

def test_model_output_sums_to_one():
    model = build_model(num_classes=43)
    dummy_input = np.random.rand(2, 32, 32, 3).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)
    np.testing.assert_allclose(output.sum(axis=1), [1.0, 1.0], atol=1e-5)

def test_model_custom_classes():
    model = build_model(num_classes=10)
    dummy_input = np.random.rand(1, 32, 32, 3).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)
    assert output.shape == (1, 10)
