"""Test the SPADHotpixelTool class."""

import numpy as np
import pytest

from spadtools import SPADHotpixelTool


@pytest.fixture
def test_data():
    """Generate a dummy SPAD data array with hot pixels."""
    data = np.random.binomial(1, 0.05, (100, 512, 512))
    # Add some hot pixels
    for _ in range(100):
        x, y = np.random.randint(0, 512, 2)
        data[:, x, y] = np.random.binomial(1, 0.8, 100)
    return data


def test_hotpixel_detection(test_data):
    """Test hotpixel detection."""
    tool = SPADHotpixelTool(test_data)
    hotpixels = tool.locate_hotpixels(hp_threshold=2)
    assert len(hotpixels) > 0
    assert hotpixels.shape[1] == 2  # x, y coordinates


def test_background_estimation(test_data):
    """Test background estimation."""
    tool = SPADHotpixelTool(test_data)
    background = tool.get_background(method="open", kernel_size=3)
    assert background.shape == (512, 512)


def test_hotpixel_correction(test_data):
    """Test hotpixel correction."""
    tool = SPADHotpixelTool(test_data)
    corrected = tool.correct_hotpixels(hp_threshold=2)
    assert corrected.shape == test_data.shape

    # Corrected data should have fewer extreme values
    assert np.sum(corrected) < np.sum(test_data)


def test_reset_functionality(test_data):
    """Test reset functionality."""
    tool = SPADHotpixelTool(test_data)
    tool.locate_hotpixels()
    assert tool.hotpixel_locations is not None

    tool.reset()
    assert tool.hotpixel_locations is None
