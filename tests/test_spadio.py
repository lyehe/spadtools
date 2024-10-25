"""Test the SPADData class."""

from pathlib import Path

import numpy as np
import pytest

from spadtools import SPADData, unbin


@pytest.fixture
def dummy_data():
    """Generate a dummy SPAD data array."""
    return np.random.binomial(1, 0.05, (100, 512, 512))


@pytest.fixture
def temp_bin_file(tmp_path: Path, dummy_data):
    """Create a temporary SPAD bin file."""
    file_path = tmp_path / "test.bin"
    unbin(file_path, dummy_data)
    return file_path


@pytest.fixture
def spad_data(temp_bin_file):
    """Create a SPADData object from a temporary SPAD bin file."""
    return SPADData(temp_bin_file)


def test_spad_data_load(spad_data, dummy_data):
    """Test loading SPAD data."""
    loaded_data = spad_data.load()
    assert loaded_data.data.shape == dummy_data.shape
    assert np.array_equal(loaded_data.data, dummy_data)


def test_spad_data_unload(spad_data):
    """Test unloading SPAD data."""
    spad_data.load()
    assert spad_data._loaded()
    spad_data.unload()
    assert not spad_data._loaded()


def test_spad_data_binz(spad_data):
    """Test binning SPAD data."""
    bin_size = 10
    binned = spad_data.binz(bin_size)
    assert binned.shape == (10, 512, 512)


def test_spad_data_addition(spad_data, dummy_data):
    """Test adding two SPADData objects."""
    other_data = SPADData(None)
    other_data.data = dummy_data
    combined = spad_data + other_data
    assert combined.data.shape == (200, 512, 512)


def test_spad_data_preview(spad_data):
    """Test previewing SPAD data."""
    preview = spad_data.preview(plot=False)
    assert preview.shape == (512, 512)
