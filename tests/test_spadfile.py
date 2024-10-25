"""Test the SPADFile class."""

from pathlib import Path

import numpy as np
import pytest

from spadtools import SPADFile, unbin


@pytest.fixture
def temp_dataset(tmp_path: Path):
    """Create a temporary SPAD dataset."""
    data_dir = tmp_path / "test_dataset"
    data_dir.mkdir()

    files = []
    for i in range(5):
        data = np.random.binomial(1, 0.05, (100, 512, 512))
        file_path = data_dir / f"test_{i:04d}.bin"
        unbin(file_path, data)
        files.append(file_path)

    return data_dir


def test_spadfile_load(temp_dataset):
    """Test loading SPAD files."""
    spad_file = SPADFile(temp_dataset)
    assert len(spad_file) == 5

    loaded = spad_file[0]
    assert loaded.data.shape == (100, 512, 512)


def test_spadfile_combine(temp_dataset):
    """Test combining SPAD files."""
    spad_file = SPADFile(temp_dataset)
    combined = spad_file.combine()
    assert combined.shape == (500, 512, 512)

    stacked = spad_file.combine(concat=False)
    assert stacked.shape == (5, 100, 512, 512)


def test_spadfile_slice(temp_dataset):
    """Test slicing SPAD files."""
    spad_file = SPADFile(temp_dataset)
    subset = spad_file[1:3]
    assert len(subset) == 2


def test_spadfile_save(temp_dataset, tmp_path):
    """Test saving SPAD files."""
    spad_file = SPADFile(temp_dataset)
    save_path = tmp_path / "saved"
    spad_file.save(save_path, file_type="zarr")
    assert (save_path / f"{spad_file.name}.zarr").exists()
