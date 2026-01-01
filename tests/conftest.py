"""Pytest fixtures for the league tests."""

import tempfile
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config_path(temp_dir):
    """Create a sample config file for testing."""
    config_data = {
        "teams": {
            "test1": {
                "name": "Test Team One",
                "color": "#FF0000",
                "training_steps": 100,
            },
            "test2": {
                "name": "Test Team Two",
                "color": "#0000FF",
                "training_steps": 100,
            },
        },
        "environment": {
            "name": "pong_v3",
            "render_mode": "rgb_array",
        },
    }

    config_path = temp_dir / "teams.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    return config_path
