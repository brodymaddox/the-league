"""Tests for competition functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from league.compete import save_video


class TestSaveVideo:
    def test_creates_videos_directory(self, temp_dir, monkeypatch):
        monkeypatch.chdir(temp_dir)

        team1 = MagicMock()
        team1.id = "team1"
        team2 = MagicMock()
        team2.id = "team2"

        # Create fake frames (simple 2x2 RGB images)
        import numpy as np

        frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(5)]

        with patch("league.compete.imageio.mimsave"):
            save_video(frames, team1, team2)

        assert (temp_dir / "videos").exists()

    def test_video_filename_format(self, temp_dir, monkeypatch):
        monkeypatch.chdir(temp_dir)

        team1 = MagicMock()
        team1.id = "la"
        team2 = MagicMock()
        team2.id = "nyc"

        import numpy as np

        frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(5)]

        with patch("league.compete.imageio.mimsave"):
            video_path = save_video(frames, team1, team2)

        assert "la_vs_nyc" in str(video_path)
        assert video_path.suffix == ".mp4"
