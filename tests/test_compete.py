"""Tests for competition and video functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np


class TestVideoRendering:
    def test_creates_videos_directory(self, temp_dir, monkeypatch):
        monkeypatch.chdir(temp_dir)

        from league.video import save_match_video

        team1 = MagicMock()
        team1.id = "team1"
        team1.name = "Test Team 1"
        team1.color = "#FF0000"

        team2 = MagicMock()
        team2.id = "team2"
        team2.name = "Test Team 2"
        team2.color = "#0000FF"

        # Create fake frames
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
        scores = [(0, 0), (1, 0), (1, 1)]

        with patch("league.video.imageio.mimsave"):
            save_match_video(frames, team1, team2, scores, winner=team1)

        assert (temp_dir / "videos").exists()

    def test_video_filename_format(self, temp_dir, monkeypatch):
        monkeypatch.chdir(temp_dir)

        from league.video import save_match_video

        team1 = MagicMock()
        team1.id = "la"
        team1.name = "Los Angeles"
        team1.color = "#FFD700"

        team2 = MagicMock()
        team2.id = "nyc"
        team2.name = "New York"
        team2.color = "#1E90FF"

        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
        scores = [(0, 0), (0, 0), (0, 0)]

        with patch("league.video.imageio.mimsave"):
            video_path = save_match_video(frames, team1, team2, scores, winner=None)

        assert "la_vs_nyc" in str(video_path)
        assert video_path.suffix == ".gif"

    def test_hex_to_rgb(self):
        from league.video import hex_to_rgb

        assert hex_to_rgb("#FF0000") == (255, 0, 0)
        assert hex_to_rgb("#00FF00") == (0, 255, 0)
        assert hex_to_rgb("#0000FF") == (0, 0, 255)
        assert hex_to_rgb("FFFFFF") == (255, 255, 255)
