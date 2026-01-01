"""Tests for configuration loading."""

import os

import pytest

from league.config import load_config, get_team, Team, Config


class TestLoadConfig:
    def test_loads_teams(self, sample_config_path, temp_dir, monkeypatch):
        monkeypatch.chdir(temp_dir)
        config = load_config(str(sample_config_path))

        assert len(config.teams) == 2
        assert "test1" in config.teams
        assert "test2" in config.teams

    def test_team_attributes(self, sample_config_path, temp_dir, monkeypatch):
        monkeypatch.chdir(temp_dir)
        config = load_config(str(sample_config_path))

        team = config.teams["test1"]
        assert team.id == "test1"
        assert team.name == "Test Team One"
        assert team.color == "#FF0000"
        assert team.training_steps == 100

    def test_environment_config(self, sample_config_path, temp_dir, monkeypatch):
        monkeypatch.chdir(temp_dir)
        config = load_config(str(sample_config_path))

        assert config.env_name == "pong_v3"
        assert config.render_mode == "rgb_array"

    def test_creates_models_directory(self, sample_config_path, temp_dir, monkeypatch):
        monkeypatch.chdir(temp_dir)
        load_config(str(sample_config_path))

        assert (temp_dir / "models").exists()


class TestGetTeam:
    def test_returns_team(self, sample_config_path, temp_dir, monkeypatch):
        monkeypatch.chdir(temp_dir)
        config = load_config(str(sample_config_path))

        team = get_team(config, "test1")
        assert team.id == "test1"
        assert team.name == "Test Team One"

    def test_raises_for_unknown_team(self, sample_config_path, temp_dir, monkeypatch):
        monkeypatch.chdir(temp_dir)
        config = load_config(str(sample_config_path))

        with pytest.raises(ValueError, match="Unknown team: unknown"):
            get_team(config, "unknown")


class TestTeam:
    def test_trained_property_false_when_no_model(self, temp_dir):
        team = Team(
            id="test",
            name="Test",
            color="#000",
            training_steps=100,
            model_path=temp_dir / "nonexistent.zip",
        )
        assert team.trained is False

    def test_trained_property_true_when_model_exists(self, temp_dir):
        model_path = temp_dir / "test.zip"
        model_path.touch()

        team = Team(
            id="test",
            name="Test",
            color="#000",
            training_steps=100,
            model_path=model_path,
        )
        assert team.trained is True
