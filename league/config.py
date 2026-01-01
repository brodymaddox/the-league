"""Configuration loading for teams and environments."""

from pathlib import Path
from dataclasses import dataclass

import yaml


@dataclass
class Team:
    id: str
    name: str
    color: str
    training_steps: int
    model_path: Path

    @property
    def trained(self) -> bool:
        return self.model_path.exists()


@dataclass
class Config:
    teams: dict[str, Team]
    env_name: str
    render_mode: str


def load_config(config_path: str = "teams.yaml") -> Config:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        data = yaml.safe_load(f)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    teams = {}
    for team_id, team_data in data["teams"].items():
        teams[team_id] = Team(
            id=team_id,
            name=team_data["name"],
            color=team_data["color"],
            training_steps=team_data["training_steps"],
            model_path=models_dir / f"{team_id}.zip",
        )

    return Config(
        teams=teams,
        env_name=data["environment"]["name"],
        render_mode=data["environment"]["render_mode"],
    )


def get_team(config: Config, team_id: str) -> Team:
    """Get a team by ID."""
    if team_id not in config.teams:
        available = ", ".join(config.teams.keys())
        raise ValueError(f"Unknown team: {team_id}. Available: {available}")
    return config.teams[team_id]
