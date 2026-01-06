"""Configuration loading for teams and environments."""

from pathlib import Path
from dataclasses import dataclass

import yaml

from .games import get_game, GameInfo


@dataclass
class Team:
    id: str
    name: str
    color: str
    training_steps: int
    model_path: Path
    skill_level: float = 1.0  # 0.0-1.0, probability of optimal action

    @property
    def trained(self) -> bool:
        return self.model_path.exists()


@dataclass
class Config:
    teams: dict[str, Team]
    game_id: str
    game: GameInfo


def load_config(config_path: str = "teams.yaml") -> Config:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        data = yaml.safe_load(f)

    game_id = data.get("game", "connect_four")
    game = get_game(game_id)

    # Models go in game-specific directory
    models_dir = Path("models") / game_id
    models_dir.mkdir(parents=True, exist_ok=True)

    teams = {}
    for team_id, team_data in data["teams"].items():
        teams[team_id] = Team(
            id=team_id,
            name=team_data["name"],
            color=team_data["color"],
            training_steps=team_data["training_steps"],
            model_path=models_dir / f"{team_id}.zip",
            skill_level=team_data.get("skill_level", 1.0),
        )

    return Config(
        teams=teams,
        game_id=game_id,
        game=game,
    )


def get_team(config: Config, team_id: str) -> Team:
    """Get a team by ID."""
    if team_id not in config.teams:
        available = ", ".join(config.teams.keys())
        raise ValueError(f"Unknown team: {team_id}. Available: {available}")
    return config.teams[team_id]
