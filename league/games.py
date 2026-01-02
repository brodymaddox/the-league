"""Game registry - supported games and their configurations."""

from dataclasses import dataclass
from typing import Callable

from pettingzoo.classic import connect_four_v3, chess_v6, go_v5, rps_v2


@dataclass
class GameInfo:
    name: str  # Display name
    env_fn: Callable  # Function to create the environment
    agent_prefix: str  # Prefix for agent IDs (e.g., "player_" -> "player_0", "player_1")


GAMES = {
    "connect_four": GameInfo(
        name="Connect Four",
        env_fn=connect_four_v3.env,
        agent_prefix="player_",
    ),
    "chess": GameInfo(
        name="Chess",
        env_fn=chess_v6.env,
        agent_prefix="player_",
    ),
    "go": GameInfo(
        name="Go",
        env_fn=lambda **kwargs: go_v5.env(board_size=9, **kwargs),  # 9x9 board for faster games
        agent_prefix="black_",  # Go uses black_0, white_0
    ),
    "rps": GameInfo(
        name="Rock Paper Scissors",
        env_fn=rps_v2.env,
        agent_prefix="player_",
    ),
}


def get_game(game_id: str) -> GameInfo:
    """Get game info by ID."""
    if game_id not in GAMES:
        available = ", ".join(GAMES.keys())
        raise ValueError(f"Unknown game: {game_id}. Available: {available}")
    return GAMES[game_id]


def list_games() -> list[str]:
    """List all available game IDs."""
    return list(GAMES.keys())
