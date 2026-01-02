"""Training logic for RL agents."""

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
import numpy as np

from .config import Team, Config


class GameWrapper:
    """Wraps a PettingZoo game for single-agent training against a random opponent."""

    def __init__(self, env, agent_id: str):
        self.env = env
        self.agent_id = agent_id
        self.agents = list(env.possible_agents)
        self.opponent_id = [a for a in self.agents if a != agent_id][0]

        # Expose gym spaces
        self.observation_space = env.observation_space(agent_id)["observation"]
        self.action_space = env.action_space(agent_id)

        # Required for SB3 compatibility
        self.unwrapped = self
        self.render_mode = None
        self.metadata = {"render_modes": []}

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed)

        # If opponent goes first, play random move
        if self.env.agent_selection == self.opponent_id:
            self._play_opponent()

        obs, _, _, _, info = self.env.last()
        return obs["observation"], info

    def step(self, action):
        # Our agent's turn
        self.env.step(action)

        # Check if game over after our move
        obs, reward, term, trunc, info = self.env.last()
        if term or trunc or not self.env.agents:
            return obs["observation"], reward, True, trunc, info

        # Opponent's turn
        self._play_opponent()

        # Get final state after opponent
        obs, reward, term, trunc, info = self.env.last()
        done = term or trunc or not self.env.agents
        return obs["observation"], reward, done, trunc, info

    def _play_opponent(self):
        """Play a random valid move for the opponent."""
        obs, _, term, trunc, _ = self.env.last()
        if term or trunc or not self.env.agents:
            return
        mask = obs["action_mask"]
        valid_actions = np.where(mask == 1)[0]
        if len(valid_actions) > 0:
            action = np.random.choice(valid_actions)
            self.env.step(action)

    def action_masks(self) -> np.ndarray:
        """Return valid action mask for current state."""
        obs, _, _, _, _ = self.env.last()
        return obs["action_mask"]

    def close(self):
        self.env.close()


def _mask_fn(env: GameWrapper) -> np.ndarray:
    return env.action_masks()


def train_team(team: Team, config: Config) -> None:
    """Train a team's agent using Maskable PPO."""
    print(f"Training {team.name} ({team.id}) for {config.game.name}...")

    # Create environment using game from config
    raw_env = config.game.env_fn()
    raw_env.reset()

    # Get first agent ID
    agent_id = raw_env.possible_agents[0]
    env = GameWrapper(raw_env, agent_id)
    env = ActionMasker(env, _mask_fn)

    # Create or load model
    if team.model_path.exists():
        print(f"  Loading existing model from {team.model_path}")
        model = MaskablePPO.load(team.model_path, env=env)
    else:
        print(f"  Creating new model")
        model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)

    # Train
    print(f"  Training for {team.training_steps} steps...")
    model.learn(total_timesteps=team.training_steps)

    # Save
    model.save(team.model_path.with_suffix(""))  # SB3 adds .zip
    print(f"  Saved model to {team.model_path}")

    env.close()
