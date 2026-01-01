"""Training logic for RL agents."""

from pettingzoo.atari import pong_v3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .config import Team, Config


def _make_env(config: Config, agent_id: str):
    """Create a single-agent wrapper for training."""

    def _init():
        env = pong_v3.env()
        env.reset()
        # Wrap PettingZoo env for single agent training
        return PettingZooWrapper(env, agent_id)

    return _init


class PettingZooWrapper:
    """Wraps a PettingZoo env to train a single agent with random opponent."""

    def __init__(self, env, agent_id: str):
        self.env = env
        self.agent_id = agent_id
        self.agents = list(env.possible_agents)
        self.opponent_id = [a for a in self.agents if a != agent_id][0]

        # Expose gym spaces
        self.observation_space = env.observation_space(agent_id)
        self.action_space = env.action_space(agent_id)

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        obs, _, _, _, info = self.env.last()
        return obs, info

    def step(self, action):
        # Our agent's turn
        self.env.step(action)

        # Check if game over
        if not self.env.agents:
            obs, reward, term, trunc, info = self.env.last()
            return obs, reward, True, trunc, info

        # Opponent's turn (random action)
        if self.env.agent_selection == self.opponent_id:
            opponent_action = self.env.action_space(self.opponent_id).sample()
            self.env.step(opponent_action)

        obs, reward, term, trunc, info = self.env.last()
        done = term or trunc or not self.env.agents
        return obs, reward, done, trunc, info

    def close(self):
        self.env.close()


def train_team(team: Team, config: Config) -> None:
    """Train a team's agent."""
    print(f"Training {team.name} ({team.id})...")

    # Create vectorized environment
    env = DummyVecEnv([_make_env(config, "first_0")])

    # Create or load model
    if team.model_path.exists():
        print(f"  Loading existing model from {team.model_path}")
        model = PPO.load(team.model_path, env=env)
    else:
        print(f"  Creating new model")
        model = PPO("CnnPolicy", env, verbose=1)

    # Train
    print(f"  Training for {team.training_steps} steps...")
    model.learn(total_timesteps=team.training_steps)

    # Save
    model.save(team.model_path.with_suffix(""))  # SB3 adds .zip
    print(f"  Saved model to {team.model_path}")

    env.close()
