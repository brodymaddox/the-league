"""Tests for training functionality."""

import pytest
import numpy as np

from league.train import ConnectFourWrapper


class TestConnectFourWrapper:
    def test_wrapper_exposes_spaces(self):
        """Test that wrapper correctly exposes observation and action spaces."""
        from pettingzoo.classic import connect_four_v3

        env = connect_four_v3.env()
        env.reset()

        wrapper = ConnectFourWrapper(env, "player_0")

        assert wrapper.observation_space is not None
        assert wrapper.action_space is not None
        assert wrapper.agent_id == "player_0"

        wrapper.close()

    def test_wrapper_reset_returns_obs(self):
        """Test that reset returns observation and info."""
        from pettingzoo.classic import connect_four_v3

        env = connect_four_v3.env()
        env.reset()

        wrapper = ConnectFourWrapper(env, "player_0")
        obs, info = wrapper.reset()

        assert obs is not None
        assert isinstance(info, dict)

        wrapper.close()

    def test_wrapper_step_returns_tuple(self):
        """Test that step returns correct tuple format."""
        from pettingzoo.classic import connect_four_v3

        env = connect_four_v3.env()
        env.reset()

        wrapper = ConnectFourWrapper(env, "player_0")
        wrapper.reset()

        # Get a valid action from action mask
        mask = wrapper.action_masks()
        valid_actions = np.where(mask == 1)[0]
        action = valid_actions[0]

        result = wrapper.step(action)

        assert len(result) == 5  # obs, reward, done, trunc, info

        wrapper.close()

    def test_action_masks(self):
        """Test that action_masks returns valid mask."""
        from pettingzoo.classic import connect_four_v3

        env = connect_four_v3.env()
        env.reset()

        wrapper = ConnectFourWrapper(env, "player_0")
        wrapper.reset()

        mask = wrapper.action_masks()

        assert isinstance(mask, np.ndarray)
        assert len(mask) == 7  # Connect Four has 7 columns
        assert all(m in [0, 1] for m in mask)

        wrapper.close()
