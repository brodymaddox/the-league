"""Tests for training functionality."""

import pytest

from league.train import PettingZooWrapper


class TestPettingZooWrapper:
    def test_wrapper_exposes_spaces(self):
        """Test that wrapper correctly exposes observation and action spaces."""
        from pettingzoo.atari import pong_v3

        env = pong_v3.env()
        env.reset()

        wrapper = PettingZooWrapper(env, "first_0")

        assert wrapper.observation_space is not None
        assert wrapper.action_space is not None
        assert wrapper.agent_id == "first_0"

        wrapper.close()

    def test_wrapper_reset_returns_obs(self):
        """Test that reset returns observation and info."""
        from pettingzoo.atari import pong_v3

        env = pong_v3.env()
        env.reset()

        wrapper = PettingZooWrapper(env, "first_0")
        obs, info = wrapper.reset()

        assert obs is not None
        assert isinstance(info, dict)

        wrapper.close()

    def test_wrapper_step_returns_tuple(self):
        """Test that step returns correct tuple format."""
        from pettingzoo.atari import pong_v3

        env = pong_v3.env()
        env.reset()

        wrapper = PettingZooWrapper(env, "first_0")
        wrapper.reset()

        action = wrapper.action_space.sample()
        result = wrapper.step(action)

        assert len(result) == 5  # obs, reward, done, trunc, info

        wrapper.close()
