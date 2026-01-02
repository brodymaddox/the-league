"""Competition and video recording logic."""

import numpy as np
from sb3_contrib import MaskablePPO

from .config import Team, Config
from .video import save_match_video


def run_match(
    team1: Team,
    team2: Team,
    config: Config,
    record: bool = True,
    max_turns: int = 500,
) -> dict:
    """Run a match between two teams and optionally record video."""
    print(f"Match: {team1.name} vs {team2.name} ({config.game.name})")

    # Load models
    model1 = MaskablePPO.load(team1.model_path)
    model2 = MaskablePPO.load(team2.model_path)
    print(f"  Loaded both models")

    # Create environment with rendering
    env = config.game.env_fn(render_mode="rgb_array")
    env.reset()

    agents = list(env.possible_agents)
    agent_to_team = {agents[0]: (team1, model1), agents[1]: (team2, model2)}

    frames = []
    scores_over_time = []
    rewards = {team1.id: 0, team2.id: 0}
    turn_count = 0

    # Run the match
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()

        team, model = agent_to_team[agent]
        rewards[team.id] += reward

        # Track scores over time
        scores_over_time.append((rewards[team1.id], rewards[team2.id]))

        if term or trunc:
            action = None
        else:
            # Get action mask and predict
            mask = obs["action_mask"]
            action, _ = model.predict(obs["observation"], action_masks=mask, deterministic=True)

        env.step(action)

        # Record frame (sample every few turns to keep video manageable)
        if record and turn_count % 2 == 0:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        turn_count += 1
        if turn_count >= max_turns:
            break

    env.close()

    # Determine winner
    if rewards[team1.id] > rewards[team2.id]:
        winner = team1
    elif rewards[team2.id] > rewards[team1.id]:
        winner = team2
    else:
        winner = None

    result = {
        "team1": team1.id,
        "team2": team2.id,
        "game": config.game_id,
        "rewards": rewards,
        "winner": winner.id if winner else "draw",
        "turns": turn_count,
    }

    print(f"  Result: {rewards}")
    print(f"  Winner: {result['winner']}")

    # Save video with overlays
    if record and frames:
        # Sample scores to match frames
        sampled_scores = scores_over_time[::2][: len(frames)]
        video_path = save_match_video(
            frames, team1, team2, sampled_scores, winner, game_name=config.game.name
        )
        result["video"] = str(video_path)
        print(f"  Video saved: {video_path}")

    return result
