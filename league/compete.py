"""Competition and video recording logic."""

from pathlib import Path
from datetime import datetime

import imageio
from pettingzoo.atari import pong_v3
from stable_baselines3 import PPO

from .config import Team, Config


def run_match(
    team1: Team,
    team2: Team,
    config: Config,
    record: bool = True,
    max_frames: int = 3000,
) -> dict:
    """Run a match between two teams and optionally record video."""
    print(f"Match: {team1.name} vs {team2.name}")

    # Load models
    model1 = PPO.load(team1.model_path)
    model2 = PPO.load(team2.model_path)
    print(f"  Loaded both models")

    # Create environment
    env = pong_v3.env(render_mode="rgb_array")
    env.reset()

    agents = list(env.possible_agents)
    agent_to_team = {agents[0]: (team1, model1), agents[1]: (team2, model2)}

    frames = []
    rewards = {team1.id: 0, team2.id: 0}
    frame_count = 0

    # Run the match
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()

        team, model = agent_to_team[agent]
        rewards[team.id] += reward

        if term or trunc:
            action = None
        else:
            action, _ = model.predict(obs, deterministic=True)

        env.step(action)

        # Record frame
        if record:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        frame_count += 1
        if frame_count >= max_frames:
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
        "rewards": rewards,
        "winner": winner.id if winner else "draw",
        "frames": frame_count,
    }

    print(f"  Result: {rewards}")
    print(f"  Winner: {result['winner']}")

    # Save video
    if record and frames:
        video_path = save_video(frames, team1, team2)
        result["video"] = str(video_path)
        print(f"  Video saved: {video_path}")

    return result


def save_video(frames: list, team1: Team, team2: Team) -> Path:
    """Save frames as a video file."""
    videos_dir = Path("videos")
    videos_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{team1.id}_vs_{team2.id}_{timestamp}.mp4"
    video_path = videos_dir / filename

    imageio.mimsave(video_path, frames, fps=30)
    return video_path
