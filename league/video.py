"""Video rendering for engaging phone-format content."""

from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio

from .config import Team


# Phone format: 9:16 aspect ratio
VIDEO_WIDTH = 720
VIDEO_HEIGHT = 1280


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def create_frame(
    game_frame: np.ndarray,
    team1: Team,
    team2: Team,
    score1: float,
    score2: float,
    turn: int,
) -> np.ndarray:
    """Create an engaging phone-format frame with overlays."""
    # Create base canvas
    canvas = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), color=(20, 20, 30))
    draw = ImageDraw.Draw(canvas)

    # Try to load a font, fall back to default
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 42)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except OSError:
        font_large = ImageFont.load_default()
        font_medium = font_large
        font_small = font_large

    # Header - "THE LEAGUE"
    draw.text((VIDEO_WIDTH // 2, 40), "THE LEAGUE", fill=(255, 215, 0), font=font_large, anchor="mm")

    # Team 1 section (left side, top)
    team1_color = hex_to_rgb(team1.color)
    draw.rectangle([(20, 100), (VIDEO_WIDTH // 2 - 10, 200)], fill=team1_color, outline="white", width=2)
    draw.text((VIDEO_WIDTH // 4, 130), team1.name, fill="white", font=font_medium, anchor="mm")
    draw.text((VIDEO_WIDTH // 4, 170), f"Score: {score1:.0f}", fill="white", font=font_small, anchor="mm")

    # Team 2 section (right side, top)
    team2_color = hex_to_rgb(team2.color)
    draw.rectangle([(VIDEO_WIDTH // 2 + 10, 100), (VIDEO_WIDTH - 20, 200)], fill=team2_color, outline="white", width=2)
    draw.text((VIDEO_WIDTH * 3 // 4, 130), team2.name, fill="white", font=font_medium, anchor="mm")
    draw.text((VIDEO_WIDTH * 3 // 4, 170), f"Score: {score2:.0f}", fill="white", font=font_small, anchor="mm")

    # VS badge
    draw.ellipse([(VIDEO_WIDTH // 2 - 30, 120), (VIDEO_WIDTH // 2 + 30, 180)], fill=(40, 40, 50), outline="gold", width=2)
    draw.text((VIDEO_WIDTH // 2, 150), "VS", fill="gold", font=font_medium, anchor="mm")

    # Game frame - scale and center it
    if game_frame is not None:
        game_img = Image.fromarray(game_frame)

        # Calculate scaling to fit in the game area
        game_area_width = VIDEO_WIDTH - 40
        game_area_height = VIDEO_HEIGHT - 400  # Leave room for header and footer

        # Scale game frame to fit
        scale = min(game_area_width / game_img.width, game_area_height / game_img.height)
        new_width = int(game_img.width * scale)
        new_height = int(game_img.height * scale)
        game_img = game_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Center the game frame
        x_offset = (VIDEO_WIDTH - new_width) // 2
        y_offset = 240 + (game_area_height - new_height) // 2

        # Add border around game
        draw.rectangle(
            [(x_offset - 5, y_offset - 5), (x_offset + new_width + 5, y_offset + new_height + 5)],
            outline="gold",
            width=3,
        )
        canvas.paste(game_img, (x_offset, y_offset))

    # Footer - turn counter
    draw.rectangle([(20, VIDEO_HEIGHT - 80), (VIDEO_WIDTH - 20, VIDEO_HEIGHT - 20)], fill=(40, 40, 50), outline="gold", width=2)
    draw.text((VIDEO_WIDTH // 2, VIDEO_HEIGHT - 50), f"Turn {turn}", fill="white", font=font_medium, anchor="mm")

    return np.array(canvas)


def create_title_frame(team1: Team, team2: Team) -> np.ndarray:
    """Create a title/intro frame."""
    canvas = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), color=(20, 20, 30))
    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 56)
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except OSError:
        font_title = ImageFont.load_default()
        font_large = font_title
        font_medium = font_title

    # Title
    draw.text((VIDEO_WIDTH // 2, 200), "THE LEAGUE", fill=(255, 215, 0), font=font_title, anchor="mm")
    draw.text((VIDEO_WIDTH // 2, 260), "PRESENTS", fill=(180, 180, 180), font=font_medium, anchor="mm")

    # Team 1
    team1_color = hex_to_rgb(team1.color)
    draw.rectangle([(60, 400), (VIDEO_WIDTH - 60, 520)], fill=team1_color, outline="white", width=3)
    draw.text((VIDEO_WIDTH // 2, 460), team1.name, fill="white", font=font_large, anchor="mm")

    # VS
    draw.text((VIDEO_WIDTH // 2, 600), "VS", fill=(255, 215, 0), font=font_title, anchor="mm")

    # Team 2
    team2_color = hex_to_rgb(team2.color)
    draw.rectangle([(60, 680), (VIDEO_WIDTH - 60, 800)], fill=team2_color, outline="white", width=3)
    draw.text((VIDEO_WIDTH // 2, 740), team2.name, fill="white", font=font_large, anchor="mm")

    # Game type
    draw.text((VIDEO_WIDTH // 2, 950), "CONNECT FOUR", fill=(150, 150, 150), font=font_medium, anchor="mm")

    return np.array(canvas)


def create_winner_frame(team1: Team, team2: Team, winner: Team | None, score1: float, score2: float) -> np.ndarray:
    """Create a winner announcement frame."""
    canvas = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), color=(20, 20, 30))
    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except OSError:
        font_title = ImageFont.load_default()
        font_large = font_title
        font_medium = font_title

    # Result header
    if winner:
        draw.text((VIDEO_WIDTH // 2, 200), "WINNER!", fill=(255, 215, 0), font=font_title, anchor="mm")

        # Winner team box
        winner_color = hex_to_rgb(winner.color)
        draw.rectangle([(60, 350), (VIDEO_WIDTH - 60, 500)], fill=winner_color, outline="gold", width=4)
        draw.text((VIDEO_WIDTH // 2, 400), winner.name, fill="white", font=font_large, anchor="mm")
        draw.text((VIDEO_WIDTH // 2, 450), "CHAMPION", fill=(255, 215, 0), font=font_medium, anchor="mm")
    else:
        draw.text((VIDEO_WIDTH // 2, 200), "DRAW!", fill=(180, 180, 180), font=font_title, anchor="mm")
        draw.text((VIDEO_WIDTH // 2, 400), "No winner this time", fill="white", font=font_large, anchor="mm")

    # Final scores
    draw.text((VIDEO_WIDTH // 2, 650), "FINAL SCORE", fill=(150, 150, 150), font=font_medium, anchor="mm")

    # Team 1 score
    team1_color = hex_to_rgb(team1.color)
    draw.rectangle([(60, 720), (VIDEO_WIDTH // 2 - 20, 820)], fill=team1_color, outline="white", width=2)
    draw.text((VIDEO_WIDTH // 4 + 20, 750), team1.id.upper(), fill="white", font=font_medium, anchor="mm")
    draw.text((VIDEO_WIDTH // 4 + 20, 790), f"{score1:.0f}", fill="white", font=font_large, anchor="mm")

    # Team 2 score
    team2_color = hex_to_rgb(team2.color)
    draw.rectangle([(VIDEO_WIDTH // 2 + 20, 720), (VIDEO_WIDTH - 60, 820)], fill=team2_color, outline="white", width=2)
    draw.text((VIDEO_WIDTH * 3 // 4 - 20, 750), team2.id.upper(), fill="white", font=font_medium, anchor="mm")
    draw.text((VIDEO_WIDTH * 3 // 4 - 20, 790), f"{score2:.0f}", fill="white", font=font_large, anchor="mm")

    # Footer
    draw.text((VIDEO_WIDTH // 2, VIDEO_HEIGHT - 100), "THE LEAGUE", fill=(100, 100, 100), font=font_medium, anchor="mm")

    return np.array(canvas)


def save_match_video(
    game_frames: list,
    team1: Team,
    team2: Team,
    scores_over_time: list[tuple[float, float]],
    winner: Team | None,
) -> Path:
    """Create and save an engaging match video."""
    videos_dir = Path("videos")
    videos_dir.mkdir(exist_ok=True)

    frames = []

    # Title frames (show for ~2 seconds at 4fps = 8 frames)
    title_frame = create_title_frame(team1, team2)
    for _ in range(8):
        frames.append(title_frame)

    # Game frames with overlays
    for i, game_frame in enumerate(game_frames):
        if i < len(scores_over_time):
            score1, score2 = scores_over_time[i]
        else:
            score1, score2 = scores_over_time[-1] if scores_over_time else (0, 0)

        frame = create_frame(game_frame, team1, team2, score1, score2, turn=i + 1)
        frames.append(frame)

    # Winner frames (show for ~3 seconds = 12 frames)
    final_score1, final_score2 = scores_over_time[-1] if scores_over_time else (0, 0)
    winner_frame = create_winner_frame(team1, team2, winner, final_score1, final_score2)
    for _ in range(12):
        frames.append(winner_frame)

    # Save video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{team1.id}_vs_{team2.id}_{timestamp}.gif"
    video_path = videos_dir / filename

    imageio.mimsave(video_path, frames, duration=250, loop=0)  # 250ms per frame = 4fps

    return video_path
