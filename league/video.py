"""Video production with animated assets and audio."""

from pathlib import Path
from datetime import datetime
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from moviepy.editor import (
    ImageSequenceClip,
    AudioFileClip,
    CompositeVideoClip,
    CompositeAudioClip,
    ImageClip,
    concatenate_videoclips,
)
from moviepy.audio.AudioClip import AudioClip

from .config import Team


# Video settings
VIDEO_WIDTH = 720
VIDEO_HEIGHT = 1280
FPS = 24

# Asset paths
ASSETS_DIR = Path("assets")


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def load_font(size: int, bold: bool = False):
    """Load font with fallback."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def create_gradient_background(width: int, height: int, color1: tuple, color2: tuple) -> Image.Image:
    """Create a vertical gradient background."""
    img = Image.new("RGB", (width, height))
    for y in range(height):
        ratio = y / height
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        for x in range(width):
            img.putpixel((x, y), (r, g, b))
    return img


def create_animated_intro(team1: Team, team2: Team, game_name: str, duration: float = 3.0) -> list[np.ndarray]:
    """Create animated intro sequence."""
    frames = []
    num_frames = int(duration * FPS)

    font_title = load_font(56, bold=True)
    font_large = load_font(36, bold=True)
    font_medium = load_font(28)

    team1_color = hex_to_rgb(team1.color)
    team2_color = hex_to_rgb(team2.color)

    for i in range(num_frames):
        progress = i / num_frames

        # Create gradient background
        canvas = create_gradient_background(VIDEO_WIDTH, VIDEO_HEIGHT, (15, 15, 25), (30, 30, 45))
        draw = ImageDraw.Draw(canvas)

        # Animated particles/stars in background
        for j in range(20):
            angle = (i * 2 + j * 18) * math.pi / 180
            radius = 50 + j * 30
            x = VIDEO_WIDTH // 2 + int(math.cos(angle) * radius * (0.5 + progress * 0.5))
            y = VIDEO_HEIGHT // 2 + int(math.sin(angle) * radius * (0.5 + progress * 0.5))
            alpha = int(100 + 50 * math.sin(i * 0.2 + j))
            draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=(255, 215, 0, alpha))

        # Title animation - slide in from top
        title_y = -100 + int(300 * min(1, progress * 2))
        if progress > 0.1:
            draw.text((VIDEO_WIDTH // 2, title_y), "THE LEAGUE", fill=(255, 215, 0), font=font_title, anchor="mm")

        # "PRESENTS" fade in
        if progress > 0.3:
            alpha = min(255, int((progress - 0.3) * 500))
            draw.text((VIDEO_WIDTH // 2, title_y + 60), "PRESENTS", fill=(180, 180, 180), font=font_medium, anchor="mm")

        # Team 1 slides in from left
        if progress > 0.4:
            team1_progress = min(1, (progress - 0.4) * 3)
            team1_x = int(-400 + 400 * team1_progress + VIDEO_WIDTH // 2)

            # Glowing box effect
            for glow in range(3, 0, -1):
                glow_color = tuple(min(255, c + 30 * glow) for c in team1_color)
                draw.rectangle(
                    [(60 - glow * 2, 400 - glow * 2), (VIDEO_WIDTH - 60 + glow * 2, 520 + glow * 2)],
                    outline=glow_color, width=2
                )
            draw.rectangle([(60, 400), (VIDEO_WIDTH - 60, 520)], fill=team1_color, outline="white", width=3)
            draw.text((VIDEO_WIDTH // 2, 460), team1.name, fill="white", font=font_large, anchor="mm")

        # VS with pulse effect
        if progress > 0.5:
            vs_scale = 1.0 + 0.1 * math.sin(i * 0.3)
            vs_size = int(56 * vs_scale)
            vs_font = load_font(vs_size, bold=True)
            draw.text((VIDEO_WIDTH // 2, 600), "VS", fill=(255, 215, 0), font=vs_font, anchor="mm")

        # Team 2 slides in from right
        if progress > 0.6:
            team2_progress = min(1, (progress - 0.6) * 3)

            for glow in range(3, 0, -1):
                glow_color = tuple(min(255, c + 30 * glow) for c in team2_color)
                draw.rectangle(
                    [(60 - glow * 2, 680 - glow * 2), (VIDEO_WIDTH - 60 + glow * 2, 800 + glow * 2)],
                    outline=glow_color, width=2
                )
            draw.rectangle([(60, 680), (VIDEO_WIDTH - 60, 800)], fill=team2_color, outline="white", width=3)
            draw.text((VIDEO_WIDTH // 2, 740), team2.name, fill="white", font=font_large, anchor="mm")

        # Game name fade in at bottom
        if progress > 0.8:
            draw.text((VIDEO_WIDTH // 2, 950), game_name.upper(), fill=(150, 150, 150), font=font_medium, anchor="mm")

        frames.append(np.array(canvas))

    return frames


def create_game_frame(
    game_frame: np.ndarray,
    team1: Team,
    team2: Team,
    score1: float,
    score2: float,
    turn: int,
    frame_num: int,
) -> np.ndarray:
    """Create a single game frame with HUD overlay."""
    # Create gradient background
    canvas = create_gradient_background(VIDEO_WIDTH, VIDEO_HEIGHT, (15, 15, 25), (25, 25, 40))
    draw = ImageDraw.Draw(canvas)

    font_large = load_font(42, bold=True)
    font_medium = load_font(28)
    font_small = load_font(20)

    team1_color = hex_to_rgb(team1.color)
    team2_color = hex_to_rgb(team2.color)

    # Animated header bar
    header_pulse = int(5 * math.sin(frame_num * 0.1))
    draw.rectangle([(0, 0), (VIDEO_WIDTH, 90 + header_pulse)], fill=(20, 20, 35))
    draw.text((VIDEO_WIDTH // 2, 45), "THE LEAGUE", fill=(255, 215, 0), font=font_large, anchor="mm")

    # Team score boxes with glow effect
    # Team 1
    for glow in range(2, 0, -1):
        draw.rectangle(
            [(20 - glow, 110 - glow), (VIDEO_WIDTH // 2 - 10 + glow, 200 + glow)],
            outline=tuple(min(255, c + 40 * glow) for c in team1_color), width=1
        )
    draw.rectangle([(20, 110), (VIDEO_WIDTH // 2 - 10, 200)], fill=team1_color)
    draw.text((VIDEO_WIDTH // 4, 135), team1.name[:15], fill="white", font=font_small, anchor="mm")
    draw.text((VIDEO_WIDTH // 4, 170), f"{score1:.0f}", fill="white", font=font_medium, anchor="mm")

    # Team 2
    for glow in range(2, 0, -1):
        draw.rectangle(
            [(VIDEO_WIDTH // 2 + 10 - glow, 110 - glow), (VIDEO_WIDTH - 20 + glow, 200 + glow)],
            outline=tuple(min(255, c + 40 * glow) for c in team2_color), width=1
        )
    draw.rectangle([(VIDEO_WIDTH // 2 + 10, 110), (VIDEO_WIDTH - 20, 200)], fill=team2_color)
    draw.text((VIDEO_WIDTH * 3 // 4, 135), team2.name[:15], fill="white", font=font_small, anchor="mm")
    draw.text((VIDEO_WIDTH * 3 // 4, 170), f"{score2:.0f}", fill="white", font=font_medium, anchor="mm")

    # VS badge with pulse
    vs_pulse = int(3 * math.sin(frame_num * 0.15))
    draw.ellipse(
        [(VIDEO_WIDTH // 2 - 25 - vs_pulse, 130 - vs_pulse),
         (VIDEO_WIDTH // 2 + 25 + vs_pulse, 180 + vs_pulse)],
        fill=(40, 40, 55), outline=(255, 215, 0), width=2
    )
    draw.text((VIDEO_WIDTH // 2, 155), "VS", fill=(255, 215, 0), font=font_small, anchor="mm")

    # Game frame area
    if game_frame is not None:
        game_img = Image.fromarray(game_frame)

        game_area_width = VIDEO_WIDTH - 60
        game_area_height = VIDEO_HEIGHT - 380

        scale = min(game_area_width / game_img.width, game_area_height / game_img.height)
        new_width = int(game_img.width * scale)
        new_height = int(game_img.height * scale)
        game_img = game_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        x_offset = (VIDEO_WIDTH - new_width) // 2
        y_offset = 230 + (game_area_height - new_height) // 2

        # Game border with glow
        for glow in range(4, 0, -1):
            draw.rectangle(
                [(x_offset - 5 - glow, y_offset - 5 - glow),
                 (x_offset + new_width + 5 + glow, y_offset + new_height + 5 + glow)],
                outline=(255, 215, 0, 255 // glow), width=1
            )

        canvas.paste(game_img, (x_offset, y_offset))

    # Footer with turn counter
    draw.rectangle([(0, VIDEO_HEIGHT - 90), (VIDEO_WIDTH, VIDEO_HEIGHT)], fill=(20, 20, 35))
    draw.rectangle([(100, VIDEO_HEIGHT - 70), (VIDEO_WIDTH - 100, VIDEO_HEIGHT - 20)],
                   fill=(40, 40, 55), outline=(255, 215, 0), width=2)
    draw.text((VIDEO_WIDTH // 2, VIDEO_HEIGHT - 45), f"TURN {turn}", fill="white", font=font_medium, anchor="mm")

    return np.array(canvas)


def create_winner_sequence(
    team1: Team, team2: Team, winner: Team | None, score1: float, score2: float, duration: float = 4.0
) -> list[np.ndarray]:
    """Create animated winner announcement."""
    frames = []
    num_frames = int(duration * FPS)

    font_title = load_font(52, bold=True)
    font_large = load_font(36, bold=True)
    font_medium = load_font(28)

    team1_color = hex_to_rgb(team1.color)
    team2_color = hex_to_rgb(team2.color)
    winner_color = hex_to_rgb(winner.color) if winner else (150, 150, 150)

    for i in range(num_frames):
        progress = i / num_frames

        canvas = create_gradient_background(VIDEO_WIDTH, VIDEO_HEIGHT, (15, 15, 25), (30, 30, 45))
        draw = ImageDraw.Draw(canvas)

        # Celebration particles
        if winner:
            for j in range(30):
                t = (i + j * 5) % 100 / 100
                x = (j * 73) % VIDEO_WIDTH
                y = int(t * VIDEO_HEIGHT * 1.5) - 200
                size = 3 + int(3 * math.sin(i * 0.1 + j))
                color = winner_color if j % 2 == 0 else (255, 215, 0)
                draw.ellipse([(x - size, y - size), (x + size, y + size)], fill=color)

        # Result text with animation
        if winner:
            # Scale effect
            scale = 1.0 + 0.05 * math.sin(i * 0.2)
            result_text = "WINNER!"
            result_color = (255, 215, 0)
        else:
            scale = 1.0
            result_text = "DRAW!"
            result_color = (180, 180, 180)

        if progress > 0.1:
            y_offset = int(-50 + 250 * min(1, (progress - 0.1) * 3))
            draw.text((VIDEO_WIDTH // 2, y_offset), result_text, fill=result_color, font=font_title, anchor="mm")

        # Winner team box (animated)
        if progress > 0.3 and winner:
            box_progress = min(1, (progress - 0.3) * 2.5)
            box_width = int(box_progress * (VIDEO_WIDTH - 120))
            box_left = (VIDEO_WIDTH - box_width) // 2

            # Glowing effect
            for glow in range(5, 0, -1):
                glow_alpha = int(150 / glow)
                glow_color = tuple(min(255, c + 20 * glow) for c in winner_color)
                draw.rectangle(
                    [(box_left - glow * 3, 350 - glow * 3), (box_left + box_width + glow * 3, 500 + glow * 3)],
                    outline=glow_color, width=2
                )

            draw.rectangle([(box_left, 350), (box_left + box_width, 500)], fill=winner_color, outline=(255, 215, 0), width=4)
            if box_progress > 0.5:
                draw.text((VIDEO_WIDTH // 2, 400), winner.name, fill="white", font=font_large, anchor="mm")
                draw.text((VIDEO_WIDTH // 2, 450), "CHAMPION", fill=(255, 215, 0), font=font_medium, anchor="mm")

        # Final scores
        if progress > 0.6:
            draw.text((VIDEO_WIDTH // 2, 600), "FINAL SCORE", fill=(150, 150, 150), font=font_medium, anchor="mm")

            # Team 1 score
            draw.rectangle([(60, 660), (VIDEO_WIDTH // 2 - 20, 780)], fill=team1_color, outline="white", width=2)
            draw.text((VIDEO_WIDTH // 4 + 20, 700), team1.id.upper(), fill="white", font=font_medium, anchor="mm")
            draw.text((VIDEO_WIDTH // 4 + 20, 740), f"{score1:.0f}", fill="white", font=font_large, anchor="mm")

            # Team 2 score
            draw.rectangle([(VIDEO_WIDTH // 2 + 20, 660), (VIDEO_WIDTH - 60, 780)], fill=team2_color, outline="white", width=2)
            draw.text((VIDEO_WIDTH * 3 // 4 - 20, 700), team2.id.upper(), fill="white", font=font_medium, anchor="mm")
            draw.text((VIDEO_WIDTH * 3 // 4 - 20, 740), f"{score2:.0f}", fill="white", font=font_large, anchor="mm")

        # Footer
        if progress > 0.8:
            draw.text((VIDEO_WIDTH // 2, VIDEO_HEIGHT - 100), "THE LEAGUE", fill=(100, 100, 100), font=font_medium, anchor="mm")

        frames.append(np.array(canvas))

    return frames


def generate_silent_audio(duration: float, fps: int = 44100) -> AudioClip:
    """Generate silent audio clip."""
    def make_frame(t):
        return np.zeros((1,))
    return AudioClip(make_frame, duration=duration, fps=fps)


def save_match_video(
    game_frames: list,
    team1: Team,
    team2: Team,
    scores_over_time: list[tuple[float, float]],
    winner: Team | None,
    game_name: str = "Connect Four",
) -> Path:
    """Create and save a production-quality match video with audio."""
    videos_dir = Path("videos")
    videos_dir.mkdir(exist_ok=True)

    all_frames = []

    # Intro sequence
    intro_frames = create_animated_intro(team1, team2, game_name, duration=3.0)
    all_frames.extend(intro_frames)

    # Game frames with HUD
    for i, game_frame in enumerate(game_frames):
        if i < len(scores_over_time):
            score1, score2 = scores_over_time[i]
        else:
            score1, score2 = scores_over_time[-1] if scores_over_time else (0, 0)

        frame = create_game_frame(game_frame, team1, team2, score1, score2, turn=i + 1, frame_num=i)
        all_frames.append(frame)

    # Winner sequence
    final_score1, final_score2 = scores_over_time[-1] if scores_over_time else (0, 0)
    winner_frames = create_winner_sequence(team1, team2, winner, final_score1, final_score2, duration=4.0)
    all_frames.extend(winner_frames)

    # Create video clip
    video = ImageSequenceClip(all_frames, fps=FPS)

    # Try to add background music if available
    audio_path = ASSETS_DIR / "audio" / "background.mp3"
    if audio_path.exists():
        try:
            audio = AudioFileClip(str(audio_path))
            # Loop audio if needed
            if audio.duration < video.duration:
                audio = audio.loop(duration=video.duration)
            else:
                audio = audio.subclip(0, video.duration)
            audio = audio.volumex(0.3)  # Lower volume
            video = video.set_audio(audio)
        except Exception:
            pass  # Continue without audio if loading fails

    # Save video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{team1.id}_vs_{team2.id}_{timestamp}.mp4"
    video_path = videos_dir / filename

    video.write_videofile(
        str(video_path),
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        verbose=False,
        logger=None,
    )

    video.close()

    return video_path
