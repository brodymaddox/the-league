#!/usr/bin/env python3
"""Run a match between two teams.

Usage:
    python compete.py <team1> <team2>
    python compete.py la nyc
    python compete.py la nyc --no-video
"""

import argparse
import json

from league import load_config, get_team, run_match


def main():
    parser = argparse.ArgumentParser(description="Run a match between two teams")
    parser.add_argument("team1", help="First team ID")
    parser.add_argument("team2", help="Second team ID")
    parser.add_argument("--no-video", action="store_true", help="Skip video recording")
    args = parser.parse_args()

    config = load_config()

    team1 = get_team(config, args.team1)
    team2 = get_team(config, args.team2)

    # Verify both teams are trained
    if not team1.trained:
        print(f"Error: {team1.name} has not been trained yet. Run: python train.py {team1.id}")
        return
    if not team2.trained:
        print(f"Error: {team2.name} has not been trained yet. Run: python train.py {team2.id}")
        return

    result = run_match(team1, team2, config, record=not args.no_video)

    # Save result
    print("\nMatch result:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
