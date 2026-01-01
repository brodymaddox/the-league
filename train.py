#!/usr/bin/env python3
"""Train a team's RL agent.

Usage:
    python train.py <team_id>
    python train.py la
    python train.py --all
"""

import argparse

from league import load_config, get_team, train_team


def main():
    parser = argparse.ArgumentParser(description="Train a team's RL agent")
    parser.add_argument("team", nargs="?", help="Team ID to train (e.g., la, nyc)")
    parser.add_argument("--all", action="store_true", help="Train all teams")
    args = parser.parse_args()

    config = load_config()

    if args.all:
        for team in config.teams.values():
            train_team(team, config)
    elif args.team:
        team = get_team(config, args.team)
        train_team(team, config)
    else:
        print("Available teams:")
        for tid, team in config.teams.items():
            status = "trained" if team.trained else "not trained"
            print(f"  {tid}: {team.name} ({status})")
        print("\nUsage: python train.py <team_id> or python train.py --all")


if __name__ == "__main__":
    main()
