"""
dataset_loader.py
Downloads real Premier League CSV datasets from football-data.co.uk (2016-2025)
No API key required — completely free public data.
"""

import os
import requests
import pandas as pd
import numpy as np

# football-data.co.uk URL pattern
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"

# Seasons: format is YYYY e.g. 1617 = 2016/17
SEASONS = {
    "2016-17": "1617",
    "2017-18": "1718",
    "2018-19": "1819",
    "2019-20": "1920",
    "2020-21": "2021",
    "2021-22": "2122",
    "2022-23": "2223",
    "2023-24": "2324",
    "2024-25": "2425",
}

DATA_DIR = "data"
MERGED_CSV = os.path.join(DATA_DIR, "pl_all_seasons.csv")

# Columns we actually need
KEEP_COLS = [
    "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",   # Full Time Home/Away Goals, Result
    "HTHG", "HTAG", "HTR",   # Half Time
    "HS", "AS",               # Shots
    "HST", "AST",             # Shots on Target
    "HC", "AC",               # Corners
    "HY", "AY",               # Yellow Cards
    "HR", "AR",               # Red Cards
]


def download_seasons(force=False):
    """Download all season CSVs if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    frames = []

    for label, code in SEASONS.items():
        path = os.path.join(DATA_DIR, f"pl_{label}.csv")

        if not os.path.exists(path) or force:
            url = BASE_URL.format(season=code)
            print(f"Downloading {label} from {url} ...")
            try:
                r = requests.get(url, timeout=15)
                r.raise_for_status()
                with open(path, "wb") as f:
                    f.write(r.content)
                print(f"  ✓ Saved {path}")
            except Exception as e:
                print(f"  ✗ Failed {label}: {e}")
                continue
        else:
            print(f"  → {label} already downloaded.")

        try:
            df = pd.read_csv(path, encoding="latin1")
            df["Season"] = label

            # Keep only available columns
            available = [c for c in KEEP_COLS if c in df.columns]
            df = df[available + ["Season"]].copy()
            frames.append(df)
        except Exception as e:
            print(f"  ✗ Could not parse {label}: {e}")

    if not frames:
        raise RuntimeError("No data could be downloaded. Check internet connection.")

    combined = pd.concat(frames, ignore_index=True)
    combined.dropna(subset=["HomeTeam", "AwayTeam", "FTR"], inplace=True)
    combined.to_csv(MERGED_CSV, index=False)
    print(f"\n✓ Merged {len(combined)} matches → {MERGED_CSV}")
    return combined


def load_data(force_download=False):
    """Load merged CSV, downloading if needed."""
    if not os.path.exists(MERGED_CSV) or force_download:
        return download_seasons(force=force_download)
    df = pd.read_csv(MERGED_CSV, encoding="latin1")
    print(f"✓ Loaded {len(df)} matches from cache.")
    return df


def load_user_csv(filepath):
    """Load a user-uploaded CSV file and normalise columns."""
    df = pd.read_csv(filepath, encoding="latin1")

    # Try to map common alternative column names
    rename_map = {
        "home": "HomeTeam", "away": "AwayTeam",
        "home_team": "HomeTeam", "away_team": "AwayTeam",
        "result": "FTR", "full_time_result": "FTR",
        "home_goals": "FTHG", "away_goals": "FTAG",
        "hg": "FTHG", "ag": "FTAG",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    required = ["HomeTeam", "AwayTeam", "FTR"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Need: HomeTeam, AwayTeam, FTR (H/D/A)")

    df.dropna(subset=required, inplace=True)
    print(f"✓ User CSV loaded: {len(df)} matches")
    return df


def get_team_list(df):
    """Return sorted list of all unique teams."""
    teams = sorted(set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique()))
    return teams
