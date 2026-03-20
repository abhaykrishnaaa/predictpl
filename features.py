"""
features.py
Feature engineering for Premier League match prediction.
Builds rich per-fixture features from historical match data.
"""

import pandas as pd
import numpy as np


def form_score(results, n=5):
    """
    Compute form score from last n results.
    W=3pts, D=1pt, L=0pts → normalised to [0,1]
    """
    recent = results[-n:]
    pts = sum(3 if r == "W" else 1 if r == "D" else 0 for r in recent)
    return pts / (n * 3)


def build_team_history(df):
    """
    Build a per-team match history dict:
    { TeamName: [ {date, result, goals_for, goals_against, was_home}, ... ] }
    Sorted by date ascending.
    """
    history = {}

    # Parse dates
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        ftr = row["FTR"]  # H, D, A
        hg = row.get("FTHG", np.nan)
        ag = row.get("FTAG", np.nan)

        if home not in history:
            history[home] = []
        if away not in history:
            history[away] = []

        history[home].append({
            "date": row["Date"],
            "result": "W" if ftr == "H" else "D" if ftr == "D" else "L",
            "gf": hg, "ga": ag, "home": True
        })
        history[away].append({
            "date": row["Date"],
            "result": "W" if ftr == "A" else "D" if ftr == "D" else "L",
            "gf": ag, "ga": hg, "home": False
        })

    return history


def get_team_features(history, team, before_date, n_form=5):
    """Get features for a team from matches strictly before before_date."""
    matches = [m for m in history.get(team, []) if m["date"] < before_date]
    if not matches:
        return None

    results = [m["result"] for m in matches]
    gf = [m["gf"] for m in matches if not np.isnan(m["gf"])]
    ga = [m["ga"] for m in matches if not np.isnan(m["ga"])]

    total = len(matches)
    wins  = results.count("W")
    draws = results.count("D")
    losses = results.count("L")

    return {
        "win_rate":    wins / total,
        "draw_rate":   draws / total,
        "loss_rate":   losses / total,
        "avg_gf":      np.mean(gf) if gf else 0,
        "avg_ga":      np.mean(ga) if ga else 0,
        "goal_diff":   (np.mean(gf) - np.mean(ga)) if gf and ga else 0,
        "form_score":  form_score(results, n_form),
        "form_last3":  form_score(results, 3),
        "total_games": total,
    }


def get_h2h_features(df, home, away, before_date):
    """Head-to-head record between two teams."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    h2h = df[
        (df["Date"] < before_date) &
        (
            ((df["HomeTeam"] == home) & (df["AwayTeam"] == away)) |
            ((df["HomeTeam"] == away) & (df["AwayTeam"] == home))
        )
    ]

    if h2h.empty:
        return {"h2h_hw": 0.33, "h2h_d": 0.33, "h2h_aw": 0.34, "h2h_games": 0}

    total = len(h2h)
    home_wins = sum(
        1 for _, r in h2h.iterrows()
        if (r["HomeTeam"] == home and r["FTR"] == "H") or
           (r["HomeTeam"] == away and r["FTR"] == "A")
    )
    away_wins = sum(
        1 for _, r in h2h.iterrows()
        if (r["HomeTeam"] == away and r["FTR"] == "H") or
           (r["HomeTeam"] == home and r["FTR"] == "A")
    )
    draws = total - home_wins - away_wins

    return {
        "h2h_hw":    home_wins / total,
        "h2h_d":     draws / total,
        "h2h_aw":    away_wins / total,
        "h2h_games": total,
    }


def build_feature_matrix(df, n_form=5):
    """
    Build the full feature matrix from match dataframe.
    Returns X (features), y (labels: 0=H, 1=D, 2=A), feature_names.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "FTR"]).sort_values("Date")

    history = build_team_history(df)

    rows = []
    labels = []

    FEATURE_NAMES = [
        "home_win_rate", "home_draw_rate", "home_loss_rate",
        "home_avg_gf", "home_avg_ga", "home_goal_diff",
        "home_form_score", "home_form_last3",
        "away_win_rate", "away_draw_rate", "away_loss_rate",
        "away_avg_gf", "away_avg_ga", "away_goal_diff",
        "away_form_score", "away_form_last3",
        "h2h_hw", "h2h_d", "h2h_aw",
        "strength_diff", "form_diff",
    ]

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        date = row["Date"]
        ftr = row["FTR"]

        if ftr not in ("H", "D", "A"):
            continue

        hf = get_team_features(history, home, date, n_form)
        af = get_team_features(history, away, date, n_form)
        if hf is None or af is None:
            continue

        h2h = get_h2h_features(df, home, away, date)

        features = [
            hf["win_rate"], hf["draw_rate"], hf["loss_rate"],
            hf["avg_gf"], hf["avg_ga"], hf["goal_diff"],
            hf["form_score"], hf["form_last3"],
            af["win_rate"], af["draw_rate"], af["loss_rate"],
            af["avg_gf"], af["avg_ga"], af["goal_diff"],
            af["form_score"], af["form_last3"],
            h2h["h2h_hw"], h2h["h2h_d"], h2h["h2h_aw"],
            hf["win_rate"] - af["win_rate"],
            hf["form_score"] - af["form_score"],
        ]

        rows.append(features)
        labels.append(0 if ftr == "H" else 1 if ftr == "D" else 2)

    X = np.array(rows, dtype=float)
    y = np.array(labels, dtype=int)
    return X, y, FEATURE_NAMES


def get_prediction_features(df, home_team, away_team):
    """
    Build features for a single future prediction
    using all available history in df.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    future = df["Date"].max() + pd.Timedelta(days=1)

    history = build_team_history(df)

    hf = get_team_features(history, home_team, future)
    af = get_team_features(history, away_team, future)

    if hf is None:
        raise ValueError(f"No history found for team: {home_team}")
    if af is None:
        raise ValueError(f"No history found for team: {away_team}")

    h2h = get_h2h_features(df, home_team, away_team, future)

    features = [
        hf["win_rate"], hf["draw_rate"], hf["loss_rate"],
        hf["avg_gf"], hf["avg_ga"], hf["goal_diff"],
        hf["form_score"], hf["form_last3"],
        af["win_rate"], af["draw_rate"], af["loss_rate"],
        af["avg_gf"], af["avg_ga"], af["goal_diff"],
        af["form_score"], af["form_last3"],
        h2h["h2h_hw"], h2h["h2h_d"], h2h["h2h_aw"],
        hf["win_rate"] - af["win_rate"],
        hf["form_score"] - af["form_score"],
    ]

    return np.array([features], dtype=float), hf, af, h2h
