"""
predictor.py
Trains and runs the Premier League match prediction ML model.
Uses ensemble of Random Forest + Gradient Boosting.
"""

import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

from dataset_loader import load_data, load_user_csv, get_team_list
from features import build_feature_matrix, get_prediction_features

MODEL_PATH = "models/pl_model.pkl"
SCALER_PATH = "models/pl_scaler.pkl"
DATA_CACHE = "models/df_cache.pkl"


class PLPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.df = None
        self.teams = []
        self.accuracy = None
        self.report = {}
        self.n_matches = 0
        self.seasons_loaded = []

    def initialize(self, force_download=False):
        """Load data + model on startup."""
        os.makedirs("models", exist_ok=True)

        # Load data
        self.df = load_data(force_download=force_download)
        self.teams = sorted(get_team_list(self.df))
        self.n_matches = len(self.df)

        if "Season" in self.df.columns:
            self.seasons_loaded = sorted(self.df["Season"].unique().tolist())

        # Train or load model
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and not force_download:
            print("Loading existing model...")
            self._load_model()
        else:
            print("Training model on loaded data...")
            self.train()

    def _load_model(self):
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)

    def _save_model(self):
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(self.scaler, f)

    def train(self):
        """Train ensemble model on current self.df."""
        print(f"Building features from {len(self.df)} matches...")
        X, y, feature_names = build_feature_matrix(self.df)
        print(f"Feature matrix: {X.shape[0]} training samples, {X.shape[1]} features")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        # Ensemble: RF + GB + LR voting
        rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5,
                                    random_state=42, n_jobs=-1)
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.08,
                                        random_state=42)
        lr = LogisticRegression(max_iter=500, C=1.0, random_state=42)

        self.model = VotingClassifier(
            estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
            voting="soft"
        )
        self.model.fit(X_train_s, y_train)

        y_pred = self.model.predict(X_test_s)
        self.accuracy = round(accuracy_score(y_test, y_pred), 4)
        report = classification_report(y_test, y_pred,
                                       target_names=["Home Win", "Draw", "Away Win"],
                                       output_dict=True)
        self.report = {
            "Home Win": round(report["Home Win"]["f1-score"], 3),
            "Draw":     round(report["Draw"]["f1-score"], 3),
            "Away Win": round(report["Away Win"]["f1-score"], 3),
        }

        print(f"✓ Model accuracy: {self.accuracy:.2%}")
        print(f"  F1 → {self.report}")

        self._save_model()
        return {"accuracy": self.accuracy, "f1": self.report, "samples": len(X)}

    def load_user_data(self, filepath):
        """Replace/extend dataset with user-uploaded CSV."""
        user_df = load_user_csv(filepath)
        import pandas as pd
        self.df = pd.concat([self.df, user_df], ignore_index=True) if self.df is not None else user_df
        self.teams = sorted(get_team_list(self.df))
        self.n_matches = len(self.df)
        result = self.train()
        return result

    def predict(self, home_team, away_team):
        """Predict outcome for a fixture."""
        X, hf, af, h2h = get_prediction_features(self.df, home_team, away_team)
        X_s = self.scaler.transform(X)

        probs = self.model.predict_proba(X_s)[0]
        classes = list(self.model.classes_)

        prob_map = {}
        for i, c in enumerate(classes):
            prob_map[c] = float(probs[i])

        hp = round(prob_map.get(0, 0.33) * 100)
        dp = round(prob_map.get(1, 0.33) * 100)
        ap = round(prob_map.get(2, 0.34) * 100)

        # Normalise to 100
        total = hp + dp + ap
        diff = 100 - total
        ap += diff

        if hp > ap and hp > dp:
            outcome = f"{home_team} Win"
            confidence = hp
        elif ap > hp and ap > dp:
            outcome = f"{away_team} Win"
            confidence = ap
        else:
            outcome = "Draw"
            confidence = dp

        return {
            "prediction": outcome,
            "confidence": confidence,
            "home_win_prob": hp,
            "draw_prob": dp,
            "away_win_prob": ap,
            "home_team": home_team,
            "away_team": away_team,
            "home_stats": {
                "win_rate": round(hf["win_rate"] * 100, 1),
                "avg_gf":   round(hf["avg_gf"], 2),
                "avg_ga":   round(hf["avg_ga"], 2),
                "form":     round(hf["form_score"] * 100, 1),
                "games":    hf["total_games"],
            },
            "away_stats": {
                "win_rate": round(af["win_rate"] * 100, 1),
                "avg_gf":   round(af["avg_gf"], 2),
                "avg_ga":   round(af["avg_ga"], 2),
                "form":     round(af["form_score"] * 100, 1),
                "games":    af["total_games"],
            },
            "h2h": {
                "home_wins": round(h2h["h2h_hw"] * 100, 1),
                "draws":     round(h2h["h2h_d"] * 100, 1),
                "away_wins": round(h2h["h2h_aw"] * 100, 1),
                "games":     h2h["h2h_games"],
            },
            "model_accuracy": self.accuracy,
        }

    def get_team_stats(self, team_name):
        """Full team stat summary from all loaded data."""
        import pandas as pd
        df = self.df.copy()
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

        home_matches = df[df["HomeTeam"] == team_name]
        away_matches = df[df["AwayTeam"] == team_name]

        if home_matches.empty and away_matches.empty:
            return None

        hw = (home_matches["FTR"] == "H").sum()
        hd = (home_matches["FTR"] == "D").sum()
        hl = (home_matches["FTR"] == "A").sum()
        aw = (away_matches["FTR"] == "A").sum()
        ad = (away_matches["FTR"] == "D").sum()
        al = (away_matches["FTR"] == "H").sum()

        total_w = int(hw + aw)
        total_d = int(hd + ad)
        total_l = int(hl + al)
        total   = total_w + total_d + total_l

        hgf = home_matches["FTHG"].sum() if "FTHG" in home_matches else 0
        hga = home_matches["FTAG"].sum() if "FTAG" in home_matches else 0
        agf = away_matches["FTAG"].sum() if "FTAG" in away_matches else 0
        aga = away_matches["FTHG"].sum() if "FTHG" in away_matches else 0

        return {
            "team": team_name,
            "total_matches": total,
            "wins": total_w, "draws": total_d, "losses": total_l,
            "win_pct": round(total_w / max(total, 1) * 100, 1),
            "goals_for": int(hgf + agf),
            "goals_against": int(hga + aga),
            "goal_diff": int(hgf + agf - hga - aga),
            "home_record": {"w": int(hw), "d": int(hd), "l": int(hl)},
            "away_record": {"w": int(aw), "d": int(ad), "l": int(al)},
            "seasons": self.seasons_loaded,
        }

    def get_all_team_stats(self):
        return [self.get_team_stats(t) for t in self.teams if self.get_team_stats(t)]

    def get_model_info(self):
        return {
            "accuracy": self.accuracy,
            "f1_scores": self.report,
            "matches_trained": self.n_matches,
            "seasons": self.seasons_loaded,
            "teams": len(self.teams),
        }
