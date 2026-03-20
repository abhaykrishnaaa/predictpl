"""
app.py — PredictPL Flask Backend
Full REST API with CSV upload, dataset download, and ML prediction.
"""

import os
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from predictor import PLPredictor

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {"csv"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max upload

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

predictor = PLPredictor()

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ── STARTUP ───────────────────────────────────
@app.before_request
def startup():
    global _initialized
    if not globals().get("_initialized"):
        globals()["_initialized"] = True
        try:
            predictor.initialize()
        except Exception as e:
            print(f"Startup warning: {e}")


# ── HEALTH ────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "PredictPL API is running",
        "model_ready": predictor.model is not None
    })


# ── MODEL INFO ────────────────────────────────
@app.route("/info", methods=["GET"])
def model_info():
    if predictor.model is None:
        return jsonify({"error": "Model not trained yet"}), 503
    return jsonify(predictor.get_model_info())


# ── TEAMS ─────────────────────────────────────
@app.route("/teams", methods=["GET"])
def get_teams():
    return jsonify({"teams": predictor.teams})


# ── PREDICT ───────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if predictor.model is None:
        return jsonify({"error": "Model not ready. Try /retrain first."}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    home = data.get("home_team", "").strip()
    away = data.get("away_team", "").strip()

    if not home or not away:
        return jsonify({"error": "home_team and away_team required"}), 400
    if home == away:
        return jsonify({"error": "Teams must be different"}), 400
    if home not in predictor.teams:
        return jsonify({"error": f"Unknown team: {home}"}), 400
    if away not in predictor.teams:
        return jsonify({"error": f"Unknown team: {away}"}), 400

    try:
        result = predictor.predict(home, away)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── TEAM STATS ────────────────────────────────
@app.route("/stats", methods=["GET"])
def all_stats():
    stats = predictor.get_all_team_stats()
    return jsonify({"teams": stats})


@app.route("/stats/<team_name>", methods=["GET"])
def team_stats(team_name):
    stats = predictor.get_team_stats(team_name)
    if stats:
        return jsonify(stats)
    return jsonify({"error": "Team not found"}), 404


# ── UPLOAD CSV (user dataset) ─────────────────
@app.route("/upload", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Use field name 'file'"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "Only .csv files allowed"}), 400

    filename = secure_filename(f.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(filepath)

    try:
        metrics = predictor.load_user_data(filepath)
        return jsonify({
            "message": f"Dataset '{filename}' loaded and model retrained",
            "metrics": metrics,
            "teams": predictor.teams,
            "total_matches": predictor.n_matches,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Training failed: {str(e)}"}), 500


# ── RETRAIN (re-download + retrain) ──────────
@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        predictor.initialize(force_download=True)
        info = predictor.get_model_info()
        return jsonify({"message": "Model retrained on fresh data", "info": info})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    predictor.initialize()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
