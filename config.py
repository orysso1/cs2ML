# config.py
# Zentrale Konfiguration für das CS2 Prediction Projekt

import os
from pathlib import Path

# ── Pfade ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

RAW_CSV        = DATA_DIR / "matches_raw.csv"
FEATURES_CSV   = DATA_DIR / "matches_features.csv"
MODEL_PATH     = MODEL_DIR / "xgb_model.joblib"
ELO_PATH       = MODEL_DIR / "elo_ratings.joblib"

# ── Scraping ─────────────────────────────────────────────────────────────────
HLTV_MIN_DELAY   = 4.0   # Sekunden zwischen Requests (Bot-Schutz)
HLTV_MAX_DELAY   = 9.0
HLTV_MAX_RETRIES = 5
SCRAPE_PAGES     = 20    # Anzahl Ergebnisseiten von HLTV (à ~100 Matches)

# ── Feature Engineering ───────────────────────────────────────────────────────
FORM_WINDOW      = 10    # Letzte N Matches für Form-Feature
WINRATE_WINDOW   = 30    # Letzte N Tage für Winrate
H2H_WINDOW_DAYS  = 730   # Head-to-Head Fenster in Tagen (2 Jahre)
MIN_MATCHES      = 5     # Min. Matches damit ein Team im Modell erscheint

# ── ELO ───────────────────────────────────────────────────────────────────────
ELO_START = 1500
ELO_K     = 32

# ── Modell ────────────────────────────────────────────────────────────────────
TRAIN_CUTOFF  = "2025-08-16"   # Alles davor = Training, danach = Validierung
RANDOM_STATE  = 42
FEATURES = [
    "elo_diff",
    "winrate_30d_diff",
    "form_diff",
    "h2h_winrate_a",
    "ranking_diff",
    "map_winrate_diff",
    "days_since_last_diff",
    "lineup_age_diff",
]

# ── Streamlit ─────────────────────────────────────────────────────────────────
PAGE_TITLE = "CS2 Match Predictor"
PAGE_ICON  = "🎯"