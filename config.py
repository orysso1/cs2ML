# config.py
import os
from pathlib import Path

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

RAW_CSV       = DATA_DIR / "matches_raw.csv"
FEATURES_CSV  = DATA_DIR / "matches_features.csv"
MODEL_PATH    = MODEL_DIR / "xgb_match.joblib"
MAP_MODEL_PATH= MODEL_DIR / "xgb_map.joblib"
ELO_PATH      = MODEL_DIR / "elo_ratings.joblib"

# ── Scraping ──────────────────────────────────────────────────────────────────
HLTV_MIN_DELAY   = 4.0
HLTV_MAX_DELAY   = 9.0
HLTV_MAX_RETRIES = 5
SCRAPE_PAGES     = 20

# ── Feature Engineering ───────────────────────────────────────────────────────
FORM_WINDOW      = 10
WINRATE_WINDOW   = 30
H2H_WINDOW_DAYS  = 730
MIN_MATCHES      = 5
ELO_START        = 1500
ELO_K            = 32

# ── CS2 Maps ──────────────────────────────────────────────────────────────────
CS2_MAPS = ["mirage", "inferno", "nuke", "dust2", "overpass",
            "ancient", "vertigo", "anubis", "train"]

# ── Match-Prediction Features (aus Kaggle-Spalten) ────────────────────────────
FEATURES = [
    # Rating & Performance
    "rating_diff",
    "adr_diff",
    "kast_diff",
    "kpr_diff",
    "dpr_diff",
    # Winrate
    "team1_overall_winrate",
    "team2_overall_winrate",
    "winrate_diff",
    "team1_lan_winrate",
    "team2_lan_winrate",
    "lan_winrate_diff",
    # H2H
    "winner_head2head_percentage",
    "loser_head2head_percentage",
    "h2h_diff",
    # Form / Streak
    "winner_past3",
    "loser_past3",
    "past3_diff",
    # Star / Weak player
    "star_player_advantage",
    "weakest_link_advantage",
    # Consistency
    "team1_rating_std",
    "team2_rating_std",
    "consistency_advantage",
    # ELO (dynamisch berechnet)
    "elo_diff",
]

# ── Map-Prediction Features ───────────────────────────────────────────────────
MAP_FEATURES_TEMPLATE = [
    "rating_diff", "adr_diff", "kast_diff",
    "{map}_winrate_diff",   # z.B. mirage_winrate_diff
    "elo_diff",
    "h2h_diff",
    "past3_diff",
]

# ── Modell ────────────────────────────────────────────────────────────────────
TRAIN_CUTOFF = "2025-08-16"
RANDOM_STATE = 42

# ── Streamlit ─────────────────────────────────────────────────────────────────
PAGE_TITLE = "CS2 Match Predictor"
PAGE_ICON  = "🎯"