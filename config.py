# config.py
import os
from pathlib import Path

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

RAW_CSV           = DATA_DIR / "matches_raw.csv"
FEATURES_CSV      = DATA_DIR / "matches_features.csv"
MODEL_PATH        = MODEL_DIR / "xgb_match.joblib"
MAP_MODEL_PATH    = MODEL_DIR / "xgb_map.joblib"
ELO_PATH          = MODEL_DIR / "elo_ratings.joblib"
ENSEMBLE_PATH     = MODEL_DIR / "ensemble.joblib"
CALIBRATED_PATH   = MODEL_DIR / "calibrated.joblib"

# ── Scraping ──────────────────────────────────────────────────────────────────
HLTV_MIN_DELAY   = 4.0
HLTV_MAX_DELAY   = 9.0
HLTV_MAX_RETRIES = 5
SCRAPE_PAGES     = 20

# ── Feature Engineering ───────────────────────────────────────────────────────
FORM_WINDOW     = 10
WINRATE_WINDOW  = 30
H2H_WINDOW_DAYS = 730
MIN_MATCHES     = 5

# ── ELO — Tier-gewichtet ──────────────────────────────────────────────────────
ELO_START    = 1500
ELO_K_MAJOR  = 48    # S-Tier: Majors
ELO_K_TIER1  = 32    # A-Tier: IEM, BLAST Premier, ESL Pro League
ELO_K_TIER2  = 20    # B-Tier: Regional Leagues, Tier-2 Events
ELO_K_ONLINE = 12    # Online-Matches (weniger aussagekräftig)
ELO_K        = 32    # Default falls kein Tier erkannt

# Event-Namen → Tier (Lowercase-Matching)
TIER_KEYWORDS = {
    "major":          ELO_K_MAJOR,
    "iem katowice":   ELO_K_MAJOR,
    "iem cologne":    ELO_K_MAJOR,
    "pgl major":      ELO_K_MAJOR,
    "blast premier":  ELO_K_TIER1,
    "esl pro league": ELO_K_TIER1,
    "iem":            ELO_K_TIER1,
    "blast bounty":   ELO_K_TIER1,
    "esl":            ELO_K_TIER2,
    "cct":            ELO_K_TIER2,
    "esea":           ELO_K_TIER2,
    "online":         ELO_K_ONLINE,
}

# ── Recency-Gewichtung ────────────────────────────────────────────────────────
RECENCY_HALFLIFE_DAYS = 365   # Nach 365 Tagen hat ein Match halbes Gewicht

# ── CS2 Maps ──────────────────────────────────────────────────────────────────
CS2_MAPS = ["mirage", "inferno", "nuke", "dust2", "overpass",
            "ancient", "vertigo", "anubis", "train"]

# ── Features ─────────────────────────────────────────────────────────────────
FEATURES = [
    # Rating & Performance
    "rating_diff",
    "adr_diff",
    "kast_diff",
    "kpr_diff",
    "dpr_diff",
    # Spieler-Individualstats
    "top_player_rating_diff",    # Bester Spieler Team A vs B
    "bot_player_rating_diff",    # Schwächster Spieler Team A vs B
    "rating_std_diff",           # Konsistenz (ausgeglichen vs. Star-abhängig)
    "star_player_advantage",
    "weakest_link_advantage",
    "team1_rating_std",
    "team2_rating_std",
    "consistency_advantage",
    # Winrate
    "team1_overall_winrate",
    "team2_overall_winrate",
    "winrate_diff",
    "team1_lan_winrate",
    "team2_lan_winrate",
    "lan_winrate_diff",
    # Form & H2H
    "team1_past3",
    "team2_past3",
    "team1_h2h_pct",
    "team2_h2h_pct",
    # Qualitätsbereinigte Winrate (gegen starke Gegner)
    "quality_winrate_diff",
    # Momentum (Siege-Serie)
    "momentum_a",
    "momentum_b",
    "momentum_diff",
    # Event-Kontext
    "is_lan",
    "event_tier",                # 3=Major, 2=Tier1, 1=Tier2, 0=Online
    # ELO
    "elo_diff",
]

# ── Map-Features ──────────────────────────────────────────────────────────────
MAP_FEATURES_TEMPLATE = [
    "rating_diff", "adr_diff", "kast_diff",
    "{map}_winrate_diff",
    "elo_diff",
    "team1_h2h_pct",
    "team1_past3", "team2_past3",
    "is_lan",
]

# ── Modell ────────────────────────────────────────────────────────────────────
TRAIN_CUTOFF = "2024-01-01"
RANDOM_STATE = 42

# ── Streamlit ─────────────────────────────────────────────────────────────────
PAGE_TITLE = "CS2 Match Predictor"
PAGE_ICON  = "🎯"