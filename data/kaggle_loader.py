# data/kaggle_loader.py
# Lädt das öffentliche CS2-HLTV-Dataset von Kaggle als Bootstrap-Alternative.
# Kein Scraping nötig — gut als sofortiger Startpunkt.
#
# Dataset: https://www.kaggle.com/datasets/griffindesroches/cs2-hltv-professional-match-statistics-dataset
#
# Setup:
#   pip install kaggle
#   Kaggle API Key unter ~/.kaggle/kaggle.json ablegen
#   Dann: python data/kaggle_loader.py

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, RAW_CSV

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Kaggle-Download
# ─────────────────────────────────────────────────────────────────────────────

KAGGLE_DATASET = "griffindesroches/cs2-hltv-professional-match-statistics-dataset"


def download_kaggle(dest: Path = DATA_DIR) -> Path:
    """Lädt das Dataset via Kaggle-API herunter."""
    try:
        import kaggle  # noqa
    except ImportError:
        print("Kaggle-Paket fehlt. Installiere mit: pip install kaggle")
        sys.exit(1)

    dest.mkdir(exist_ok=True)
    import kaggle
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(KAGGLE_DATASET, path=str(dest), unzip=True)
    csv_files = list(dest.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("Kein CSV nach Kaggle-Download gefunden.")
    return csv_files[0]


# ─────────────────────────────────────────────────────────────────────────────
# Schema-Normalisierung
# ─────────────────────────────────────────────────────────────────────────────

def normalize_kaggle_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalisiert das Kaggle-CSV auf das interne Schema:
    match_id, date, team_a, team_b, score_a, score_b, winner, event, maps
    """
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Versuche bekannte Spaltennamen zu mappen
    col_map = {
        # Kaggle-Spaltenname → internes Schema
        "match_id":      "match_id",
        "id":            "match_id",
        "date":          "date",
        "team1":         "team_a",
        "team1_name":    "team_a",
        "team_1":        "team_a",
        "team2":         "team_b",
        "team_2":        "team_b",
        "team2_name":    "team_b",
        "team1_score":   "score_a",
        "score1":        "score_a",
        "maps_won_team1":"score_a",
        "team2_score":   "score_b",
        "score2":        "score_b",
        "maps_won_team2":"score_b",
        "winner":        "winner",
        "winning_team":  "winner",
        "event":         "event",
        "event_name":    "event",
        "tournament":    "event",
    }

    rename = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=rename)

    required = ["team_a", "team_b"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Pflicht-Spalte '{col}' nicht im Kaggle-CSV gefunden.\n"
                             f"Verfügbare Spalten: {list(df.columns)}")

    # Fehlende Spalten mit Defaults füllen
    if "match_id" not in df.columns:
        df["match_id"] = [f"kaggle_{i}" for i in range(len(df))]
    if "event" not in df.columns:
        df["event"] = "Unknown"
    if "score_a" not in df.columns:
        df["score_a"] = np.nan
    if "score_b" not in df.columns:
        df["score_b"] = np.nan

    # Gewinner ableiten falls fehlend
    if "winner" not in df.columns:
        mask_a = df["score_a"] > df["score_b"]
        mask_b = df["score_b"] > df["score_a"]
        df["winner"] = np.where(mask_a, df["team_a"],
                        np.where(mask_b, df["team_b"], np.nan))

    # Datum normalisieren
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Maps
    df["maps"] = (pd.to_numeric(df["score_a"], errors="coerce").fillna(0) +
                  pd.to_numeric(df["score_b"], errors="coerce").fillna(0)).astype(int)

    # Auf internes Schema beschränken
    keep = ["match_id", "date", "team_a", "team_b", "score_a", "score_b",
            "winner", "event", "maps"]
    df = df[[c for c in keep if c in df.columns]]

    df = df.dropna(subset=["date", "team_a", "team_b"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Manuelle CSV-Übergabe
# ─────────────────────────────────────────────────────────────────────────────

def load_custom_csv(path: str | Path) -> pd.DataFrame:
    """
    Lädt ein beliebiges Match-CSV und normalisiert es.
    Kann mit dem Kaggle-Dataset oder eigenen Exports genutzt werden.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV nicht gefunden: {path}")

    df = pd.read_csv(path)
    log.info(f"CSV geladen: {path} ({len(df)} Zeilen, Spalten: {list(df.columns)})")
    df = normalize_kaggle_df(df)
    df.to_csv(RAW_CSV, index=False)
    log.info(f"Normalisiert und gespeichert: {RAW_CSV}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Synthetische Demo-Daten (zum Testen ohne echte Daten)
# ─────────────────────────────────────────────────────────────────────────────

TEAMS = [
    "Natus Vincere", "FaZe Clan", "G2 Esports", "Team Vitality",
    "Heroic", "MOUZ", "Team Spirit", "Astralis",
    "Cloud9", "ENCE", "Liquid", "NIP",
    "BIG", "Complexity", "paiN", "FURIA",
]

EVENTS = ["IEM Katowice", "IEM Cologne", "BLAST Premier", "ESL Pro League",
          "PGL Major", "BLAST Bounty", "ESL Impact", "CCT"]


def generate_demo_data(n_matches: int = 2000) -> pd.DataFrame:
    """
    Erzeugt realistische synthetische Trainingsdaten.
    Nützlich für Tests ohne Scraping oder Kaggle-Account.
    """
    rng = np.random.default_rng(42)
    rows = []

    # Jedes Team bekommt ein verstecktes "Skill"-Level
    skill = {t: rng.uniform(0.35, 0.75) for t in TEAMS}

    start = pd.Timestamp("2022-01-01")
    end   = pd.Timestamp("2025-01-01")
    date_range = (end - start).days

    for i in range(n_matches):
        ta, tb = rng.choice(TEAMS, size=2, replace=False)
        pa = skill[ta]
        pb = skill[tb]
        # Gewinnwahrscheinlichkeit proportional zum Skill-Verhältnis
        p_a_wins = pa / (pa + pb)

        won_a    = rng.random() < p_a_wins
        score_a  = rng.integers(1, 4) if won_a else rng.integers(0, 3)
        score_b  = rng.integers(0, score_a) if won_a else rng.integers(score_a + 1, 4)
        score_a  = int(score_a)
        score_b  = int(min(score_b, 3))

        date = start + pd.Timedelta(days=int(rng.integers(0, date_range)))
        event = rng.choice(EVENTS)

        rows.append({
            "match_id": f"demo_{i:05d}",
            "date":     date,
            "team_a":   ta,
            "team_b":   tb,
            "score_a":  score_a,
            "score_b":  score_b,
            "winner":   ta if won_a else tb,
            "event":    event,
            "maps":     score_a + score_b,
        })

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df.to_csv(RAW_CSV, index=False)
    log.info(f"Demo-Daten generiert: {len(df)} Matches → {RAW_CSV}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Kaggle Loader / Demo-Daten")
    parser.add_argument("--kaggle", action="store_true", help="Von Kaggle herunterladen")
    parser.add_argument("--csv",    type=str,            help="Lokales CSV normalisieren")
    parser.add_argument("--demo",   action="store_true", help="Demo-Daten generieren")
    parser.add_argument("--n",      type=int, default=2000, help="Anzahl Demo-Matches")
    args = parser.parse_args()

    if args.kaggle:
        path = download_kaggle()
        df   = load_custom_csv(path)
    elif args.csv:
        df = load_custom_csv(args.csv)
    elif args.demo:
        df = generate_demo_data(args.n)
    else:
        print("Nutze --demo, --csv <pfad> oder --kaggle")
        sys.exit(0)

    print(df.tail(5).to_string())
    print(f"\nTeams: {df['team_a'].nunique()} | Matches: {len(df)}")