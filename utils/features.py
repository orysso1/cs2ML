# utils/features.py
# Feature Engineering für CS2 Prediction.
# Nutzt die reichhaltigen Kaggle-Spalten direkt +
# berechnet dynamisches ELO on top.

import logging
import sys
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ELO_START, ELO_K, FEATURES, CS2_MAPS, FEATURES_CSV,
    MIN_MATCHES, WINRATE_WINDOW, FORM_WINDOW, H2H_WINDOW_DAYS
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ELO (einzige dynamisch berechnete Feature — alles andere kommt direkt aus dem CSV)
# ─────────────────────────────────────────────────────────────────────────────

def compute_elo(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Berechnet ELO *vor* jedem Match (kein Leakage).
    Gibt df mit elo_diff-Spalte + aktuelles Ratings-Dict zurück.
    """
    df = df.sort_values("date").copy()
    elo: dict[str, float] = {}
    elo_diffs = []

    for _, row in df.iterrows():
        ta, tb = row["team_a"], row["team_b"]
        ra = elo.get(ta, ELO_START)
        rb = elo.get(tb, ELO_START)
        elo_diffs.append(ra - rb)

        ea   = 1 / (1 + 10 ** ((rb - ra) / 400))
        won  = 1 if row["winner"] == ta else 0
        elo[ta] = ra + ELO_K * (won - ea)
        elo[tb] = rb + ELO_K * ((1 - won) - (1 - ea))

    df["elo_diff"] = elo_diffs
    return df, elo


# ─────────────────────────────────────────────────────────────────────────────
# Fehlende Spalten mit neutralen Defaults auffüllen
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Stellt sicher dass alle FEATURES-Spalten existieren (mit 0-Default)."""
    all_needed = set(FEATURES)
    for m in CS2_MAPS:
        all_needed.add(f"{m}_winrate_diff")

    for col in all_needed:
        if col not in df.columns:
            log.debug(f"Spalte '{col}' fehlt — fülle mit 0")
            df[col] = 0.0

    # Spezifische Defaults wo 0 irreführend wäre
    if "winner_head2head_percentage" not in df.columns:
        df["winner_head2head_percentage"] = 0.5
    if "loser_head2head_percentage" not in df.columns:
        df["loser_head2head_percentage"]  = 0.5
    if "h2h_diff" not in df.columns:
        df["h2h_diff"] = 0.0
    if "past3_diff" not in df.columns:
        df["past3_diff"] = 0.0
    if "winner_past3" not in df.columns:
        df["winner_past3"] = 0.5
    if "loser_past3" not in df.columns:
        df["loser_past3"]  = 0.5

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Haupt-Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, save: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Bereitet den Feature-DataFrame vor.
    Da das Kaggle-CSV bereits vorberechnete Stats enthält,
    müssen wir hauptsächlich ELO ergänzen und Spalten absichern.
    """
    log.info("Starte Feature Engineering ...")
    df = df.sort_values("date").copy()

    # 1. ELO berechnen (einzige dynamische Berechnung)
    log.info("  ELO berechnen ...")
    df, elo_ratings = compute_elo(df)

    # 2. Sicherstellen dass alle Feature-Spalten vorhanden sind
    df = _ensure_columns(df)

    # 3. Label sicherstellen
    if "team_a_won" not in df.columns:
        df["team_a_won"] = (df["winner"] == df["team_a"]).astype(int)

    # 4. Auf sinnvolle Matches beschränken (keine NaN in Kern-Features)
    core = ["rating_diff", "team_a_won", "date", "team_a", "team_b"]
    df_feat = df.dropna(subset=core).reset_index(drop=True)

    # 5. Fehlende numerische Werte mit Median auffüllen
    for col in FEATURES:
        if col in df_feat.columns:
            median = df_feat[col].median()
            df_feat[col] = df_feat[col].fillna(median if not np.isnan(median) else 0)

    log.info(f"Feature Engineering fertig: {len(df_feat)} Matches")

    if save:
        df_feat.to_csv(FEATURES_CSV, index=False)
        log.info(f"Gespeichert: {FEATURES_CSV}")

    return df_feat, elo_ratings


# ─────────────────────────────────────────────────────────────────────────────
# Features für ein zukünftiges Match
# ─────────────────────────────────────────────────────────────────────────────

def build_prediction_features(
    team_a: str,
    team_b: str,
    df_hist: pd.DataFrame,
    elo_ratings: dict,
) -> dict:
    """
    Berechnet Features für ein noch nicht gespieltes Match.
    Aggregiert die letzten bekannten Stats der beiden Teams.
    """
    feats = {}

    ra = elo_ratings.get(team_a, ELO_START)
    rb = elo_ratings.get(team_b, ELO_START)
    feats["elo_diff"] = ra - rb

    # Letzte Matches von Team A und Team B
    def last_matches(team, n=10):
        mask = (df_hist["team_a"] == team) | (df_hist["team_b"] == team)
        return df_hist.loc[mask].sort_values("date").tail(n)

    last_a = last_matches(team_a)
    last_b = last_matches(team_b)

    # Rating-Stats aus letzten Matches
    def avg_stat(matches, team, col_winner, col_loser):
        """Holt den Wert aus winner_* oder loser_* je nachdem ob das Team gewann."""
        vals = []
        for _, row in matches.iterrows():
            won = row.get("winner", "") == team
            col = col_winner if won else col_loser
            v = row.get(col, np.nan)
            if not np.isnan(float(v)) if v is not None else False:
                vals.append(float(v))
        return np.mean(vals) if vals else 0.0

    # Winrates direkt aus letzten bekannten Zeilen
    def latest_col(matches, col, fallback=0.5):
        if col in matches.columns:
            v = matches[col].dropna()
            return float(v.iloc[-1]) if len(v) > 0 else fallback
        return fallback

    # Winrate
    feats["team1_overall_winrate"] = latest_col(last_a[last_a["team_a"] == team_a], "team1_overall_winrate", 0.5)
    feats["team2_overall_winrate"] = latest_col(last_b[last_b["team_a"] == team_b], "team1_overall_winrate", 0.5)
    feats["winrate_diff"]          = feats["team1_overall_winrate"] - feats["team2_overall_winrate"]

    feats["team1_lan_winrate"] = latest_col(last_a[last_a["team_a"] == team_a], "team1_lan_winrate", 0.5)
    feats["team2_lan_winrate"] = latest_col(last_b[last_b["team_a"] == team_b], "team1_lan_winrate", 0.5)
    feats["lan_winrate_diff"]  = feats["team1_lan_winrate"] - feats["team2_lan_winrate"]

    # Rating-Diffs
    for col in ["rating_diff", "adr_diff", "kast_diff", "kpr_diff", "dpr_diff"]:
        vals_a = last_a[last_a["team_a"] == team_a][col].dropna() if col in last_a.columns else pd.Series()
        feats[col] = float(vals_a.mean()) if len(vals_a) > 0 else 0.0

    # H2H
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=H2H_WINDOW_DAYS)
    h2h_mask = (
        (df_hist["date"] >= cutoff) &
        (((df_hist["team_a"] == team_a) & (df_hist["team_b"] == team_b)) |
         ((df_hist["team_a"] == team_b) & (df_hist["team_b"] == team_a)))
    )
    h2h_matches = df_hist.loc[h2h_mask]
    if len(h2h_matches) > 0:
        a_wins = (h2h_matches["winner"] == team_a).sum()
        feats["winner_head2head_percentage"] = a_wins / len(h2h_matches)
        feats["loser_head2head_percentage"]  = 1 - feats["winner_head2head_percentage"]
        feats["h2h_diff"] = feats["winner_head2head_percentage"] - feats["loser_head2head_percentage"]
    else:
        feats["winner_head2head_percentage"] = 0.5
        feats["loser_head2head_percentage"]  = 0.5
        feats["h2h_diff"] = 0.0

    # Past3
    def past3(team):
        m = last_matches(team, 3)
        if len(m) == 0:
            return 0.5
        return (m["winner"] == team).sum() / len(m)

    feats["winner_past3"] = past3(team_a)
    feats["loser_past3"]  = past3(team_b)
    feats["past3_diff"]   = feats["winner_past3"] - feats["loser_past3"]

    # Consistency & player advantage
    for col in ["star_player_advantage", "weakest_link_advantage", "consistency_advantage",
                "team1_rating_std", "team2_rating_std"]:
        vals = last_a[last_a["team_a"] == team_a][col].dropna() if col in last_a.columns else pd.Series()
        feats[col] = float(vals.mean()) if len(vals) > 0 else 0.0

    # Map-Winrate-Diffs
    for m in CS2_MAPS:
        col = f"{m}_winrate_diff"
        vals = last_a[last_a["team_a"] == team_a][col].dropna() if col in last_a.columns else pd.Series()
        feats[col] = float(vals.mean()) if len(vals) > 0 else 0.0

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(Path(__file__).parent.parent / "data" / "matches_raw.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.input, parse_dates=["date"])
    df_feat, elo = build_features(df)
    print(df_feat[["date", "team_a", "team_b", "elo_diff", "rating_diff", "team_a_won"]].tail(10))
    print(f"\nTop-5 ELO: {sorted(elo.items(), key=lambda x: -x[1])[:5]}")