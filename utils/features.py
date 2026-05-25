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
    """Stellt sicher dass alle FEATURES-Spalten existieren (mit neutralen Defaults)."""
    # Map-Winrate-Diffs
    for m in CS2_MAPS:
        if f"{m}_winrate_diff" not in df.columns:
            df[f"{m}_winrate_diff"] = 0.0

    # Alle FEATURES mit neutralem Default auffüllen
    neutral = {
        "team1_past3": 0.5, "team2_past3": 0.5,
        "team1_h2h_pct": 0.5, "team2_h2h_pct": 0.5,
        "team1_overall_winrate": 0.5, "team2_overall_winrate": 0.5,
        "team1_lan_winrate": 0.5, "team2_lan_winrate": 0.5,
        "winrate_diff": 0.0, "lan_winrate_diff": 0.0,
        "elo_diff": 0.0, "rating_diff": 0.0,
        "adr_diff": 0.0, "kast_diff": 0.0,
        "kpr_diff": 0.0, "dpr_diff": 0.0,
        "star_player_advantage": 0.0, "weakest_link_advantage": 0.0,
        "consistency_advantage": 0.0,
        "team1_rating_std": 0.0, "team2_rating_std": 0.0,
    }
    for col, default in neutral.items():
        if col not in df.columns:
            log.debug(f"Spalte '{col}' fehlt — fülle mit {default}")
            df[col] = default

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Haupt-Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, save: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Bereitet den Feature-DataFrame vor.
    Unterstützt zwei Datenquellen automatisch:
      - Kaggle-Daten: haben bereits rating_diff, adr_diff usw. vorberechnet
      - PandaScore/GRID-Daten: haben nur team_a, team_b, winner, date
        → fehlende Features werden aus dem Match-Verlauf berechnet
    """
    log.info("Starte Feature Engineering ...")
    df = df.sort_values("date").copy()
    if pd.api.types.is_datetime64tz_dtype(df["date"]):
        df["date"] = df["date"].dt.tz_convert(None)

    # Label sicherstellen
    if "team_a_won" not in df.columns:
        df["team_a_won"] = (df["winner"] == df["team_a"]).astype(int)

    # 1. ELO berechnen
    log.info("  1/4 ELO berechnen ...")
    df, elo_ratings = compute_elo(df)

    # 2. Dynamische Features aus Match-Verlauf berechnen
    #    (für PandaScore-Matches die keine vorberechneten Stats haben)
    log.info("  2/4 Dynamische Features aus Match-Verlauf ...")
    df = _compute_rolling_features(df)

    # 3. Spalten absichern
    df = _ensure_columns(df)

    # 4. Matches ohne Label entfernen, fehlende Werte füllen
    log.info("  3/4 Fehlende Werte auffüllen ...")
    df = df.dropna(subset=["team_a_won", "date", "team_a", "team_b"])

    # Für jedes Feature: fehlende Werte mit Median aus Kaggle-Matches füllen
    # (Kaggle-Matches haben rating_diff etc., PandaScore-Matches nicht)
    for col in FEATURES:
        if col in df.columns and df[col].isna().any():
            median = df[col].median()
            df[col] = df[col].fillna(0 if np.isnan(median) else median)

    df = df.reset_index(drop=True)

    # Statistik: wie viele Matches kommen von welcher Quelle
    if "source" in df.columns:
        src_counts = df["source"].value_counts().to_dict()
        log.info(f"  4/4 Quellen: {src_counts}")
    else:
        log.info(f"  4/4 Feature Engineering fertig: {len(df)} Matches")

    log.info(f"Gesamt: {len(df)} Matches mit {len(FEATURES)} Features")

    if save:
        df.to_csv(FEATURES_CSV, index=False)
        log.info(f"Gespeichert: {FEATURES_CSV}")

    return df, elo_ratings


def _compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet Winrate, Form, H2H und Past3 dynamisch aus dem Match-Verlauf.
    Füllt nur Zeilen wo diese Werte noch fehlen (NaN) — überschreibt
    keine vorhandenen Kaggle-Stats.
    """
    df = df.sort_values("date").copy()

    # Nur berechnen wenn Spalten fehlen oder leer sind
    needs_winrate = "team1_overall_winrate" not in df.columns or df["team1_overall_winrate"].isna().all()
    needs_past3   = "team1_past3" not in df.columns or df["team1_past3"].isna().all()
    needs_h2h     = "team1_h2h_pct" not in df.columns or df["team1_h2h_pct"].isna().all()

    if not (needs_winrate or needs_past3 or needs_h2h):
        # Kaggle-Daten: alles schon vorhanden, nur NaN-Lücken füllen
        _fill_missing_rolling(df)
        return df

    log.info("    Berechne Rolling-Features aus Match-Verlauf (PandaScore-Matches) ...")

    wr_a_list, wr_b_list = [], []
    p3_a_list, p3_b_list = [], []
    h2h_list             = []

    for i, row in df.iterrows():
        d  = row["date"]
        ta = row["team_a"]
        tb = row["team_b"]
        past = df[df["date"] < d]

        # Winrate letzte 30 Tage
        cutoff_30 = d - pd.Timedelta(days=30)
        def winrate(team, since):
            m = past[(past["date"] >= since) &
                     ((past["team_a"] == team) | (past["team_b"] == team))]
            if len(m) < 3: return np.nan
            return ((m["winner"] == team).sum() / len(m))

        wr_a_list.append(winrate(ta, cutoff_30))
        wr_b_list.append(winrate(tb, cutoff_30))

        # Past3
        def past3(team):
            m = past[(past["team_a"] == team) | (past["team_b"] == team)].tail(3)
            if len(m) < 2: return np.nan
            return (m["winner"] == team).sum() / len(m)

        p3_a_list.append(past3(ta))
        p3_b_list.append(past3(tb))

        # H2H letzte 2 Jahre
        cutoff_h2h = d - pd.Timedelta(days=730)
        h2h = past[(past["date"] >= cutoff_h2h) &
                   (((past["team_a"] == ta) & (past["team_b"] == tb)) |
                    ((past["team_a"] == tb) & (past["team_b"] == ta)))]
        if len(h2h) >= 2:
            h2h_list.append((h2h["winner"] == ta).sum() / len(h2h))
        else:
            h2h_list.append(np.nan)

    # Nur leere Zellen füllen (Kaggle-Werte nicht überschreiben)
    def _fill_col(col, values):
        if col not in df.columns:
            df[col] = values
        else:
            df[col] = df[col].fillna(pd.Series(values, index=df.index))

    _fill_col("team1_overall_winrate", wr_a_list)
    _fill_col("team2_overall_winrate", wr_b_list)
    _fill_col("winrate_diff",
              [a - b if not (np.isnan(a) or np.isnan(b)) else np.nan
               for a, b in zip(wr_a_list, wr_b_list)])
    _fill_col("team1_past3", p3_a_list)
    _fill_col("team2_past3", p3_b_list)
    _fill_col("team1_h2h_pct", h2h_list)
    _fill_col("team2_h2h_pct",
              [1 - h if not np.isnan(h) else np.nan for h in h2h_list])

    # LAN-Winrate: nutze overall als Proxy wenn nicht vorhanden
    if "team1_lan_winrate" not in df.columns or df["team1_lan_winrate"].isna().all():
        df["team1_lan_winrate"] = df.get("team1_overall_winrate", np.nan)
        df["team2_lan_winrate"] = df.get("team2_overall_winrate", np.nan)
        df["lan_winrate_diff"]  = df["team1_lan_winrate"] - df["team2_lan_winrate"]

    return df


def _fill_missing_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Füllt nur die NaN-Lücken in Kaggle-Daten (z.B. neue PandaScore-Matches)."""
    # Einfach: Median der vorhandenen Werte nutzen
    for col in ["team1_overall_winrate", "team2_overall_winrate",
                "team1_past3", "team2_past3", "team1_h2h_pct"]:
        if col in df.columns:
            m = df[col].median()
            df[col] = df[col].fillna(m if not np.isnan(m) else 0.5)
    return df


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

    # H2H — Timezone-sicherer Vergleich
    cutoff = pd.Timestamp.now().tz_localize(None) - pd.Timedelta(days=H2H_WINDOW_DAYS)
    hist_dates = df_hist["date"].dt.tz_convert(None) if pd.api.types.is_datetime64tz_dtype(df_hist["date"]) else df_hist["date"]
    h2h_mask = (
        (hist_dates >= cutoff) &
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