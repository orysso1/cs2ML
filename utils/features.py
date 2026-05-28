# utils/features.py
# Feature Engineering — vollständige Version mit allen Verbesserungen:
#   - Tier-gewichtetes ELO
#   - Recency-Gewichtung
#   - Spieler-Individualstats
#   - Qualitätsbereinigte Winrate
#   - Momentum-Feature
#   - Walk-Forward-Validation-Splits
#   - Dynamische Berechnung für PandaScore-Matches

import logging
import sys
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ELO_START, ELO_K, ELO_K_MAJOR, ELO_K_TIER1, ELO_K_TIER2, ELO_K_ONLINE,
    TIER_KEYWORDS, RECENCY_HALFLIFE_DAYS,
    FEATURES, CS2_MAPS, FEATURES_CSV,
    MIN_MATCHES, WINRATE_WINDOW, FORM_WINDOW, H2H_WINDOW_DAYS
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tier-Erkennung
# ─────────────────────────────────────────────────────────────────────────────

def event_to_tier(event: str) -> tuple[int, int]:
    """
    Gibt (tier_value, elo_k) für einen Event-Namen zurück.
    tier_value: 3=Major, 2=Tier1, 1=Tier2, 0=Online
    """
    if not isinstance(event, str):
        return 1, ELO_K
    event_lower = event.lower()
    for keyword, k in TIER_KEYWORDS.items():
        if keyword in event_lower:
            if k == ELO_K_MAJOR:  return 3, k
            if k == ELO_K_TIER1:  return 2, k
            if k == ELO_K_ONLINE: return 0, k
            return 1, k
    return 1, ELO_K


# ─────────────────────────────────────────────────────────────────────────────
# ELO — Tier-gewichtet
# ─────────────────────────────────────────────────────────────────────────────

def compute_elo(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Tier-gewichtetes ELO:
    - Major-Matches bewegen ELO stärker als Online-Matches
    - ELO wird VOR dem Match gespeichert (kein Leakage)
    """
    df = df.sort_values("date").copy()
    elo: dict[str, float] = {}
    elo_diffs, tiers = [], []

    for _, row in df.iterrows():
        ta, tb = row["team_a"], row["team_b"]
        ra = elo.get(ta, ELO_START)
        rb = elo.get(tb, ELO_START)
        elo_diffs.append(ra - rb)

        event  = row.get("event", "") or row.get("tournament", "") or ""
        tier_v, k = event_to_tier(event)
        tiers.append(tier_v)

        ea   = 1 / (1 + 10 ** ((rb - ra) / 400))
        won  = 1 if row["winner"] == row["team_a"] else 0
        elo[ta] = ra + k * (won - ea)
        elo[tb] = rb + k * ((1 - won) - (1 - ea))

    df["elo_diff"]  = elo_diffs
    df["event_tier"] = tiers
    return df, elo


# ─────────────────────────────────────────────────────────────────────────────
# Recency-Gewichte
# ─────────────────────────────────────────────────────────────────────────────

def compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Exponentieller Abfall: neuere Matches bekommen höheres Gewicht.
    Nach RECENCY_HALFLIFE_DAYS hat ein Match noch 50% Gewicht.
    """
    max_date  = df["date"].max()
    days_old  = (max_date - df["date"]).dt.days.clip(lower=0)
    weights   = np.exp(-days_old * np.log(2) / RECENCY_HALFLIFE_DAYS)
    return weights.values


# ─────────────────────────────────────────────────────────────────────────────
# Momentum-Feature
# ─────────────────────────────────────────────────────────────────────────────

def _momentum(df: pd.DataFrame, team: str,
               before_date: pd.Timestamp, n: int = 5) -> float:
    """
    Momentum = gewichtete Siegesserie.
    Letzter Sieg zählt mehr als älterer.
    Rückgabe: -1 (5 Niederlagen) bis +1 (5 Siege)
    """
    mask = (df["date"] < before_date) & \
           ((df["team_a"] == team) | (df["team_b"] == team))
    sub  = df.loc[mask].tail(n)
    if len(sub) == 0:
        return 0.0
    wins    = (sub["winner"] == team).values.astype(float)
    weights = np.exp(np.arange(len(wins)) * 0.3)   # neuere gewichten mehr
    weights /= weights.sum()
    return float(np.dot(wins, weights) * 2 - 1)    # auf -1…+1 skalieren


# ─────────────────────────────────────────────────────────────────────────────
# Qualitätsbereinigte Winrate
# ─────────────────────────────────────────────────────────────────────────────

def _quality_winrate(df: pd.DataFrame, team: str,
                      before_date: pd.Timestamp,
                      elo_snapshot: dict) -> float:
    """
    Qualitätsbereinigte Winrate: Siege gegen starke Gegner zählen mehr.
    Gewicht = Gegner-ELO / 1500
    """
    cutoff = before_date - timedelta(days=WINRATE_WINDOW)
    mask   = (df["date"] >= cutoff) & (df["date"] < before_date) & \
             ((df["team_a"] == team) | (df["team_b"] == team))
    sub    = df.loc[mask]
    if len(sub) < MIN_MATCHES:
        return 0.5

    total_w, total_weight = 0.0, 0.0
    for _, row in sub.iterrows():
        opp    = row["team_b"] if row["team_a"] == team else row["team_a"]
        opp_elo = elo_snapshot.get(opp, ELO_START)
        w       = opp_elo / ELO_START
        won     = float(row["winner"] == team)
        total_w      += won * w
        total_weight += w

    return total_w / total_weight if total_weight > 0 else 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Spieler-Individualstats
# ─────────────────────────────────────────────────────────────────────────────

def compute_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet Spieler-Level-Features aus vorhandenen Spalten:
    - top_player_rating_diff: Bester Spieler Team A vs B
    - bot_player_rating_diff: Schwächster Spieler
    - rating_std_diff: Konsistenz (ausgeglichenes vs. Star-Team)
    """
    player_cols_a = [f"team1_player_{i}_RATING" for i in range(1, 6)]
    player_cols_b = [f"team2_player_{i}_RATING" for i in range(1, 6)]

    available_a = [c for c in player_cols_a if c in df.columns]
    available_b = [c for c in player_cols_b if c in df.columns]

    if len(available_a) >= 3 and len(available_b) >= 3:
        ratings_a = df[available_a].apply(pd.to_numeric, errors="coerce")
        ratings_b = df[available_b].apply(pd.to_numeric, errors="coerce")

        df["top_player_rating_diff"] = (ratings_a.max(axis=1) -
                                         ratings_b.max(axis=1))
        df["bot_player_rating_diff"] = (ratings_a.min(axis=1) -
                                         ratings_b.min(axis=1))
        df["rating_std_diff"]        = (ratings_a.std(axis=1) -
                                         ratings_b.std(axis=1))
    else:
        # Fallback aus aggregierten Spalten
        for col in ["top_player_rating_diff", "bot_player_rating_diff", "rating_std_diff"]:
            if col not in df.columns:
                df[col] = 0.0

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Rolling Features (für PandaScore-Matches ohne vorberechnete Stats)
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_winrate(df, team, before_date, days):
    cutoff = before_date - timedelta(days=days)
    mask   = (df["date"] >= cutoff) & (df["date"] < before_date) & \
             ((df["team_a"] == team) | (df["team_b"] == team))
    sub    = df.loc[mask]
    if len(sub) < MIN_MATCHES:
        return np.nan
    return (sub["winner"] == team).sum() / len(sub)


def _past3(df, team, before_date):
    mask = (df["date"] < before_date) & \
           ((df["team_a"] == team) | (df["team_b"] == team))
    sub  = df.loc[mask].tail(3)
    if len(sub) < 2:
        return np.nan
    return (sub["winner"] == team).sum() / len(sub)


def _h2h(df, ta, tb, before_date):
    cutoff = before_date - timedelta(days=H2H_WINDOW_DAYS)
    mask   = (df["date"] >= cutoff) & (df["date"] < before_date) & \
             (((df["team_a"] == ta) & (df["team_b"] == tb)) |
              ((df["team_a"] == tb) & (df["team_b"] == ta)))
    sub    = df.loc[mask]
    if len(sub) < 2:
        return np.nan
    return (sub["winner"] == ta).sum() / len(sub)


def _compute_rolling_features(df: pd.DataFrame,
                               elo_snapshot: dict) -> pd.DataFrame:
    """
    Berechnet alle Rolling-Features für Matches die noch keine haben.
    Überschreibt keine vorhandenen Kaggle-Werte.
    """
    df = df.sort_values("date").copy()

    needs = {
        "team1_overall_winrate": [],
        "team2_overall_winrate": [],
        "winrate_diff":          [],
        "team1_past3":           [],
        "team2_past3":           [],
        "team1_h2h_pct":         [],
        "team2_h2h_pct":         [],
        "momentum_a":            [],
        "momentum_b":            [],
        "momentum_diff":         [],
        "quality_winrate_diff":  [],
    }

    # ELO zu jedem Zeitpunkt snapshot — wir brauchen running ELO
    elo_running: dict[str, float] = {}

    for _, row in df.iterrows():
        d  = row["date"]
        ta = row["team_a"]
        tb = row["team_b"]

        # Winrate
        needs["team1_overall_winrate"].append(_rolling_winrate(df, ta, d, WINRATE_WINDOW))
        needs["team2_overall_winrate"].append(_rolling_winrate(df, tb, d, WINRATE_WINDOW))

        wr_a = needs["team1_overall_winrate"][-1]
        wr_b = needs["team2_overall_winrate"][-1]
        needs["winrate_diff"].append(
            wr_a - wr_b if not (np.isnan(wr_a) or np.isnan(wr_b)) else np.nan
        )

        # Past3
        needs["team1_past3"].append(_past3(df, ta, d))
        needs["team2_past3"].append(_past3(df, tb, d))

        # H2H
        h = _h2h(df, ta, tb, d)
        needs["team1_h2h_pct"].append(h)
        needs["team2_h2h_pct"].append(1 - h if not np.isnan(h) else np.nan)

        # Momentum
        ma = _momentum(df, ta, d)
        mb = _momentum(df, tb, d)
        needs["momentum_a"].append(ma)
        needs["momentum_b"].append(mb)
        needs["momentum_diff"].append(ma - mb)

        # Qualitätsbereinigte Winrate
        qa = _quality_winrate(df, ta, d, elo_running.copy())
        qb = _quality_winrate(df, tb, d, elo_running.copy())
        needs["quality_winrate_diff"].append(qa - qb)

        # ELO updaten
        ra = elo_running.get(ta, ELO_START)
        rb = elo_running.get(tb, ELO_START)
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        won = 1 if row["winner"] == row["team_a"] else 0
        _, k = event_to_tier(str(row.get("event", "")))
        elo_running[ta] = ra + k * (won - ea)
        elo_running[tb] = rb + k * ((1 - won) - (1 - ea))

    # Nur leere Zellen füllen
    for col, values in needs.items():
        series = pd.Series(values, index=df.index)
        if col not in df.columns:
            df[col] = series
        else:
            df[col] = df[col].fillna(series)

    # LAN-Winrate als Proxy wenn nicht vorhanden
    if "team1_lan_winrate" not in df.columns or df["team1_lan_winrate"].isna().all():
        df["team1_lan_winrate"] = df["team1_overall_winrate"]
        df["team2_lan_winrate"] = df["team2_overall_winrate"]
        df["lan_winrate_diff"]  = df["winrate_diff"]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Ensure Columns
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    neutral = {
        "team1_overall_winrate": 0.5, "team2_overall_winrate": 0.5,
        "team1_lan_winrate":     0.5, "team2_lan_winrate":     0.5,
        "team1_past3":           0.5, "team2_past3":           0.5,
        "team1_h2h_pct":         0.5, "team2_h2h_pct":         0.5,
        "winrate_diff":          0.0, "lan_winrate_diff":      0.0,
        "elo_diff":              0.0, "rating_diff":           0.0,
        "adr_diff":              0.0, "kast_diff":             0.0,
        "kpr_diff":              0.0, "dpr_diff":              0.0,
        "star_player_advantage": 0.0, "weakest_link_advantage":0.0,
        "consistency_advantage": 0.0,
        "team1_rating_std":      0.0, "team2_rating_std":      0.0,
        "top_player_rating_diff":0.0, "bot_player_rating_diff":0.0,
        "rating_std_diff":       0.0,
        "momentum_a":            0.0, "momentum_b":            0.0,
        "momentum_diff":         0.0, "quality_winrate_diff":  0.0,
        "is_lan":                0,   "event_tier":            1,
    }
    for m in CS2_MAPS:
        neutral[f"{m}_winrate_diff"] = 0.0

    for col, default in neutral.items():
        if col not in df.columns:
            df[col] = default

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Haupt-Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, save: bool = True) -> tuple[pd.DataFrame, dict]:
    """Vollständige Feature-Engineering-Pipeline."""
    log.info("Starte Feature Engineering ...")
    df = df.sort_values("date").copy()

    if pd.api.types.is_datetime64tz_dtype(df["date"]):
        df["date"] = df["date"].dt.tz_convert(None)

    # Label
    if "team_a_won" not in df.columns:
        df["team_a_won"] = (df["winner"] == df["team_a"]).astype(int)

    # 1. ELO (tier-gewichtet)
    log.info("  1/5 Tier-gewichtetes ELO ...")
    df, elo_ratings = compute_elo(df)

    # 2. Spieler-Individualstats
    log.info("  2/5 Spieler-Features ...")
    df = compute_player_features(df)

    # 3. Rolling Features (Winrate, Momentum, H2H, Quality)
    log.info("  3/5 Rolling Features (Winrate, Momentum, H2H, Quality) ...")
    df = _compute_rolling_features(df, elo_ratings)

    # 4. is_lan aus event_type
    if "event_type" in df.columns:
        df["is_lan"] = (df["event_type"].str.lower() == "lan").astype(int)
    elif "is_lan" not in df.columns:
        df["is_lan"] = 0

    # 5. Spalten absichern + fehlende Werte füllen
    log.info("  4/5 Spalten absichern ...")
    df = _ensure_columns(df)

    df = df.dropna(subset=["team_a_won", "date", "team_a", "team_b"])
    for col in FEATURES:
        if col in df.columns:
            med = df[col].median()
            df[col] = df[col].fillna(0 if np.isnan(med) else med)

    df = df.reset_index(drop=True)

    log.info(f"  5/5 Fertig: {len(df)} Matches, {len(FEATURES)} Features")
    if save:
        df.to_csv(FEATURES_CSV, index=False)
        log.info(f"Gespeichert: {FEATURES_CSV}")

    return df, elo_ratings


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward-Validation
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_splits(df: pd.DataFrame,
                         n_splits: int = 6,
                         min_train_months: int = 12) -> list[tuple]:
    """
    Erzeugt Walk-Forward-Splits für realistische Evaluation.
    Jeder Split: wachsendes Trainingsfenster, festes 2-Monats-Testfenster.

    Returns: Liste von (train_df, val_df, cutoff_date) Tuples
    """
    df = df.sort_values("date").copy()
    min_date = df["date"].min()
    max_date = df["date"].max()

    # Testfenster: letzte n_splits * 2 Monate
    test_window = pd.DateOffset(months=2)
    splits = []

    for i in range(n_splits, 0, -1):
        val_end   = max_date - pd.DateOffset(months=(i - 1) * 2)
        val_start = val_end - test_window
        train_end = val_start

        # Mindest-Trainingszeit
        if (train_end - min_date).days < min_train_months * 30:
            continue

        train = df[df["date"] < train_end]
        val   = df[(df["date"] >= val_start) & (df["date"] < val_end)]

        if len(train) < 100 or len(val) < 20:
            continue
        if val["team_a_won"].nunique() < 2:
            continue

        splits.append((train, val, str(val_start.date())))

    log.info(f"Walk-Forward: {len(splits)} Splits erstellt")
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# Features für zukünftiges Match
# ─────────────────────────────────────────────────────────────────────────────

def build_prediction_features(
    team_a: str, team_b: str,
    df_hist: pd.DataFrame,
    elo_ratings: dict,
) -> dict:
    """Features für ein noch nicht gespieltes Match."""
    if pd.api.types.is_datetime64tz_dtype(df_hist["date"]):
        df_hist = df_hist.copy()
        df_hist["date"] = df_hist["date"].dt.tz_convert(None)

    now = pd.Timestamp.now().tz_localize(None)
    ra  = elo_ratings.get(team_a, ELO_START)
    rb  = elo_ratings.get(team_b, ELO_START)

    feats = {"elo_diff": ra - rb}

    # Winrate
    wr_a = _rolling_winrate(df_hist, team_a, now, WINRATE_WINDOW)
    wr_b = _rolling_winrate(df_hist, team_b, now, WINRATE_WINDOW)
    feats["team1_overall_winrate"] = wr_a if not np.isnan(wr_a) else 0.5
    feats["team2_overall_winrate"] = wr_b if not np.isnan(wr_b) else 0.5
    feats["winrate_diff"]          = feats["team1_overall_winrate"] - feats["team2_overall_winrate"]
    feats["team1_lan_winrate"]     = feats["team1_overall_winrate"]
    feats["team2_lan_winrate"]     = feats["team2_overall_winrate"]
    feats["lan_winrate_diff"]      = feats["winrate_diff"]

    # Past3
    p3a = _past3(df_hist, team_a, now)
    p3b = _past3(df_hist, team_b, now)
    feats["team1_past3"] = p3a if not np.isnan(p3a) else 0.5
    feats["team2_past3"] = p3b if not np.isnan(p3b) else 0.5

    # H2H
    h = _h2h(df_hist, team_a, team_b, now)
    feats["team1_h2h_pct"] = h   if not np.isnan(h) else 0.5
    feats["team2_h2h_pct"] = 1-h if not np.isnan(h) else 0.5

    # Momentum
    feats["momentum_a"]    = _momentum(df_hist, team_a, now)
    feats["momentum_b"]    = _momentum(df_hist, team_b, now)
    feats["momentum_diff"] = feats["momentum_a"] - feats["momentum_b"]

    # Qualitätsbereinigte Winrate
    qa = _quality_winrate(df_hist, team_a, now, elo_ratings)
    qb = _quality_winrate(df_hist, team_b, now, elo_ratings)
    feats["quality_winrate_diff"] = qa - qb

    # Rating-Diffs aus letzten bekannten Matches
    for col in ["rating_diff", "adr_diff", "kast_diff", "kpr_diff", "dpr_diff",
                "star_player_advantage", "weakest_link_advantage",
                "team1_rating_std", "team2_rating_std", "consistency_advantage",
                "top_player_rating_diff", "bot_player_rating_diff", "rating_std_diff"]:
        last = df_hist[df_hist["team_a"] == team_a][col].dropna() if col in df_hist.columns else pd.Series()
        feats[col] = float(last.tail(5).mean()) if len(last) > 0 else 0.0

    # Map-Winrate-Diffs
    for m in CS2_MAPS:
        col = f"{m}_winrate_diff"
        last = df_hist[df_hist["team_a"] == team_a][col].dropna() if col in df_hist.columns else pd.Series()
        feats[col] = float(last.tail(5).mean()) if len(last) > 0 else 0.0

    feats["is_lan"]     = 1
    feats["event_tier"] = 2

    return feats