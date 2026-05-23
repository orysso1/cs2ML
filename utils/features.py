# utils/features.py
# Feature Engineering für das CS2-Prediction-Modell.
# Berechnet: ELO, Winrate, Form, Head-to-Head, Ranking-Diff, Map-Winrate, ...

import logging
import sys
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ELO_START, ELO_K, FORM_WINDOW, WINRATE_WINDOW,
    H2H_WINDOW_DAYS, MIN_MATCHES, FEATURES_CSV
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. ELO-Rating
# ─────────────────────────────────────────────────────────────────────────────

def compute_elo(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Berechnet für jedes Match das ELO-Rating der Teams *vor* dem Match
    (wichtig: kein Data-Leakage) und gibt die aktuellen Ratings zurück.

    Returns:
        df:          DataFrame mit Spalten elo_a_pre, elo_b_pre
        elo_ratings: Dict mit aktuellen ELO-Werten aller Teams
    """
    df = df.sort_values("date").copy()
    elo: dict[str, float] = {}

    elo_a_pre, elo_b_pre = [], []

    for _, row in df.iterrows():
        ta, tb = row["team_a"], row["team_b"]
        ra = elo.get(ta, ELO_START)
        rb = elo.get(tb, ELO_START)

        elo_a_pre.append(ra)
        elo_b_pre.append(rb)

        # Erwartete Gewinnwahrscheinlichkeit
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        eb = 1 - ea

        won_a = 1 if row["winner"] == ta else 0
        elo[ta] = ra + ELO_K * (won_a - ea)
        elo[tb] = rb + ELO_K * ((1 - won_a) - eb)

    df["elo_a_pre"] = elo_a_pre
    df["elo_b_pre"] = elo_b_pre
    df["elo_diff"]  = df["elo_a_pre"] - df["elo_b_pre"]

    return df, elo


# ─────────────────────────────────────────────────────────────────────────────
# 2. Rolling Winrate
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_winrate(df: pd.DataFrame, team: str, before_date: pd.Timestamp,
                     days: int) -> float:
    """Winrate eines Teams in den letzten `days` Tagen vor `before_date`."""
    cutoff = before_date - timedelta(days=days)
    mask = (
        (df["date"] >= cutoff) &
        (df["date"] < before_date) &
        ((df["team_a"] == team) | (df["team_b"] == team))
    )
    sub = df.loc[mask]
    if len(sub) < MIN_MATCHES:
        return 0.5  # Neutral wenn zu wenig Daten
    wins = (sub["winner"] == team).sum()
    return wins / len(sub)


def compute_winrates(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt winrate_30d_a, winrate_30d_b und _diff hinzu."""
    df = df.sort_values("date").copy()
    wr_a, wr_b = [], []

    for _, row in df.iterrows():
        wr_a.append(_rolling_winrate(df, row["team_a"], row["date"], WINRATE_WINDOW))
        wr_b.append(_rolling_winrate(df, row["team_b"], row["date"], WINRATE_WINDOW))

    df["winrate_30d_a"]    = wr_a
    df["winrate_30d_b"]    = wr_b
    df["winrate_30d_diff"] = df["winrate_30d_a"] - df["winrate_30d_b"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Form (letzte N Matches)
# ─────────────────────────────────────────────────────────────────────────────

def _recent_form(df: pd.DataFrame, team: str, before_date: pd.Timestamp,
                 n: int) -> float:
    """Winrate der letzten N Matches eines Teams."""
    mask = (
        (df["date"] < before_date) &
        ((df["team_a"] == team) | (df["team_b"] == team))
    )
    sub = df.loc[mask].tail(n)
    if len(sub) == 0:
        return 0.5
    wins = (sub["winner"] == team).sum()
    return wins / len(sub)


def compute_form(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt form_a, form_b und form_diff (letzte 10 Matches) hinzu."""
    df = df.sort_values("date").copy()
    fa, fb = [], []

    for _, row in df.iterrows():
        fa.append(_recent_form(df, row["team_a"], row["date"], FORM_WINDOW))
        fb.append(_recent_form(df, row["team_b"], row["date"], FORM_WINDOW))

    df["form_a"]    = fa
    df["form_b"]    = fb
    df["form_diff"] = df["form_a"] - df["form_b"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Head-to-Head
# ─────────────────────────────────────────────────────────────────────────────

def compute_h2h(df: pd.DataFrame) -> pd.DataFrame:
    """H2H-Winrate von Team A gegen Team B (letzten 2 Jahre)."""
    df = df.sort_values("date").copy()
    h2h = []

    for _, row in df.iterrows():
        cutoff = row["date"] - timedelta(days=H2H_WINDOW_DAYS)
        ta, tb = row["team_a"], row["team_b"]
        mask = (
            (df["date"] >= cutoff) &
            (df["date"] < row["date"]) &
            (
                ((df["team_a"] == ta) & (df["team_b"] == tb)) |
                ((df["team_a"] == tb) & (df["team_b"] == ta))
            )
        )
        sub = df.loc[mask]
        if len(sub) == 0:
            h2h.append(0.5)
        else:
            wins_a = (sub["winner"] == ta).sum()
            h2h.append(wins_a / len(sub))

    df["h2h_winrate_a"] = h2h
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Tage seit letztem Match (Frische-Indikator)
# ─────────────────────────────────────────────────────────────────────────────

def compute_days_since_last(df: pd.DataFrame) -> pd.DataFrame:
    """Tage seit dem letzten Match (Ermüdung / Frische)."""
    df = df.sort_values("date").copy()
    last_a, last_b = [], []

    for _, row in df.iterrows():
        d = row["date"]
        ta, tb = row["team_a"], row["team_b"]

        prev_a = df.loc[(df["date"] < d) & ((df["team_a"] == ta) | (df["team_b"] == ta)), "date"]
        prev_b = df.loc[(df["date"] < d) & ((df["team_a"] == tb) | (df["team_b"] == tb)), "date"]

        last_a.append((d - prev_a.max()).days if len(prev_a) > 0 else 14)
        last_b.append((d - prev_b.max()).days if len(prev_b) > 0 else 14)

    df["days_since_last_a"]    = last_a
    df["days_since_last_b"]    = last_b
    df["days_since_last_diff"] = df["days_since_last_a"] - df["days_since_last_b"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Map-Winrate (wenn vorhanden)
# ─────────────────────────────────────────────────────────────────────────────

def compute_map_winrate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Allgemeine Map-Winrate pro Team (falls keine map-spezifische Spalte existiert,
    nutzen wir die generelle Winrate als Proxy).
    """
    if "map" in df.columns:
        # Map-spezifische Winrate berechnen
        for team_col in ["team_a", "team_b"]:
            col_name = f"map_winrate_{team_col[-1]}"
            rates = []
            for _, row in df.iterrows():
                mask = (
                    (df["date"] < row["date"]) &
                    (df.get("map", "") == row.get("map", "")) &
                    ((df["team_a"] == row[team_col]) | (df["team_b"] == row[team_col]))
                )
                sub = df.loc[mask]
                if len(sub) < 3:
                    rates.append(0.5)
                else:
                    wins = (sub["winner"] == row[team_col]).sum()
                    rates.append(wins / len(sub))
            df[col_name] = rates
    else:
        # Fallback: allgemeine Winrate als Map-Proxy
        df["map_winrate_a"] = df["winrate_30d_a"]
        df["map_winrate_b"] = df["winrate_30d_b"]

    df["map_winrate_diff"] = df["map_winrate_a"] - df["map_winrate_b"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. Ranking-Differenz
# ─────────────────────────────────────────────────────────────────────────────

def compute_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Schätzt Rankings dynamisch aus ELO (falls kein echtes Ranking vorhanden).
    Echte Rankings können aus HLTV-Daten ergänzt werden.
    """
    if "rank_a" in df.columns and "rank_b" in df.columns:
        df["ranking_diff"] = df["rank_b"] - df["rank_a"]  # Negativ = A besser
    else:
        # ELO als Ranking-Proxy (höheres ELO → niedrigeres Ranking → besser)
        df["ranking_diff"] = (df["elo_b_pre"] - df["elo_a_pre"]) / 100
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 8. Lineup-Alter (Roster-Stabilität)
# ─────────────────────────────────────────────────────────────────────────────

def compute_lineup_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Proxy für Roster-Stabilität: wie lange spielen die Teams schon zusammen?
    Hier: Tage seit dem ersten gemeinsamen Match in den Daten.
    """
    first_seen: dict[str, pd.Timestamp] = {}
    ages_a, ages_b = [], []

    for _, row in df.iterrows():
        d = row["date"]
        ta, tb = row["team_a"], row["team_b"]

        if ta not in first_seen:
            first_seen[ta] = d
        if tb not in first_seen:
            first_seen[tb] = d

        ages_a.append((d - first_seen[ta]).days)
        ages_b.append((d - first_seen[tb]).days)

    df["lineup_age_a"]    = ages_a
    df["lineup_age_b"]    = ages_b
    df["lineup_age_diff"] = df["lineup_age_a"] - df["lineup_age_b"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 9. Label
# ─────────────────────────────────────────────────────────────────────────────

def add_label(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt Zielvariable hinzu: 1 = Team A gewinnt, 0 = Team B gewinnt."""
    df["team_a_won"] = (df["winner"] == df["team_a"]).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline: Alles auf einmal
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, save: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Führt alle Feature-Engineering-Schritte durch.

    Args:
        df:   Rohdaten (aus scraper oder kaggle_loader)
        save: Ergebnis als CSV speichern

    Returns:
        df_feat:     DataFrame mit allen Features
        elo_ratings: Aktuelle ELO-Ratings (für Predictions auf neuen Matches)
    """
    log.info("Starte Feature Engineering ...")
    n = len(df)

    log.info("  1/7 ELO berechnen ...")
    df, elo_ratings = compute_elo(df)

    log.info("  2/7 Winrates berechnen ...")
    df = compute_winrates(df)

    log.info("  3/7 Form berechnen ...")
    df = compute_form(df)

    log.info("  4/7 Head-to-Head berechnen ...")
    df = compute_h2h(df)

    log.info("  5/7 Tage seit letztem Match ...")
    df = compute_days_since_last(df)

    log.info("  6/7 Map-Winrate & Ranking ...")
    df = compute_map_winrate(df)
    df = compute_ranking(df)
    df = compute_lineup_age(df)

    log.info("  7/7 Label hinzufügen ...")
    df = add_label(df)

    # Nur Matches mit allen Features behalten
    from config import FEATURES
    feat_cols = FEATURES + ["team_a_won", "date", "team_a", "team_b", "winner", "event"]
    df_feat = df[[c for c in feat_cols if c in df.columns]].dropna(subset=FEATURES)

    log.info(f"Feature Engineering abgeschlossen: {n} → {len(df_feat)} Matches")

    if save:
        df_feat.to_csv(FEATURES_CSV, index=False)
        log.info(f"Features gespeichert: {FEATURES_CSV}")

    return df_feat, elo_ratings


# ─────────────────────────────────────────────────────────────────────────────
# Features für ein einzelnes neues Match berechnen
# ─────────────────────────────────────────────────────────────────────────────

def build_prediction_features(
    team_a: str,
    team_b: str,
    df_hist: pd.DataFrame,
    elo_ratings: dict,
    pred_date: pd.Timestamp | None = None,
) -> dict:
    """
    Berechnet Features für ein noch nicht gespieltes Match.
    Nutzt historische Daten + aktuelle ELO-Ratings.
    """
    if pred_date is None:
        pred_date = pd.Timestamp.now()

    ra = elo_ratings.get(team_a, ELO_START)
    rb = elo_ratings.get(team_b, ELO_START)

    return {
        "elo_diff":             ra - rb,
        "winrate_30d_diff":     _rolling_winrate(df_hist, team_a, pred_date, WINRATE_WINDOW)
                               - _rolling_winrate(df_hist, team_b, pred_date, WINRATE_WINDOW),
        "form_diff":            _recent_form(df_hist, team_a, pred_date, FORM_WINDOW)
                               - _recent_form(df_hist, team_b, pred_date, FORM_WINDOW),
        "h2h_winrate_a":        _h2h(df_hist, team_a, team_b, pred_date),
        "ranking_diff":         (rb - ra) / 100,
        "map_winrate_diff":     _rolling_winrate(df_hist, team_a, pred_date, WINRATE_WINDOW)
                               - _rolling_winrate(df_hist, team_b, pred_date, WINRATE_WINDOW),
        "days_since_last_diff": _days_since(df_hist, team_a, pred_date)
                               - _days_since(df_hist, team_b, pred_date),
        "lineup_age_diff":      0.0,  # Kann manuell gesetzt werden
    }


def _h2h(df: pd.DataFrame, ta: str, tb: str, before: pd.Timestamp) -> float:
    cutoff = before - timedelta(days=H2H_WINDOW_DAYS)
    mask = (
        (df["date"] >= cutoff) & (df["date"] < before) &
        (((df["team_a"] == ta) & (df["team_b"] == tb)) |
         ((df["team_a"] == tb) & (df["team_b"] == ta)))
    )
    sub = df.loc[mask]
    if len(sub) == 0:
        return 0.5
    return (sub["winner"] == ta).sum() / len(sub)


def _days_since(df: pd.DataFrame, team: str, before: pd.Timestamp) -> int:
    prev = df.loc[(df["date"] < before) &
                  ((df["team_a"] == team) | (df["team_b"] == team)), "date"]
    return (before - prev.max()).days if len(prev) > 0 else 14


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Feature Engineering")
    parser.add_argument("--input", type=str, default=str(Path(__file__).parent.parent / "data" / "matches_raw.csv"))
    args = parser.parse_args()

    df_raw = pd.read_csv(args.input, parse_dates=["date"])
    df_feat, elo = build_features(df_raw)
    print(df_feat[["date", "team_a", "team_b", "elo_diff", "form_diff", "team_a_won"]].tail(10).to_string())
    print(f"\nTop-5 ELO: {sorted(elo.items(), key=lambda x: -x[1])[:5]}")