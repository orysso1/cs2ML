# data/kaggle_loader.py
# Lädt und normalisiert das Kaggle CS2-HLTV-Dataset auf das interne Schema.

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, RAW_CSV, CS2_MAPS

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Kaggle-CSV einlesen und normalisieren
# ─────────────────────────────────────────────────────────────────────────────

def load_custom_csv(path: str | Path) -> pd.DataFrame:
    """
    Lädt das Kaggle CS2-Dataset und normalisiert es auf internes Schema.
    Behält alle nützlichen Spalten (Spieler-Stats, Map-Winrates, H2H, ...).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV nicht gefunden: {path}")

    df = pd.read_csv(path, low_memory=False)
    log.info(f"Geladen: {path} ({len(df)} Zeilen, {len(df.columns)} Spalten)")

    df = _normalize(df)
    df.to_csv(RAW_CSV, index=False)
    log.info(f"Normalisiert und gespeichert: {RAW_CSV} ({len(df)} Zeilen)")
    return df


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalisiert Kaggle-Spalten auf internes Schema."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # ── Pflicht-Umbenennungen ─────────────────────────────────────────────────
    rename = {}

    # team_a / team_b
    for src, dst in [("team1_name", "team_a"), ("team2_name", "team_b")]:
        if src in df.columns:
            rename[src] = dst

    # winner bleibt winner
    # date
    if "date" not in df.columns:
        for alt in ["scraped_date", "match_date"]:
            if alt in df.columns:
                rename[alt] = "date"
                break

    # event
    if "tournament" in df.columns and "event" not in df.columns:
        rename["tournament"] = "event"

    df = df.rename(columns=rename)

    # ── Datum ─────────────────────────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ── Pflicht-Spalten prüfen ────────────────────────────────────────────────
    for col in ["team_a", "team_b", "winner"]:
        if col not in df.columns:
            raise ValueError(
                f"Pflicht-Spalte '{col}' fehlt. Verfügbare Spalten: {list(df.columns)[:20]}"
            )

    # ── Score ─────────────────────────────────────────────────────────────────
    if "score_a" not in df.columns and "score_team1" in df.columns:
        df["score_a"] = pd.to_numeric(df["score_team1"], errors="coerce")
    if "score_b" not in df.columns and "score_team2" in df.columns:
        df["score_b"] = pd.to_numeric(df["score_team2"], errors="coerce")

    df["maps"] = (df.get("score_a", pd.Series(0, index=df.index)).fillna(0) +
                  df.get("score_b", pd.Series(0, index=df.index)).fillna(0)).astype(int)

    # ── Strings normalisieren (Leerzeichen, Groß/Klein) ─────────────────────
    for col in ["team_a", "team_b", "winner"]:
        df[col] = df[col].astype(str).str.strip()

    # ── Label: team_a_won ─────────────────────────────────────────────────────
    # Versuch 1: exakter Vergleich
    # winner enthält "team1"/"team2" (nicht Teamnamen) → direkt aus Score ableiten
    w = df["winner"].astype(str).str.strip().str.lower()
    if w.isin(["team1", "team2"]).mean() > 0.8:
        df["team_a_won"] = (w == "team1").astype(int)
        log.info(f"winner enthält 'team1'/'team2' → team_a_won via Keyword-Mapping")
    elif "score_a" in df.columns and "score_b" in df.columns:
        df["team_a_won"] = (pd.to_numeric(df["score_a"], errors="coerce") >
                            pd.to_numeric(df["score_b"], errors="coerce")).astype(int)
        log.info("winner unklar → team_a_won via score_a > score_b")
    else:
        df["team_a_won"] = (df["winner"].str.strip() == df["team_a"].str.strip()).astype(int)

    # Diagnose: wenn Label fast immer 0 → Fallback auf Score-Vergleich
    label_rate = df["team_a_won"].mean()
    log.info(f"Label-Check: team_a_won = 1 bei {label_rate:.1%} der Matches")

    if label_rate < 0.05 or label_rate > 0.95:
        log.warning(
            f"team_a_won fast immer {int(round(label_rate))} ({label_rate:.1%}) — "
            f"winner-Spalte stimmt nicht mit team_a überein. Versuche Score-Fallback."
        )
        # Fallback: winner via score_team1 > score_team2
        if "score_a" in df.columns and "score_b" in df.columns:
            score_a = pd.to_numeric(df["score_a"], errors="coerce")
            score_b = pd.to_numeric(df["score_b"], errors="coerce")
            df["team_a_won"] = (score_a > score_b).astype(int)
            log.info(f"Score-Fallback: team_a_won = 1 bei {df['team_a_won'].mean():.1%}")
        else:
            # Letzter Fallback: winner enthält möglicherweise "team1"/"team2" statt Namen
            # Prüfe ob winner == "team1" / 1 / True
            w = df["winner"].astype(str).str.lower().str.strip()
            if w.isin(["team1", "1", "true", "a"]).any():
                df["team_a_won"] = w.isin(["team1", "1", "true", "a"]).astype(int)
                log.info(f"Keyword-Fallback: team_a_won = 1 bei {df['team_a_won'].mean():.1%}")
            else:
                log.error(
                    f"Kann team_a_won nicht bestimmen. "
                    f"winner-Beispiele: {df['winner'].value_counts().head(5).to_dict()}  |  "
                    f"team_a-Beispiele: {df['team_a'].value_counts().head(3).to_dict()}"
                )

    # ── Numerische Spalten bereinigen ─────────────────────────────────────────
    num_cols = [
        "rating_diff", "adr_diff", "kast_diff", "kpr_diff", "dpr_diff",
        "team1_avg_RATING", "team2_avg_RATING",
        "team1_avg_ADR",    "team2_avg_ADR",
        "team1_avg_KAST",   "team2_avg_KAST",
        "team1_avg_KPR",    "team2_avg_KPR",
        "team1_avg_DPR",    "team2_avg_DPR",
        "winner_head2head_percentage", "loser_head2head_percentage",
        "winner_head2head_freq",       "loser_head2head_freq",
        "winner_past3",     "loser_past3",
        "team1_overall_winrate", "team2_overall_winrate",
        "team1_lan_winrate",     "team2_lan_winrate",
        "team1_online_winrate",  "team2_online_winrate",
        "team1_totalwinrate",    "team2_totalwinrate",
        "team1_totallossrate",   "team2_totallossrate",
        "team1_rating_std",      "team2_rating_std",
        "consistency_advantage", "star_player_advantage",
        "weakest_link_advantage",
    ] + [f"winner_{m}" for m in CS2_MAPS] + [f"loser_{m}" for m in CS2_MAPS]

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Abgeleitete Diff-Features (team1 = winner-Perspektive) ───────────────
    # Winrate-Diffs
    df = _add_diff(df, "team1_overall_winrate", "team2_overall_winrate", "winrate_diff")
    df = _add_diff(df, "team1_lan_winrate",     "team2_lan_winrate",     "lan_winrate_diff")
    df = _add_diff(df, "team1_online_winrate",  "team2_online_winrate",  "online_winrate_diff")

    # H2H aus winner/loser-Perspektive auf team_a/team_b umrechnen
    # winner_* = Statistik des Match-Gewinners, loser_* = Statistik des Verlierers
    # Wir brauchen team1_h2h und team2_h2h
    if "winner_head2head_percentage" in df.columns:
        df["h2h_diff"] = np.where(
            df["team_a_won"] == 1,
            df["winner_head2head_percentage"] - df["loser_head2head_percentage"],
            df["loser_head2head_percentage"]  - df["winner_head2head_percentage"],
        )
        df["team1_h2h_pct"] = np.where(
            df["team_a_won"] == 1,
            df["winner_head2head_percentage"],
            df["loser_head2head_percentage"],
        )
        df["team2_h2h_pct"] = np.where(
            df["team_a_won"] == 1,
            df["loser_head2head_percentage"],
            df["winner_head2head_percentage"],
        )

    # past3 (letzte 3 Matches Winrate)
    if "winner_past3" in df.columns:
        df["past3_diff"] = np.where(
            df["team_a_won"] == 1,
            df["winner_past3"] - df["loser_past3"],
            df["loser_past3"]  - df["winner_past3"],
        )
        df["team1_past3"] = np.where(
            df["team_a_won"] == 1, df["winner_past3"], df["loser_past3"]
        )
        df["team2_past3"] = np.where(
            df["team_a_won"] == 1, df["loser_past3"], df["winner_past3"]
        )

    # Map-Winrate-Diffs pro Map
    for m in CS2_MAPS:
        w_col = f"winner_{m}"
        l_col = f"loser_{m}"
        if w_col in df.columns and l_col in df.columns:
            df[f"team1_{m}_winrate"] = np.where(
                df["team_a_won"] == 1, df[w_col], df[l_col]
            )
            df[f"team2_{m}_winrate"] = np.where(
                df["team_a_won"] == 1, df[l_col], df[w_col]
            )
            df[f"{m}_winrate_diff"] = df[f"team1_{m}_winrate"] - df[f"team2_{m}_winrate"]

    # Spieler-Ratings (team1/2 normalisieren auf team_a/b-Perspektive)
    # rating_diff ist im Dataset bereits vorhanden (team1 - team2)
    # Wir müssen prüfen ob es aus team1=winner oder team1=actual team1 berechnet wurde
    # Im Kaggle-Schema ist team1 = der erste genannte Team (nicht zwingend der Gewinner)
    # → rating_diff = team1_avg_RATING - team2_avg_RATING (bereits korrekt für team_a)

    # Falls rating_diff fehlt, selbst berechnen
    if "rating_diff" not in df.columns:
        if "team1_avg_RATING" in df.columns and "team2_avg_RATING" in df.columns:
            df["rating_diff"] = df["team1_avg_RATING"] - df["team2_avg_RATING"]
        else:
            df["rating_diff"] = 0.0

    if "adr_diff" not in df.columns and "team1_avg_ADR" in df.columns:
        df["adr_diff"] = df["team1_avg_ADR"] - df["team2_avg_ADR"]
    if "kast_diff" not in df.columns and "team1_avg_KAST" in df.columns:
        df["kast_diff"] = df["team1_avg_KAST"] - df["team2_avg_KAST"]
    if "kpr_diff" not in df.columns and "team1_avg_KPR" in df.columns:
        df["kpr_diff"] = df["team1_avg_KPR"] - df["team2_avg_KPR"]
    if "dpr_diff" not in df.columns and "team1_avg_DPR" in df.columns:
        df["dpr_diff"] = df["team1_avg_DPR"] - df["team2_avg_DPR"]

    # ── event_type für LAN/Online ─────────────────────────────────────────────
    if "event_type" in df.columns:
        df["is_lan"] = (df["event_type"].str.lower() == "lan").astype(int)
    else:
        df["is_lan"] = 0

    # ── Duplikate entfernen ───────────────────────────────────────────────────
    if "match_id" in df.columns:
        df = df.drop_duplicates(subset=["match_id"])
    elif "hltv_match_id" in df.columns:
        df = df.drop_duplicates(subset=["hltv_match_id"])

    df = df.dropna(subset=["team_a", "team_b", "winner"])
    df = df[(df["team_a"].str.len() > 0) & (df["team_b"].str.len() > 0)]
    df = df.reset_index(drop=True)

    # winner auf echten Teamnamen umschreiben (war "team1"/"team2")
    # Damit Dashboard-Logik (winner == team_name) funktioniert
    w = df["winner"].astype(str).str.strip().str.lower()
    if w.isin(["team1", "team2"]).mean() > 0.5:
        df["winner"] = np.where(df["team_a_won"] == 1, df["team_a"], df["team_b"])
        log.info("winner-Spalte auf echte Teamnamen umgeschrieben")

    log.info(f"Normalisierung fertig: {len(df)} Matches, {len(df.columns)} Spalten")
    return df


def _add_diff(df, col_a, col_b, out_col):
    if col_a in df.columns and col_b in df.columns:
        df[out_col] = df[col_a] - df[col_b]
    elif out_col not in df.columns:
        df[out_col] = 0.0
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()

    df = load_custom_csv(args.csv)
    print(df[["date", "team_a", "team_b", "winner", "rating_diff"]].tail(10).to_string())
    print(f"\nTeams: {df['team_a'].nunique()} | Matches: {len(df)}")
    print(f"Spalten: {list(df.columns)}")