# data/api_fetcher.py
# Holt aktuelle CS2 Pro-Match-Daten von:
#   - PandaScore (kostenloser Free-Tier) → Schedules, Ergebnisse, Teams
#   - GRID Open Access (kostenlos) → In-Game-Stats, Map-Daten
#
# Setup:
#   1. PandaScore: Account auf pandascore.co erstellen → API-Key kopieren
#   2. GRID:       Formular auf grid.gg/open-access ausfüllen → API-Key per Mail
#   3. Keys in .env eintragen:
#        PANDASCORE_API_KEY=dein_key
#        GRID_API_KEY=dein_key
#
# Nutzung:
#   python data/api_fetcher.py --fetch-results   # Vergangene Matches laden
#   python data/api_fetcher.py --fetch-upcoming  # Kommende Matches laden
#   python data/api_fetcher.py --update          # Alles auf einmal + CSV updaten
#   python data/api_fetcher.py --test            # API-Keys testen

import logging
import sys
import time
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
import os

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_CSV, DATA_DIR, CS2_MAPS

load_dotenv()

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# Konfiguration
# ─────────────────────────────────────────────────────────────────────────────

PANDASCORE_KEY  = os.getenv("PANDASCORE_API_KEY", "")
GRID_KEY        = os.getenv("GRID_API_KEY", "")

PANDASCORE_BASE = "https://api.pandascore.co"
GRID_BASE       = "https://api.grid.gg/central-data/graphql"

# Rate-Limits (PandaScore: 1000 req/h Free-Tier → 1 req/3.6s ist sicher)
PANDASCORE_DELAY = 1.5   # Sekunden zwischen Requests
GRID_DELAY       = 1.0

# Upcoming-Datei (wird vom Dashboard geladen)
UPCOMING_CSV = DATA_DIR / "upcoming_matches.csv"


# ─────────────────────────────────────────────────────────────────────────────
# HTTP-Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ps_get(endpoint: str, params: dict = None) -> list | dict | None:
    """PandaScore REST GET mit Auto-Paginierung."""
    if not PANDASCORE_KEY:
        log.error("PANDASCORE_API_KEY nicht gesetzt! Trage ihn in .env ein.")
        return None

    headers = {"Authorization": f"Bearer {PANDASCORE_KEY}"}
    url     = f"{PANDASCORE_BASE}{endpoint}"
    params  = params or {}
    params.setdefault("per_page", 100)

    all_items = []
    page = 1

    while True:
        params["page"] = page
        try:
            r = requests.get(url, headers=headers, params=params, timeout=15)
        except requests.RequestException as e:
            log.error(f"PandaScore Request-Fehler: {e}")
            break

        if r.status_code == 401:
            log.error("PandaScore: Ungültiger API-Key (401). Prüfe PANDASCORE_API_KEY in .env")
            return None
        if r.status_code == 429:
            log.warning("PandaScore Rate-Limit erreicht. Warte 60s ...")
            time.sleep(60)
            continue
        if r.status_code != 200:
            log.error(f"PandaScore HTTP {r.status_code}: {r.text[:200]}")
            break

        data = r.json()
        if not isinstance(data, list):
            return data   # Einzelnes Objekt

        all_items.extend(data)

        # Paginierung: X-Total Header prüfen
        total = int(r.headers.get("X-Total", len(data)))
        per_page = int(r.headers.get("X-Per-Page", 100))
        if page * per_page >= total:
            break
        page += 1
        time.sleep(PANDASCORE_DELAY)

    return all_items


def _grid_query(query: str, variables: dict = None) -> dict | None:
    """GRID GraphQL POST."""
    if not GRID_KEY:
        log.warning("GRID_API_KEY nicht gesetzt. GRID-Daten werden übersprungen.")
        return None

    headers = {
        "x-api-key":    GRID_KEY,
        "Content-Type": "application/json",
    }
    payload = {"query": query, "variables": variables or {}}

    try:
        r = requests.post(GRID_BASE, headers=headers,
                          json=payload, timeout=20)
    except requests.RequestException as e:
        log.error(f"GRID Request-Fehler: {e}")
        return None

    if r.status_code == 401:
        log.error("GRID: Ungültiger API-Key (401). Prüfe GRID_API_KEY in .env")
        return None
    if r.status_code != 200:
        log.error(f"GRID HTTP {r.status_code}: {r.text[:200]}")
        return None

    result = r.json()
    if "errors" in result:
        log.error(f"GRID GraphQL-Fehler: {result['errors']}")
        return None

    time.sleep(GRID_DELAY)
    return result.get("data")


# ─────────────────────────────────────────────────────────────────────────────
# PandaScore — Vergangene Matches
# ─────────────────────────────────────────────────────────────────────────────

def fetch_past_matches(days_back: int = 30) -> list[dict]:
    """
    Holt abgeschlossene CS2 Matches der letzten N Tage von PandaScore.
    Paginiert automatisch durch ALLE Seiten bis zum gewünschten Zeitraum.
    Kein künstliches max_pages-Limit mehr.
    """
    since_dt  = datetime.now(timezone.utc) - timedelta(days=days_back)
    until_dt  = datetime.now(timezone.utc)
    since_str = since_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    until_str = until_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    log.info(f"PandaScore: Lade Matches von {since_dt.strftime('%Y-%m-%d')} "
             f"bis {until_dt.strftime('%Y-%m-%d')} ...")

    # PandaScore range-Filter: begin_at ist zuverlässiger als end_at
    params = {
        "filter[status]":    "finished",
        "filter[videogame]": "cs-go",
        "range[begin_at]":   f"{since_str},{until_str}",
        "sort":              "-begin_at",
        "per_page":          100,
    }

    all_raw = []
    page    = 1

    while True:
        params["page"] = page
        headers = {"Authorization": f"Bearer {PANDASCORE_KEY}"}
        try:
            r = requests.get(
                f"{PANDASCORE_BASE}/csgo/matches",
                headers=headers,
                params=params,
                timeout=15,
            )
        except requests.RequestException as e:
            log.error(f"PandaScore Request-Fehler Seite {page}: {e}")
            break

        if r.status_code == 401:
            log.error("PandaScore: Ungültiger API-Key (401)")
            break
        if r.status_code == 429:
            log.warning("Rate-Limit — warte 60s ...")
            time.sleep(60)
            continue
        if r.status_code != 200:
            log.error(f"PandaScore HTTP {r.status_code}: {r.text[:200]}")
            break

        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            break

        all_raw.extend(data)

        # Paginierung über X-Total
        total    = int(r.headers.get("X-Total",    len(data)))
        per_page = int(r.headers.get("X-Per-Page", 100))
        loaded   = page * per_page

        log.info(f"  Seite {page}: {len(data)} Matches "
                 f"(geladen: {len(all_raw)}/{total})")

        if loaded >= total:
            break

        page += 1
        time.sleep(PANDASCORE_DELAY)

    matches = []
    for m in all_raw:
        parsed = _parse_ps_match(m)
        if parsed:
            matches.append(parsed)

    log.info(f"PandaScore: {len(matches)} Matches geladen "
             f"({days_back} Tage, {page} Seiten)")
    return matches


def fetch_upcoming_matches() -> list[dict]:
    """
    Holt kommende CS2 Matches von PandaScore.
    Diese werden in UPCOMING_CSV gespeichert (nicht ins Trainings-CSV).
    """
    log.info("PandaScore: Lade kommende Matches ...")

    params = {
        "filter[status]":    "not_started",
        "filter[videogame]": "cs-go",
        "sort":              "begin_at",
        "per_page":          50,
    }

    raw = _ps_get("/csgo/matches", params)
    if not raw:
        return []

    upcoming = []
    for m in raw:
        parsed = _parse_ps_upcoming(m)
        if parsed:
            upcoming.append(parsed)

    log.info(f"PandaScore: {len(upcoming)} kommende Matches gefunden")
    return upcoming


def _parse_ps_match(m: dict) -> dict | None:
    """Normalisiert einen PandaScore-Match auf internes Schema."""
    try:
        opponents = m.get("opponents", [])
        if len(opponents) < 2:
            return None

        team_a = opponents[0]["opponent"]["name"]
        team_b = opponents[1]["opponent"]["name"]

        # Gewinner bestimmen
        winner_obj = m.get("winner")
        if winner_obj and winner_obj.get("name"):
            winner = winner_obj["name"]
        else:
            # Aus Results lesen
            results = m.get("results", [])
            if len(results) >= 2:
                score_a = results[0].get("score", 0)
                score_b = results[1].get("score", 0)
                winner = team_a if score_a > score_b else team_b
            else:
                return None

        results = m.get("results", [])
        score_a = results[0].get("score", 0) if len(results) > 0 else 0
        score_b = results[1].get("score", 0) if len(results) > 1 else 0

        date_str = m.get("end_at") or m.get("begin_at") or ""
        date = pd.to_datetime(date_str, errors="coerce", utc=True)
        if pd.isna(date):
            return None

        team_a_won = int(winner == team_a)

        return {
            "match_id":    f"ps_{m.get('id', '')}",
            "date":        date.tz_convert(None),
            "team_a":      team_a,
            "team_b":      team_b,
            "score_a":     score_a,
            "score_b":     score_b,
            "winner":      winner,
            "team_a_won":  team_a_won,
            "event":       m.get("league", {}).get("name", "Unknown"),
            "tournament":  m.get("serie", {}).get("full_name", ""),
            "maps":        score_a + score_b,
            "match_type":  m.get("match_type", ""),
            "source":      "pandascore",
        }
    except (KeyError, TypeError, IndexError) as e:
        log.debug(f"Parse-Fehler bei Match {m.get('id')}: {e}")
        return None


def _parse_ps_upcoming(m: dict) -> dict | None:
    """Normalisiert einen kommenden PandaScore-Match."""
    try:
        opponents = m.get("opponents", [])
        team_a = opponents[0]["opponent"]["name"] if len(opponents) > 0 else "TBD"
        team_b = opponents[1]["opponent"]["name"] if len(opponents) > 1 else "TBD"

        date_str = m.get("begin_at") or m.get("scheduled_at") or ""
        date = pd.to_datetime(date_str, errors="coerce", utc=True)
        if pd.isna(date):
            return None

        return {
            "match_id":   f"ps_{m.get('id', '')}",
            "date":       date.tz_convert(None),
            "team_a":     team_a,
            "team_b":     team_b,
            "event":      m.get("league", {}).get("name", "Unknown"),
            "tournament": m.get("serie", {}).get("full_name", ""),
            "match_type": m.get("match_type", ""),
            "status":     m.get("status", ""),
            "stream_url": (m.get("streams_list") or [{}])[0].get("raw_url", ""),
            "source":     "pandascore",
        }
    except (KeyError, TypeError, IndexError) as e:
        log.debug(f"Upcoming-Parse-Fehler: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GRID — In-Game-Stats für abgeschlossene Matches
# ─────────────────────────────────────────────────────────────────────────────

# CS2 Title-ID in GRID = 3
GRID_CS2_QUERY = """
query GetCS2Series($after: Cursor, $first: Int) {
  allSeries(
    first: $first
    after: $after
    filter: { titleId: { eq: 3 } }
    orderBy: STARTED_AT_DESC
  ) {
    pageInfo { hasNextPage endCursor }
    edges {
      node {
        id
        startTimeScheduled
        format { nameShortened }
        teams {
          baseInfo { id name }
          won
        }
        games {
          sequenceNumber
          map { name }
          teams {
            baseInfo { name }
            won
            players {
              baseInfo { nickname }
              kills deaths assists
              adr hsPercent
              kastPercent rating
            }
          }
        }
      }
    }
  }
}
"""


def fetch_grid_stats(max_series: int = 50) -> list[dict]:
    """
    Holt CS2 In-Game-Stats von GRID Open Access.
    Gibt normalisierte Match-Dicts zurück (kompatibel mit internem Schema).
    """
    if not GRID_KEY:
        log.info("GRID_API_KEY nicht gesetzt — GRID-Daten werden übersprungen.")
        return []

    log.info(f"GRID: Lade bis zu {max_series} CS2 Series ...")

    all_matches = []
    cursor = None
    fetched = 0

    while fetched < max_series:
        variables = {
            "first": min(20, max_series - fetched),
            "after": cursor,
        }
        data = _grid_query(GRID_CS2_QUERY, variables)
        if not data:
            break

        series_data = data.get("allSeries", {})
        edges = series_data.get("edges", [])

        for edge in edges:
            node = edge["node"]
            parsed = _parse_grid_series(node)
            if parsed:
                all_matches.append(parsed)

        fetched += len(edges)
        page_info = series_data.get("pageInfo", {})
        if not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor")

    log.info(f"GRID: {len(all_matches)} Matches geladen")
    return all_matches


def _parse_grid_series(node: dict) -> dict | None:
    """Normalisiert eine GRID-Series auf internes Schema."""
    try:
        teams = node.get("teams", [])
        if len(teams) < 2:
            return None

        team_a_obj = teams[0]
        team_b_obj = teams[1]
        team_a = team_a_obj["baseInfo"]["name"]
        team_b = team_b_obj["baseInfo"]["name"]

        # Gewinner aus won-Flag
        won_a = team_a_obj.get("won")
        won_b = team_b_obj.get("won")
        if won_a is True:
            winner = team_a
        elif won_b is True:
            winner = team_b
        else:
            return None   # Kein klarer Gewinner (noch laufend?)

        date = pd.to_datetime(node.get("startTimeScheduled"), errors="coerce", utc=True)
        if pd.isna(date):
            return None

        team_a_won = int(winner == team_a)

        # Maps aus games extrahieren
        games = node.get("games", [])
        score_a = sum(1 for g in games
                      if any(t["baseInfo"]["name"] == team_a and t.get("won")
                             for t in g.get("teams", [])))
        score_b = len(games) - score_a

        # Spieler-Ratings aggregieren
        def _player_stats(team_name):
            ratings, kdrs, adrs, kasts = [], [], [], []
            for g in games:
                for t in g.get("teams", []):
                    if t["baseInfo"]["name"] != team_name:
                        continue
                    for p in t.get("players", []):
                        if p.get("rating") is not None:
                            ratings.append(float(p["rating"]))
                        if p.get("adr") is not None:
                            adrs.append(float(p["adr"]))
                        if p.get("kastPercent") is not None:
                            kasts.append(float(p["kastPercent"]))
                        k = p.get("kills", 0) or 0
                        d = p.get("deaths", 1) or 1
                        kdrs.append(k / d)
            return {
                "avg_rating": float(np.mean(ratings)) if ratings else np.nan,
                "avg_adr":    float(np.mean(adrs))    if adrs    else np.nan,
                "avg_kast":   float(np.mean(kasts))   if kasts   else np.nan,
                "avg_kdr":    float(np.mean(kdrs))    if kdrs    else np.nan,
            }

        stats_a = _player_stats(team_a)
        stats_b = _player_stats(team_b)

        # Map-Winrates aus games
        map_stats = {}
        for m in CS2_MAPS:
            map_games = [g for g in games if (g.get("map") or {}).get("name", "").lower() == m]
            if map_games:
                a_wins = sum(1 for g in map_games
                             if any(t["baseInfo"]["name"] == team_a and t.get("won")
                                    for t in g.get("teams", [])))
                map_stats[f"team1_{m}_winrate"] = a_wins / len(map_games)
                map_stats[f"team2_{m}_winrate"] = (len(map_games) - a_wins) / len(map_games)
                map_stats[f"{m}_winrate_diff"]  = map_stats[f"team1_{m}_winrate"] - map_stats[f"team2_{m}_winrate"]

        result = {
            "match_id":         f"grid_{node['id']}",
            "date":             date.tz_convert(None),
            "team_a":           team_a,
            "team_b":           team_b,
            "score_a":          score_a,
            "score_b":          score_b,
            "winner":           winner,
            "team_a_won":       team_a_won,
            "event":            "GRID",
            "maps":             len(games),
            # Ratings
            "team1_avg_RATING": stats_a["avg_rating"],
            "team2_avg_RATING": stats_b["avg_rating"],
            "team1_avg_ADR":    stats_a["avg_adr"],
            "team2_avg_ADR":    stats_b["avg_adr"],
            "team1_avg_KAST":   stats_a["avg_kast"],
            "team2_avg_KAST":   stats_b["avg_kast"],
            # Diffs (aus team1-Sicht = team_a)
            "rating_diff": _safe_diff(stats_a["avg_rating"], stats_b["avg_rating"]),
            "adr_diff":    _safe_diff(stats_a["avg_adr"],    stats_b["avg_adr"]),
            "kast_diff":   _safe_diff(stats_a["avg_kast"],   stats_b["avg_kast"]),
            "source":      "grid",
        }
        result.update(map_stats)
        return result

    except (KeyError, TypeError, ValueError) as e:
        log.debug(f"GRID-Parse-Fehler bei Series {node.get('id')}: {e}")
        return None


def _safe_diff(a, b):
    if a is None or b is None or np.isnan(a) or np.isnan(b):
        return np.nan
    return float(a) - float(b)


# ─────────────────────────────────────────────────────────────────────────────
# CSV-Merge: Neue Matches in bestehende CSV einpflegen
# ─────────────────────────────────────────────────────────────────────────────

def merge_into_csv(new_matches: list[dict], csv_path: Path = RAW_CSV) -> pd.DataFrame:
    """
    Fügt neue Matches in die bestehende raw CSV ein.
    Duplikate werden via match_id oder (date, team_a, team_b) erkannt.
    """
    if not new_matches:
        log.warning("Keine neuen Matches zum Mergen.")
        if csv_path.exists():
            return pd.read_csv(csv_path, parse_dates=["date"])
        return pd.DataFrame()

    df_new = pd.DataFrame(new_matches)
    df_new["date"] = pd.to_datetime(df_new["date"], errors="coerce")
    df_new = df_new.dropna(subset=["date", "team_a", "team_b", "winner"])

    if csv_path.exists():
        df_old = pd.read_csv(csv_path, parse_dates=["date"])
        df_old["date"] = pd.to_datetime(df_old["date"], errors="coerce").dt.tz_localize(None)

        # Duplikat-Check
        if "match_id" in df_old.columns and "match_id" in df_new.columns:
            existing_ids = set(df_old["match_id"].dropna())
            df_new = df_new[~df_new["match_id"].isin(existing_ids)]
        else:
            # Fallback: date + teams
            existing = set(zip(df_old["date"].dt.date,
                               df_old["team_a"], df_old["team_b"]))
            df_new = df_new[~df_new.apply(
                lambda r: (r["date"].date(), r["team_a"], r["team_b"]) in existing, axis=1
            )]

        if len(df_new) == 0:
            log.info("Keine neuen Matches (alle bereits in CSV vorhanden).")
            return df_old

        df_merged = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_merged = df_new

    df_merged = df_merged.sort_values("date").reset_index(drop=True)
    df_merged.to_csv(csv_path, index=False)
    log.info(f"CSV aktualisiert: {csv_path} (+{len(df_new)} neue Matches, gesamt {len(df_merged)})")
    return df_merged


def save_upcoming(upcoming: list[dict]):
    """Speichert kommende Matches in upcoming_matches.csv."""
    if not upcoming:
        log.warning("Keine kommenden Matches gefunden.")
        return
    df = pd.DataFrame(upcoming)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(UPCOMING_CSV, index=False)
    log.info(f"Kommende Matches gespeichert: {UPCOMING_CSV} ({len(df)} Matches)")


# ─────────────────────────────────────────────────────────────────────────────
# API-Key Test
# ─────────────────────────────────────────────────────────────────────────────

def test_keys():
    """Testet ob die API-Keys funktionieren."""
    print("\n── PandaScore ───────────────────────────────")
    if not PANDASCORE_KEY:
        print("❌ PANDASCORE_API_KEY nicht gesetzt")
    else:
        r = requests.get(
            f"{PANDASCORE_BASE}/csgo/matches",
            headers={"Authorization": f"Bearer {PANDASCORE_KEY}"},
            params={"per_page": 1},
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json()
            print(f"✅ PandaScore OK — {len(data)} Match-Sample geladen")
            if data:
                m = data[0]
                teams = m.get("opponents", [])
                print(f"   Beispiel: {teams[0]['opponent']['name'] if teams else '?'} "
                      f"vs {teams[1]['opponent']['name'] if len(teams)>1 else '?'}")
        elif r.status_code == 401:
            print("❌ PandaScore: Ungültiger API-Key (401)")
        else:
            print(f"❌ PandaScore HTTP {r.status_code}: {r.text[:100]}")

    print("\n── GRID Open Access ─────────────────────────")
    if not GRID_KEY:
        print("❌ GRID_API_KEY nicht gesetzt (optional)")
    else:
        test_query = """{ allSeries(first: 1, filter: {titleId: {eq: 3}}) {
            edges { node { id startTimeScheduled teams { baseInfo { name } } } }
        }}"""
        data = _grid_query(test_query)
        if data:
            edges = data.get("allSeries", {}).get("edges", [])
            if edges:
                node = edges[0]["node"]
                teams = [t["baseInfo"]["name"] for t in node.get("teams", [])]
                print(f"✅ GRID OK — Beispiel-Series: {' vs '.join(teams)}")
            else:
                print("✅ GRID OK (keine Daten zurückgegeben — möglicherweise Zugriffsbeschränkung)")
        else:
            print("❌ GRID: Fehler beim Abrufen (prüfe GRID_API_KEY)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CS2 API Fetcher — PandaScore + GRID",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python data/api_fetcher.py --test                    API-Keys testen
  python data/api_fetcher.py --fetch-results           Letzte 30 Tage Ergebnisse
  python data/api_fetcher.py --fetch-results --days 90 Letzte 90 Tage
  python data/api_fetcher.py --fetch-upcoming          Kommende Matches
  python data/api_fetcher.py --update                  Alles auf einmal
        """
    )
    parser.add_argument("--test",           action="store_true", help="API-Keys testen")
    parser.add_argument("--fetch-results",  action="store_true", help="Vergangene Matches laden")
    parser.add_argument("--fetch-upcoming", action="store_true", help="Kommende Matches laden")
    parser.add_argument("--fetch-grid",     action="store_true", help="GRID In-Game-Stats laden")
    parser.add_argument("--update",         action="store_true", help="Alles auf einmal")
    parser.add_argument("--days",           type=int, default=30, help="Tage zurück (default: 30)")
    parser.add_argument("--retrain",        action="store_true", help="Nach Update Modell neu trainieren")
    args = parser.parse_args()

    if args.test:
        test_keys()

    elif args.fetch_results:
        matches = fetch_past_matches(days_back=args.days)
        if matches:
            merge_into_csv(matches)

    elif args.fetch_upcoming:
        upcoming = fetch_upcoming_matches()
        save_upcoming(upcoming)
        print(f"\n{len(upcoming)} kommende Matches:")
        for u in upcoming[:10]:
            print(f"  {u['date'].strftime('%Y-%m-%d %H:%M')} — "
                  f"{u['team_a']} vs {u['team_b']} ({u['event']})")

    elif args.fetch_grid:
        matches = fetch_grid_stats(max_series=100)
        if matches:
            merge_into_csv(matches)

    elif args.update:
        print("═" * 50)
        print("VOLLSTÄNDIGES UPDATE")
        print("═" * 50)

        # 1. PandaScore Ergebnisse
        print("\n[1/3] PandaScore Ergebnisse ...")
        ps_matches = fetch_past_matches(days_back=args.days)
        if ps_matches:
            merge_into_csv(ps_matches)

        # 2. GRID Stats
        print("\n[2/3] GRID In-Game-Stats ...")
        grid_matches = fetch_grid_stats(max_series=50)
        if grid_matches:
            merge_into_csv(grid_matches)

        # 3. Upcoming Matches
        print("\n[3/3] Kommende Matches ...")
        upcoming = fetch_upcoming_matches()
        save_upcoming(upcoming)

        print("\n✅ Update abgeschlossen!")
        print(f"   Raw CSV:  {RAW_CSV}")
        print(f"   Upcoming: {UPCOMING_CSV}")

        # Optional: Modell neu trainieren
        if args.retrain:
            print("\n[4/4] Modell neu trainieren ...")
            import subprocess
            subprocess.run([sys.executable, "train.py", "--rebuild"], check=True)

    else:
        parser.print_help()