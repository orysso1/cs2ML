# data/scraper.py
# Scrapt CS2 Pro-Match-Daten von HLTV.org
# Nutzt hltv-async-api + Fallback auf direktes HTML-Parsing

import asyncio
import random
import time
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Optionaler Import — hltv-async-api muss installiert sein
try:
    from hltv_async_api import Hltv
    HLTV_ASYNC_AVAILABLE = True
except ImportError:
    HLTV_ASYNC_AVAILABLE = False
    logging.warning("hltv-async-api nicht installiert. Nutze direktes HTML-Parsing.")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RAW_CSV, HLTV_MIN_DELAY, HLTV_MAX_DELAY,
    HLTV_MAX_RETRIES, SCRAPE_PAGES
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Header-Rotation (vermindert Bot-Erkennung) ───────────────────────────────
USER_AGENTS = [
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/121.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
]


def _sleep():
    """Zufällige Pause zwischen Requests."""
    t = random.uniform(HLTV_MIN_DELAY, HLTV_MAX_DELAY)
    log.debug(f"Warte {t:.1f}s ...")
    time.sleep(t)


def _get(url: str, retries: int = HLTV_MAX_RETRIES) -> BeautifulSoup | None:
    """HTTP GET mit Retry und User-Agent-Rotation."""
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                return BeautifulSoup(r.text, "html.parser")
            elif r.status_code == 403:
                log.warning(f"403 Forbidden (Versuch {attempt}/{retries}) – längere Pause ...")
                time.sleep(random.uniform(15, 30))
            else:
                log.warning(f"HTTP {r.status_code} bei {url}")
        except requests.RequestException as e:
            log.error(f"Request-Fehler: {e}")
        _sleep()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Direktes HTML-Scraping (ohne externe Lib)
# ─────────────────────────────────────────────────────────────────────────────

def scrape_results_page(offset: int = 0) -> list[dict]:
    """Scrapt eine Seite der HLTV-Ergebnisliste."""
    url = f"https://www.hltv.org/results?offset={offset}"
    soup = _get(url)
    if not soup:
        return []

    matches = []
    for result in soup.select(".result-con"):
        try:
            teams = result.select(".team")
            if len(teams) < 2:
                continue

            team_a = teams[0].get_text(strip=True)
            team_b = teams[1].get_text(strip=True)

            scores = result.select(".result-score span")
            if len(scores) < 2:
                continue

            score_a = int(scores[0].get_text(strip=True))
            score_b = int(scores[1].get_text(strip=True))

            # Datum aus data-Attribut (Unix ms → datetime)
            date_el = result.select_one("[data-time-format]") or result.select_one(".date")
            date_str = ""
            if date_el and date_el.get("data-unix"):
                ts = int(date_el["data-unix"]) / 1000
                date_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
            elif date_el:
                date_str = date_el.get_text(strip=True)

            # Event-Name
            event_el = result.select_one(".event-name")
            event = event_el.get_text(strip=True) if event_el else "Unknown"

            # Match-Link → Match-ID
            link_el = result.select_one("a.a-reset")
            match_id = ""
            if link_el and link_el.get("href"):
                parts = link_el["href"].split("/")
                match_id = parts[2] if len(parts) > 2 else ""

            # Gewinner bestimmen
            winner = team_a if score_a > score_b else team_b

            matches.append({
                "match_id": match_id,
                "date":     date_str,
                "team_a":   team_a,
                "team_b":   team_b,
                "score_a":  score_a,
                "score_b":  score_b,
                "winner":   winner,
                "event":    event,
                "maps":     score_a + score_b,
            })
        except (ValueError, IndexError, AttributeError) as e:
            log.debug(f"Parse-Fehler bei Match: {e}")
            continue

    return matches


def scrape_upcoming_page() -> list[dict]:
    """Scrapt kommende Matches von HLTV."""
    url = "https://www.hltv.org/matches"
    soup = _get(url)
    if not soup:
        return []

    upcoming = []
    for match in soup.select(".upcomingMatch"):
        try:
            teams = match.select(".matchTeamName")
            if len(teams) < 2:
                continue

            team_a = teams[0].get_text(strip=True)
            team_b = teams[1].get_text(strip=True)

            date_el = match.select_one("[data-unix]")
            date_str = ""
            if date_el:
                ts = int(date_el["data-unix"]) / 1000
                date_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M")

            event_el = match.select_one(".matchEventName")
            event = event_el.get_text(strip=True) if event_el else "Unknown"

            link_el = match.select_one("a")
            match_id = ""
            if link_el and link_el.get("href"):
                parts = link_el["href"].split("/")
                match_id = parts[2] if len(parts) > 2 else ""

            upcoming.append({
                "match_id": match_id,
                "date":     date_str,
                "team_a":   team_a,
                "team_b":   team_b,
                "event":    event,
            })
        except (ValueError, AttributeError) as e:
            log.debug(f"Upcoming-Parse-Fehler: {e}")
            continue

    return upcoming


# ─────────────────────────────────────────────────────────────────────────────
# Async-API-Variante (hltv-async-api)
# ─────────────────────────────────────────────────────────────────────────────

async def _async_fetch_results(pages: int) -> list[dict]:
    """Nutzt hltv-async-api für zuverlässigeres Scraping."""
    matches = []
    async with Hltv(max_delay=HLTV_MAX_DELAY, min_delay=HLTV_MIN_DELAY) as hltv:
        for page in range(pages):
            try:
                results = await hltv.get_results(max_age_days=365 * 3, offset=page * 100)
                for r in results or []:
                    matches.append({
                        "match_id": str(r.get("id", "")),
                        "date":     str(r.get("date", "")),
                        "team_a":   r.get("team1", {}).get("name", ""),
                        "team_b":   r.get("team2", {}).get("name", ""),
                        "score_a":  r.get("result", {}).get("team1", 0),
                        "score_b":  r.get("result", {}).get("team2", 0),
                        "winner":   r.get("winner", {}).get("name", ""),
                        "event":    r.get("event", {}).get("name", "Unknown"),
                        "maps":     (r.get("result", {}).get("team1", 0) +
                                     r.get("result", {}).get("team2", 0)),
                    })
                log.info(f"Seite {page+1}/{pages} geladen ({len(results or [])} Matches)")
            except Exception as e:
                log.error(f"Async-Fetch Fehler Seite {page}: {e}")
    return matches


# ─────────────────────────────────────────────────────────────────────────────
# Öffentliche API
# ─────────────────────────────────────────────────────────────────────────────

def load_or_scrape(force_scrape: bool = False) -> pd.DataFrame:
    """
    Lädt gecachte Daten aus CSV oder scrapt neu von HLTV.
    Nutzt hltv-async-api wenn verfügbar, sonst direktes HTML-Parsing.
    """
    if RAW_CSV.exists() and not force_scrape:
        log.info(f"Lade gecachte Daten aus {RAW_CSV}")
        df = pd.read_csv(RAW_CSV, parse_dates=["date"])
        log.info(f"{len(df)} Matches geladen.")
        return df

    log.info("Starte HLTV-Scraping ...")
    all_matches: list[dict] = []

    if HLTV_ASYNC_AVAILABLE:
        log.info("Nutze hltv-async-api ...")
        all_matches = asyncio.run(_async_fetch_results(SCRAPE_PAGES))
    else:
        log.info("Nutze direktes HTML-Parsing ...")
        for page in range(SCRAPE_PAGES):
            offset = page * 100
            log.info(f"Seite {page+1}/{SCRAPE_PAGES} (offset={offset})")
            page_matches = scrape_results_page(offset)
            all_matches.extend(page_matches)
            log.info(f"  +{len(page_matches)} Matches (gesamt: {len(all_matches)})")
            _sleep()

    if not all_matches:
        raise RuntimeError("Keine Matches gescrapt. Prüfe HLTV-Erreichbarkeit / Bot-Schutz.")

    df = pd.DataFrame(all_matches)
    df = _clean_raw(df)
    df.to_csv(RAW_CSV, index=False)
    log.info(f"Gespeichert: {RAW_CSV} ({len(df)} Matches)")
    return df


def get_upcoming() -> pd.DataFrame:
    """Gibt kommende Matches als DataFrame zurück."""
    upcoming = scrape_upcoming_page()
    if not upcoming:
        return pd.DataFrame(columns=["match_id", "date", "team_a", "team_b", "event"])
    return pd.DataFrame(upcoming)


def _clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Basis-Bereinigung der Rohdaten."""
    df = df.drop_duplicates(subset=["match_id"]) if "match_id" in df.columns else df
    df = df.dropna(subset=["team_a", "team_b", "winner"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Leere Team-Namen entfernen
    df = df[(df["team_a"].str.len() > 0) & (df["team_b"].str.len() > 0)]
    # Nur Matches mit echtem Gewinner
    df = df[df["winner"].isin(df["team_a"]) | df["winner"].isin(df["team_b"])]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI-Aufruf
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HLTV Scraper")
    parser.add_argument("--force", action="store_true", help="Cache ignorieren und neu scrapen")
    args = parser.parse_args()

    df = load_or_scrape(force_scrape=args.force)
    print(df.tail(10).to_string())
    print(f"\nGesamt: {len(df)} Matches | Teams: {df['team_a'].nunique()}")