#!/usr/bin/env python3
# auto_updater.py
# Läuft im Hintergrund und:
#   1. Holt neue abgeschlossene Matches von PandaScore / GRID
#   2. Fügt sie in matches_raw.csv ein
#   3. Trainiert das Modell neu (inkrementell)
#   4. Holt kommende Matches für das Dashboard
#
# Starten:
#   python auto_updater.py               # Einmal ausführen
#   python auto_updater.py --watch       # Dauerhaft alle N Stunden
#   python auto_updater.py --watch --interval 6   # Alle 6 Stunden

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import RAW_CSV, FEATURES_CSV, MODEL_PATH, DATA_DIR

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("auto_updater.log"),
    ]
)

# ─────────────────────────────────────────────────────────────────────────────
# Haupt-Update-Schritt
# ─────────────────────────────────────────────────────────────────────────────

def run_update(days_back: int = 3, force_retrain: bool = False) -> dict:
    """
    Führt einen vollständigen Update-Zyklus durch:
    1. Neue Matches fetchen
    2. In CSV mergen
    3. Falls neue Matches → Modell neu trainieren
    4. Upcoming Matches aktualisieren

    Returns: Dict mit Statistiken über den Update-Lauf
    """
    stats = {
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "new_matches":  0,
        "retrained":    False,
        "errors":       [],
    }

    log.info("=" * 55)
    log.info(f"UPDATE-ZYKLUS — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 55)

    # ── 1. Neue Matches holen ─────────────────────────────────────────────────
    new_matches = []

    try:
        from data.api_fetcher import fetch_past_matches, fetch_grid_stats
        from data.api_fetcher import PANDASCORE_KEY, GRID_KEY

        if PANDASCORE_KEY:
            log.info(f"[1a] PandaScore: Lade Matches der letzten {days_back} Tage ...")
            ps = fetch_past_matches(days_back=days_back)
            new_matches.extend(ps)
            log.info(f"     → {len(ps)} Matches von PandaScore")
        else:
            log.info("[1a] PandaScore: Kein API-Key — übersprungen")

        if GRID_KEY:
            log.info("[1b] GRID: Lade aktuelle In-Game-Stats ...")
            gr = fetch_grid_stats(max_series=30)
            new_matches.extend(gr)
            log.info(f"     → {len(gr)} Matches von GRID")
        else:
            log.info("[1b] GRID: Kein API-Key — übersprungen")

    except Exception as e:
        msg = f"Fetch-Fehler: {e}"
        log.error(msg)
        stats["errors"].append(msg)

    if not new_matches:
        log.info("Keine neuen Matches gefunden.")
    
    # ── 2. In CSV mergen ──────────────────────────────────────────────────────
    matches_before = _count_csv(RAW_CSV)

    if new_matches:
        try:
            log.info(f"[2] Merge {len(new_matches)} Matches in {RAW_CSV.name} ...")
            from data.api_fetcher import merge_into_csv
            df_merged = merge_into_csv(new_matches, RAW_CSV)
            matches_after = len(df_merged)
            actually_new  = matches_after - matches_before
            stats["new_matches"] = actually_new
            log.info(f"    → {actually_new} wirklich neue Matches hinzugefügt "
                     f"(gesamt: {matches_after})")
        except Exception as e:
            msg = f"Merge-Fehler: {e}"
            log.error(msg)
            stats["errors"].append(msg)
            actually_new = 0
    else:
        actually_new = 0

    # ── 3. Modell neu trainieren ──────────────────────────────────────────────
    should_retrain = force_retrain or actually_new > 0

    if should_retrain:
        log.info(f"[3] Starte Re-Training ({actually_new} neue Matches) ...")
        try:
            _retrain()
            stats["retrained"] = True
            log.info("    → Modell erfolgreich neu trainiert")
        except Exception as e:
            msg = f"Training-Fehler: {e}"
            log.error(msg)
            stats["errors"].append(msg)
    else:
        log.info("[3] Kein Re-Training nötig (keine neuen Matches)")

    # ── 4. Upcoming Matches aktualisieren ─────────────────────────────────────
    log.info("[4] Lade kommende Matches ...")
    try:
        from data.api_fetcher import fetch_upcoming_matches, save_upcoming, PANDASCORE_KEY
        if PANDASCORE_KEY:
            upcoming = fetch_upcoming_matches()
            save_upcoming(upcoming)
            log.info(f"    → {len(upcoming)} kommende Matches gespeichert")
        else:
            log.info("    → Kein PandaScore-Key — Upcoming übersprungen")
    except Exception as e:
        msg = f"Upcoming-Fehler: {e}"
        log.warning(msg)

    # ── Zusammenfassung ───────────────────────────────────────────────────────
    log.info("")
    log.info(f"✅ Update abgeschlossen | "
             f"Neue Matches: {stats['new_matches']} | "
             f"Neu trainiert: {stats['retrained']} | "
             f"Fehler: {len(stats['errors'])}")

    _save_last_run(stats)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Re-Training
# ─────────────────────────────────────────────────────────────────────────────

def _retrain():
    """
    Lädt aktualisierte Rohdaten, berechnet Features neu und trainiert Modell.
    Nutzt den bestehenden Kaggle-Trainingsdatensatz + neue API-Matches zusammen.
    """
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Keine Rohdaten unter {RAW_CSV}")

    log.info("    Lade Rohdaten ...")
    df_raw = pd.read_csv(RAW_CSV, parse_dates=["date"])

    # Timezone entfernen
    if pd.api.types.is_datetime64tz_dtype(df_raw["date"]):
        df_raw["date"] = df_raw["date"].dt.tz_convert(None)

    log.info(f"    Rohdaten: {len(df_raw)} Matches")

    # Features neu berechnen
    log.info("    Berechne Features ...")
    FEATURES_CSV.unlink(missing_ok=True)   # Cache löschen
    from utils.features import build_features
    df_feat, elo_ratings = build_features(df_raw, save=True)

    log.info(f"    Features: {len(df_feat)} Matches")

    # Modell trainieren
    from models.trainer import train, train_map_models, save_model, backtest
    model, metrics, _, val_df = train(df_feat)

    log.info(f"    Val Accuracy: {metrics['val_accuracy']:.1%} | "
             f"AUC: {metrics.get('val_auc', float('nan')):.3f}")

    # Map-Modelle
    map_models = train_map_models(df_feat)

    # Backtest speichern
    bt = backtest(model, val_df)
    bt.to_csv(DATA_DIR / "backtest_results.csv", index=False)

    # Modell speichern
    save_model(model, elo_ratings, map_models)

    # Streamlit Cache invalidieren (Datei-Timestamp ändern)
    _touch_reload_flag()


def _touch_reload_flag():
    """Schreibt eine Timestamp-Datei — das Dashboard erkennt sie und lädt neu."""
    flag = DATA_DIR / ".model_updated"
    flag.write_text(datetime.now(timezone.utc).isoformat())


# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def _count_csv(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return sum(1 for _ in open(path)) - 1   # Minus Header
    except Exception:
        return 0


def _save_last_run(stats: dict):
    """Speichert den letzten Update-Status als JSON."""
    import json
    status_file = DATA_DIR / "updater_status.json"
    try:
        status_file.write_text(json.dumps(stats, indent=2))
    except Exception:
        pass


def _load_last_run() -> dict:
    import json
    status_file = DATA_DIR / "updater_status.json"
    if status_file.exists():
        try:
            return json.loads(status_file.read_text())
        except Exception:
            pass
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Watch-Mode: Dauerschleife
# ─────────────────────────────────────────────────────────────────────────────

def run_watch(interval_hours: float = 6.0, days_back: int = 3):
    """
    Läuft dauerhaft und führt alle `interval_hours` Stunden einen Update durch.
    Starte mit: python auto_updater.py --watch --interval 6
    """
    log.info(f"Watch-Mode gestartet — Update alle {interval_hours}h")
    log.info("Stoppe mit Ctrl+C")

    while True:
        try:
            run_update(days_back=days_back)
        except KeyboardInterrupt:
            log.info("Watch-Mode beendet.")
            break
        except Exception as e:
            log.error(f"Unerwarteter Fehler im Update-Zyklus: {e}")

        next_run = datetime.now()
        next_run_ts = next_run.timestamp() + interval_hours * 3600
        next_str = datetime.fromtimestamp(next_run_ts).strftime("%H:%M:%S")
        log.info(f"Nächster Update: {next_str} (in {interval_hours}h)")

        try:
            time.sleep(interval_hours * 3600)
        except KeyboardInterrupt:
            log.info("Watch-Mode beendet.")
            break


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CS2 Auto-Updater — holt Matches & trainiert Modell nach",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python auto_updater.py                        Einmal updaten
  python auto_updater.py --force-retrain        Update + erzwungenes Re-Training
  python auto_updater.py --watch                Alle 6h automatisch updaten
  python auto_updater.py --watch --interval 12  Alle 12h updaten
  python auto_updater.py --status               Letzten Update-Status anzeigen
        """
    )
    parser.add_argument("--watch",         action="store_true",
                        help="Dauerhaft im Hintergrund laufen")
    parser.add_argument("--interval",      type=float, default=6.0,
                        help="Update-Intervall in Stunden (default: 6)")
    parser.add_argument("--days-back",     type=int,   default=3,
                        help="Wie viele Tage zurück nach Matches suchen (default: 3)")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Modell auch ohne neue Matches neu trainieren")
    parser.add_argument("--status",        action="store_true",
                        help="Status des letzten Runs anzeigen")
    args = parser.parse_args()

    if args.status:
        import json
        s = _load_last_run()
        if s:
            print(json.dumps(s, indent=2))
        else:
            print("Noch kein Update-Status gefunden.")

    elif args.watch:
        run_watch(interval_hours=args.interval, days_back=args.days_back)

    else:
        stats = run_update(days_back=args.days_back, force_retrain=args.force_retrain)
        if stats["errors"]:
            print(f"\n⚠️  {len(stats['errors'])} Fehler:")
            for e in stats["errors"]:
                print(f"   - {e}")
        sys.exit(1 if stats["errors"] and not stats["retrained"] else 0)
