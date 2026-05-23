#!/usr/bin/env python3
# train.py
# Führt die komplette Pipeline aus:
# Daten laden → Features berechnen → Modell trainieren → Speichern
#
# Nutzung:
#   python train.py                    # Nutze gecachte Daten
#   python train.py --demo             # Demo-Daten generieren
#   python train.py --scrape           # Frisch von HLTV scrapen
#   python train.py --csv mein.csv     # Eigenes CSV laden
#   python train.py --tune             # Mit Hyperparameter-Tuning

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from config import RAW_CSV, FEATURES_CSV
from data.kaggle_loader import generate_demo_data, load_custom_csv
from data.scraper import load_or_scrape
from utils.features import build_features
from models.trainer import train, save_model, get_feature_importance, backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train.log"),
    ]
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="CS2 Match Prediction — Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python train.py --demo           Demo-Daten generieren und Modell trainieren
  python train.py --scrape         Von HLTV scrapen (langsam, ~20 Minuten)
  python train.py --csv data.csv   Kaggle-CSV oder eigenes CSV nutzen
  python train.py --tune           Mit Optuna Hyperparameter-Tuning (empfohlen)
        """
    )
    parser.add_argument("--demo",   action="store_true", help="Synthetische Demo-Daten")
    parser.add_argument("--scrape", action="store_true", help="Von HLTV scrapen")
    parser.add_argument("--csv",    type=str,            help="Lokales CSV laden")
    parser.add_argument("--tune",   action="store_true", help="Hyperparameter-Tuning")
    parser.add_argument("--trials", type=int, default=30, help="Optuna Trials (default: 30)")
    parser.add_argument("--rebuild-features", action="store_true",
                        help="Features neu berechnen auch wenn Cache existiert")
    args = parser.parse_args()

    # ── 1. Rohdaten ───────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("SCHRITT 1: Daten laden")
    log.info("=" * 60)

    if args.demo:
        log.info("Generiere synthetische Demo-Daten ...")
        df_raw = generate_demo_data(n_matches=3000)

    elif args.csv:
        log.info(f"Lade CSV: {args.csv}")
        df_raw = load_custom_csv(args.csv)

    elif args.scrape:
        log.info("Scrape von HLTV (kann 10–20 Minuten dauern) ...")
        df_raw = load_or_scrape(force_scrape=True)

    elif RAW_CSV.exists():
        log.info(f"Nutze gecachte Rohdaten: {RAW_CSV}")
        df_raw = pd.read_csv(RAW_CSV, parse_dates=["date"])

    else:
        log.warning("Keine Datenquelle angegeben und kein Cache vorhanden.")
        log.warning("Nutze Demo-Daten als Fallback. Für echte Daten: --scrape oder --csv")
        df_raw = generate_demo_data(n_matches=3000)

    log.info(f"Rohdaten: {len(df_raw)} Matches, {df_raw['team_a'].nunique()} Teams")

    # ── 2. Feature Engineering ────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("SCHRITT 2: Feature Engineering")
    log.info("=" * 60)

    rebuild = args.rebuild_features or not FEATURES_CSV.exists()

    if not rebuild:
        log.info(f"Nutze gecachte Features: {FEATURES_CSV}")
        df_feat = pd.read_csv(FEATURES_CSV, parse_dates=["date"])
        # ELO neu berechnen für Predictions
        from utils.features import compute_elo
        _, elo_ratings = compute_elo(df_raw)
    else:
        df_feat, elo_ratings = build_features(df_raw, save=True)

    log.info(f"Features bereit: {len(df_feat)} Matches")

    # ── 3. Hyperparameter-Tuning ─────────────────────────────────────────────
    best_params = {}
    if args.tune:
        log.info("=" * 60)
        log.info("SCHRITT 3: Hyperparameter-Tuning (Optuna)")
        log.info("=" * 60)
        from models.trainer import tune_hyperparameters
        best_params = tune_hyperparameters(df_feat, n_trials=args.trials)

    # ── 4. Training ───────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("SCHRITT 4: Modell trainieren")
    log.info("=" * 60)

    model, metrics, train_df, val_df = train(df_feat, params=best_params)

    # ── 5. Evaluation ─────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("SCHRITT 5: Evaluation")
    log.info("=" * 60)

    bt = backtest(model, val_df)

    # Accuracy nach Confidence-Bracket
    log.info("Accuracy nach Confidence-Level:")
    for threshold in [0.3, 0.5, 0.7]:
        high_conf = bt[bt["confidence"] >= threshold]
        if len(high_conf) > 0:
            acc = high_conf["correct"].mean()
            log.info(f"  Confidence >= {threshold:.0%}: {acc:.1%} ({len(high_conf)} Matches)")

    fi = get_feature_importance(model)
    log.info("\nFeature Importance (Top 5):")
    for _, row in fi.head(5).iterrows():
        bar = "█" * int(row["importance"] * 50)
        log.info(f"  {row['feature']:<25} {row['importance']:.4f} {bar}")

    # ── 6. Speichern ──────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("SCHRITT 6: Modell speichern")
    log.info("=" * 60)

    save_model(model, elo_ratings)

    # Backtest-Ergebnisse speichern
    bt_path = Path("data/backtest_results.csv")
    bt.to_csv(bt_path, index=False)
    log.info(f"Backtest-Ergebnisse: {bt_path}")

    # ── Zusammenfassung ───────────────────────────────────────────────────────
    log.info("")
    log.info("╔══════════════════════════════════════╗")
    log.info("║         TRAINING ABGESCHLOSSEN        ║")
    log.info("╠══════════════════════════════════════╣")
    log.info(f"║  Trainings-Matches: {metrics['n_train']:>6}            ║")
    log.info(f"║  Validierungs-M.:   {metrics['n_val']:>6}            ║")
    log.info(f"║  Train Accuracy:    {metrics['train_accuracy']:>6.1%}            ║")
    log.info(f"║  Val Accuracy:      {metrics['val_accuracy']:>6.1%}            ║")
    log.info(f"║  Val AUC:           {metrics['val_auc']:>6.3f}            ║")
    log.info(f"║  Val Log-Loss:      {metrics['val_logloss']:>6.3f}            ║")
    log.info("╚══════════════════════════════════════╝")
    log.info("")
    log.info("Nächster Schritt: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()