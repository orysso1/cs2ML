#!/usr/bin/env python3
# train.py — Haupt-Pipeline
#
# python train.py --csv data/matches.csv          # Kaggle-CSV
# python train.py --csv data/matches.csv --tune   # + Hyperparameter-Tuning
# python train.py --csv data/matches.csv --no-map # Ohne Map-Modelle

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from config import RAW_CSV, FEATURES_CSV
from data.kaggle_loader import load_custom_csv
from utils.features import build_features
from models.trainer import (
    train, train_map_models, save_model,
    get_feature_importance, backtest,
    tune_hyperparameters
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("train.log")],
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="CS2 Match Prediction — Training")
    parser.add_argument("--csv",      type=str,            help="Kaggle-CSV Pfad")
    parser.add_argument("--tune",     action="store_true", help="Optuna Hyperparameter-Tuning")
    parser.add_argument("--trials",   type=int, default=30)
    parser.add_argument("--no-map",   action="store_true", help="Keine Map-Modelle trainieren")
    parser.add_argument("--rebuild",  action="store_true", help="Features neu berechnen")
    args = parser.parse_args()

    # ── 1. Rohdaten ───────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("SCHRITT 1: Daten laden")
    log.info("=" * 60)

    if args.csv:
        df_raw = load_custom_csv(args.csv)
        # CSV neu geladen → Features-Cache immer invalidieren
        args.rebuild = True
        if FEATURES_CSV.exists():
            FEATURES_CSV.unlink()
            log.info("Features-Cache gelöscht (neues CSV geladen)")
    elif RAW_CSV.exists():
        log.info(f"Nutze gecachten Raw-CSV: {RAW_CSV}")
        df_raw = pd.read_csv(RAW_CSV, parse_dates=["date"])
    else:
        log.error("Kein CSV angegeben und kein Cache. Nutze: --csv dein_dataset.csv")
        sys.exit(1)

    log.info(f"Rohdaten: {len(df_raw)} Matches | {df_raw['team_a'].nunique()} Teams")
    log.info(f"Zeitraum: {df_raw['date'].min().date()} – {df_raw['date'].max().date()}")

    # ── 2. Features ───────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("SCHRITT 2: Feature Engineering")
    log.info("=" * 60)

    if FEATURES_CSV.exists() and not args.rebuild:
        log.info(f"Nutze gecachte Features: {FEATURES_CSV}")
        df_feat = pd.read_csv(FEATURES_CSV, parse_dates=["date"])
        from utils.features import compute_elo
        _, elo_ratings = compute_elo(df_raw)
    else:
        df_feat, elo_ratings = build_features(df_raw, save=True)

    log.info(f"Features: {len(df_feat)} Matches, {len(df_feat.columns)} Spalten")

    # ── 3. Hyperparameter-Tuning ──────────────────────────────────────────────
    best_params = {}
    if args.tune:
        log.info("=" * 60)
        log.info("SCHRITT 3: Hyperparameter-Tuning")
        log.info("=" * 60)
        best_params = tune_hyperparameters(df_feat, args.trials)

    # ── 4. Match-Winner Modell ────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("SCHRITT 4: Match-Winner Training")
    log.info("=" * 60)
    model, metrics, train_df, val_df = train(df_feat, best_params)

    # ── 5. Map-Modelle ────────────────────────────────────────────────────────
    map_models = {}
    if not args.no_map:
        log.info("=" * 60)
        log.info("SCHRITT 5: Map-Prediction Training")
        log.info("=" * 60)
        map_models = train_map_models(df_feat, best_params)

    # ── 6. Evaluation ─────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("SCHRITT 6: Evaluation")
    log.info("=" * 60)
    bt = backtest(model, val_df)
    bt.to_csv("data/backtest_results.csv", index=False)

    log.info("Accuracy nach Konfidenz:")
    for t in [0.2, 0.4, 0.6]:
        sub = bt[bt["confidence"] >= t]
        if len(sub) > 0:
            log.info(f"  >= {t:.0%}: {sub['correct'].mean():.1%} ({len(sub)} Matches)")

    fi = get_feature_importance(model)
    log.info("\nTop-10 Feature Importance:")
    for _, row in fi.head(10).iterrows():
        bar = "█" * int(row["importance"] * 60)
        log.info(f"  {row['feature']:<30} {row['importance']:.4f} {bar}")

    # ── 7. Speichern ──────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("SCHRITT 7: Speichern")
    log.info("=" * 60)
    save_model(model, elo_ratings, map_models if map_models else None)

    log.info("")
    log.info("╔══════════════════════════════════════════╗")
    log.info("║          TRAINING ABGESCHLOSSEN           ║")
    log.info("╠══════════════════════════════════════════╣")
    log.info(f"║  Trainings-Matches : {metrics['n_train']:>6}              ║")
    log.info(f"║  Validierungs-M.   : {metrics['n_val']:>6}              ║")
    log.info(f"║  Train Accuracy    : {metrics['train_accuracy']:>6.1%}              ║")
    log.info(f"║  Val Accuracy      : {metrics['val_accuracy']:>6.1%}              ║")
    log.info(f"║  Val AUC           : {metrics['val_auc']:>6.3f}              ║")
    log.info(f"║  Map-Modelle       : {len(map_models):>6}              ║")
    log.info("╚══════════════════════════════════════════╝")
    log.info("→ streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()