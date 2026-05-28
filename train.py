#!/usr/bin/env python3
# train.py — Haupt-Pipeline mit allen Verbesserungen
#
# Schnellstart:
#   python train.py --csv data.csv                  Standard
#   python train.py --csv data.csv --ensemble        Stacking Ensemble
#   python train.py --csv data.csv --tune            Hyperparameter-Tuning
#   python train.py --csv data.csv --walk-forward    Walk-Forward-Validation
#   python train.py --csv data.csv --full            Alles auf einmal

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from config import RAW_CSV, FEATURES_CSV, CALIBRATED_PATH
from data.kaggle_loader import load_custom_csv
from utils.features import build_features
from models.trainer import (
    train, train_ensemble, train_map_models, train_tier_models,
    calibrate_model, walk_forward_eval,
    save_model, get_feature_importance, backtest,
    tune_hyperparameters, time_split
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("train.log")],
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="CS2 Prediction — Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modi:
  Standard (schnell):
    python train.py --csv data.csv

  Beste Qualität (langsam, ~30 Min):
    python train.py --csv data.csv --full

  Einzelne Optionen:
    --ensemble      Stacking Ensemble statt einzelnem XGBoost
    --calibrate     Wahrscheinlichkeiten kalibrieren (wichtig für Value Betting)
    --walk-forward  Realistische Validation über 6 Zeitfenster
    --tier-models   Separate Modelle für Tier1/Major
    --tune          Hyperparameter-Tuning via Optuna
    --no-map        Keine Map-Modelle (schneller)
        """
    )
    parser.add_argument("--csv",          type=str)
    parser.add_argument("--rebuild",      action="store_true")
    parser.add_argument("--ensemble",     action="store_true",
                        help="Stacking Ensemble (XGB+RF+LGBM)")
    parser.add_argument("--calibrate",    action="store_true",
                        help="Kalibriere Wahrscheinlichkeiten")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Walk-Forward-Validation")
    parser.add_argument("--tier-models",  action="store_true",
                        help="Separate Tier1/Major-Modelle")
    parser.add_argument("--tune",         action="store_true")
    parser.add_argument("--trials",       type=int, default=30)
    parser.add_argument("--no-map",       action="store_true")
    parser.add_argument("--full",         action="store_true",
                        help="Alle Optionen aktivieren")
    args = parser.parse_args()

    # --full aktiviert alles
    if args.full:
        args.ensemble = args.calibrate = args.walk_forward = \
        args.tier_models = args.tune = True

    # ── 1. Daten ──────────────────────────────────────────────────────────────
    log.info("=" * 58)
    log.info("SCHRITT 1: Daten laden")
    log.info("=" * 58)

    if args.csv:
        df_raw = load_custom_csv(args.csv)
        args.rebuild = True
        FEATURES_CSV.unlink(missing_ok=True)
    elif RAW_CSV.exists():
        log.info(f"Nutze: {RAW_CSV}")
        df_raw = pd.read_csv(RAW_CSV, parse_dates=["date"])
    else:
        log.error("Kein CSV. Nutze: --csv dein_dataset.csv")
        sys.exit(1)

    log.info(f"Matches: {len(df_raw)} | Teams: {df_raw['team_a'].nunique()} | "
             f"Zeitraum: {df_raw['date'].min().date()} – {df_raw['date'].max().date()}")

    # ── 2. Features ───────────────────────────────────────────────────────────
    log.info("=" * 58)
    log.info("SCHRITT 2: Feature Engineering")
    log.info("=" * 58)

    if FEATURES_CSV.exists() and not args.rebuild:
        log.info(f"Cache: {FEATURES_CSV}")
        df_feat = pd.read_csv(FEATURES_CSV, parse_dates=["date"])
        from utils.features import compute_elo
        _, elo_ratings = compute_elo(df_raw)
    else:
        df_feat, elo_ratings = build_features(df_raw, save=True)

    log.info(f"Features: {len(df_feat)} Matches, {len(df_feat.columns)} Spalten")

    # ── 3. Hyperparameter-Tuning ──────────────────────────────────────────────
    best_params = {}
    if args.tune:
        log.info("=" * 58)
        log.info("SCHRITT 3: Hyperparameter-Tuning")
        log.info("=" * 58)
        best_params = tune_hyperparameters(df_feat, args.trials)

    # ── 4. Walk-Forward-Validation ────────────────────────────────────────────
    if args.walk_forward:
        log.info("=" * 58)
        log.info("SCHRITT 4: Walk-Forward-Validation")
        log.info("=" * 58)
        wf_results = walk_forward_eval(df_feat, best_params)
        wf_results.to_csv("data/walk_forward_results.csv", index=False)
        log.info(f"\n{wf_results.to_string(index=False)}")
        log.info(f"\nØ Walk-Forward Accuracy: {wf_results['accuracy'].mean():.3f} "
                 f"± {wf_results['accuracy'].std():.3f}")

    # ── 5. Haupt-Training ─────────────────────────────────────────────────────
    log.info("=" * 58)
    log.info("SCHRITT 5: Modell trainieren")
    log.info("=" * 58)

    ensemble_model = None
    if args.ensemble:
        log.info("Trainiere Stacking Ensemble ...")
        main_model, metrics, train_df, val_df = train_ensemble(df_feat, best_params)
        # Zusätzlich einzelnes XGBoost für schnelle Predictions
        xgb_model, _, _, _ = train(df_feat, best_params)
    else:
        main_model, metrics, train_df, val_df = train(df_feat, best_params)
        xgb_model = main_model

    # ── 6. Kalibrierung ───────────────────────────────────────────────────────
    calibrated_model = None
    if args.calibrate:
        log.info("=" * 58)
        log.info("SCHRITT 6: Kalibrierung")
        log.info("=" * 58)
        available = getattr(xgb_model, "_feature_names",
                            [f for f in xgb_model._feature_names if f in df_feat.columns])
        X_val = val_df[[f for f in available if f in val_df.columns]].fillna(0).values
        y_val = val_df["team_a_won"].values
        calibrated_model = calibrate_model(xgb_model, X_val, y_val)

    # ── 7. Map-Modelle ────────────────────────────────────────────────────────
    map_models = {}
    if not args.no_map:
        log.info("=" * 58)
        log.info("SCHRITT 7: Map-Modelle")
        log.info("=" * 58)
        map_models = train_map_models(df_feat, best_params)

    # ── 8. Tier-Modelle ───────────────────────────────────────────────────────
    tier_models = {}
    if args.tier_models:
        log.info("=" * 58)
        log.info("SCHRITT 8: Tier-spezifische Modelle")
        log.info("=" * 58)
        tier_models = train_tier_models(df_feat, best_params)
        if tier_models:
            import joblib
            joblib.dump(tier_models, "models/tier_models.joblib")
            log.info(f"Tier-Modelle gespeichert: {list(tier_models.keys())}")

    # ── 9. Evaluation ─────────────────────────────────────────────────────────
    log.info("=" * 58)
    log.info("SCHRITT 9: Evaluation")
    log.info("=" * 58)

    use_model = calibrated_model if calibrated_model else main_model
    bt = backtest(use_model, val_df)
    bt.to_csv("data/backtest_results.csv", index=False)

    log.info("Accuracy nach Konfidenz:")
    for t in [0.2, 0.4, 0.6]:
        sub = bt[bt["confidence"] >= t]
        if len(sub) > 0:
            log.info(f"  >= {t:.0%}: {sub['correct'].mean():.1%} ({len(sub)} Matches)")

    fi = get_feature_importance(xgb_model)
    log.info("\nTop-10 Features:")
    for _, row in fi.head(10).iterrows():
        bar = "█" * int(row["importance"] * 60)
        log.info(f"  {row['feature']:<30} {row['importance']:.4f} {bar}")

    # Betting-Simulation
    avg_odds = 1.9
    bt["ev"] = bt["prob_a"] * avg_odds - 1
    value_bets = bt[bt["ev"] > 0.05]
    if len(value_bets) > 0:
        roi = (value_bets["correct"] * (avg_odds - 1) - (1 - value_bets["correct"])).mean()
        log.info(f"\nBetting-Simulation (EV>5%, Ø Quote {avg_odds}):")
        log.info(f"  Wetten:   {len(value_bets)}")
        log.info(f"  Accuracy: {value_bets['correct'].mean():.1%}")
        log.info(f"  ROI:      {roi*100:+.1f}%")

    # ── 10. Speichern ──────────────────────────────────────────────────────────
    log.info("=" * 58)
    log.info("SCHRITT 10: Speichern")
    log.info("=" * 58)
    save_model(
        model       = xgb_model,
        elo_ratings = elo_ratings,
        map_models  = map_models if map_models else None,
        calibrated  = calibrated_model,
        ensemble    = main_model if args.ensemble else None,
    )

    log.info("")
    log.info("╔══════════════════════════════════════════════╗")
    log.info("║           TRAINING ABGESCHLOSSEN              ║")
    log.info("╠══════════════════════════════════════════════╣")
    log.info(f"║  Trainings-Matches  : {metrics['n_train']:>6}                ║")
    log.info(f"║  Val-Matches        : {metrics['n_val']:>6}                ║")
    log.info(f"║  Val Accuracy       : {metrics['val_accuracy']:>6.1%}                ║")
    log.info(f"║  Val AUC            : {metrics['val_auc']:>6.3f}                ║")
    log.info(f"║  Kalibriert         : {'  Ja' if calibrated_model else ' Nein'}                  ║")
    log.info(f"║  Ensemble           : {'  Ja' if args.ensemble else ' Nein'}                  ║")
    log.info(f"║  Map-Modelle        : {len(map_models):>6}                ║")
    log.info(f"║  Tier-Modelle       : {len(tier_models):>6}                ║")
    log.info("╚══════════════════════════════════════════════╝")
    log.info("→ streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()