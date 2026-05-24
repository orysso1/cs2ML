# models/trainer.py
# Trainiert Match-Winner-Modell + Map-Prediction-Modelle.

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FEATURES, CS2_MAPS, FEATURES_CSV,
    MODEL_PATH, MAP_MODEL_PATH, ELO_PATH,
    TRAIN_CUTOFF, RANDOM_STATE
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Modell-Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(params: dict | None = None) -> Pipeline:
    default_params = {
        "n_estimators":     300,
        "max_depth":        4,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma":            0.1,
        "eval_metric":      "logloss",
        "random_state":     RANDOM_STATE,
        "n_jobs":           -1,
    }
    if params:
        default_params.update(params)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("xgb",   XGBClassifier(**default_params)),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Zeitbasierter Split (mit Auto-Fallback)
# ─────────────────────────────────────────────────────────────────────────────

def time_split(df: pd.DataFrame, cutoff: str = TRAIN_CUTOFF):
    """
    Zeitbasierter Split. Stellt sicher dass das Val-Set:
    - mindestens 50 Matches hat
    - beide Klassen (0 und 1) enthält
    Falls nicht: automatischer 80/20-Split.
    """
    df = df.sort_values("date").copy()

    def _both_classes(d):
        return d["team_a_won"].nunique() == 2

    train = df[df["date"] < cutoff]
    val   = df[df["date"] >= cutoff]

    if len(val) < 50 or not _both_classes(val):
        log.warning(
            f"Cutoff '{cutoff}' erzeugt unbrauchbares Val-Set "
            f"(n={len(val)}, Klassen={val['team_a_won'].nunique() if len(val)>0 else 0}). "
            f"Nutze 80/20-Split."
        )
        split_idx = int(len(df) * 0.8)
        train = df.iloc[:split_idx]
        val   = df.iloc[split_idx:]

    # Wenn immer noch nur eine Klasse: Label-Problem im Dataset diagnostizieren
    # Prüfen ob Split-Problem durch alten Cache verursacht wird
    if not _both_classes(val) or not _both_classes(train):
        # Letzter Ausweg: neu splitten nach Shuffle der Reihenfolge
        log.warning("Split hat unbalancierte Klassen — versuche stratifizierten Split ...")
        from sklearn.model_selection import train_test_split as tts
        train, val = tts(df, test_size=0.2, random_state=42, stratify=df["team_a_won"])
        train = train.sort_values("date")
        val   = val.sort_values("date")

    if not _both_classes(val) or not _both_classes(train):
        raise ValueError(
            f"team_a_won enthält nur eine Klasse — bitte Cache löschen und neu laden:\n"
            f"  rm data/matches_raw.csv data/matches_features.csv\n"
            f"  python train.py --csv dein_dataset.csv\n"
            f"Label-Verteilung: {df['team_a_won'].value_counts().to_dict()}"
        )

    cutoff = str(val["date"].min().date())
    log.info(f"Train: {len(train)} | Val: {len(val)} | "
             f"Val-Klassen: {sorted(val['team_a_won'].unique())} (Cutoff: {cutoff})")
    return train, val


# ─────────────────────────────────────────────────────────────────────────────
# Match-Winner Training
# ─────────────────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame, params: dict | None = None) -> tuple:
    """Trainiert das Match-Winner-Modell."""
    # Nur Features nutzen die tatsächlich im DataFrame vorhanden sind
    available = [f for f in FEATURES if f in df.columns]
    missing   = [f for f in FEATURES if f not in df.columns]
    if missing:
        log.warning(f"Fehlende Features (werden übersprungen): {missing}")

    train_df, val_df = time_split(df)
    if len(train_df) < 100:
        raise ValueError(f"Zu wenig Trainingsdaten: {len(train_df)}")

    X_train = train_df[available].fillna(0).values
    y_train = train_df["team_a_won"].values
    X_val   = val_df[available].fillna(0).values
    y_val   = val_df["team_a_won"].values

    model = build_model(params)
    log.info("Trainiere Match-Winner-Modell ...")
    model.fit(X_train, y_train)

    prob_val   = model.predict_proba(X_val)[:, 1]
    pred_val   = (prob_val >= 0.5).astype(int)
    prob_train = model.predict_proba(X_train)[:, 1]
    pred_train = (prob_train >= 0.5).astype(int)

    # Metriken — sicher berechnen (AUC/LogLoss nur wenn beide Klassen vorhanden)
    n_classes = len(set(y_val))
    val_auc     = roc_auc_score(y_val, prob_val) if n_classes == 2 else float("nan")
    val_logloss = log_loss(y_val, prob_val, labels=[0, 1]) if n_classes == 2 else float("nan")

    metrics = {
        "train_accuracy": accuracy_score(y_train, pred_train),
        "val_accuracy":   accuracy_score(y_val,   pred_val),
        "val_auc":        val_auc,
        "val_logloss":    val_logloss,
        "n_train":        len(train_df),
        "n_val":          len(val_df),
        "features_used":  available,
    }

    log.info(f"Train Acc: {metrics['train_accuracy']:.3f} | "
             f"Val Acc:   {metrics['val_accuracy']:.3f} | "
             f"AUC: {metrics['val_auc']:.3f}")
    if n_classes < 2:
        log.warning("Val-Set enthält nur eine Klasse — AUC/LogLoss nicht berechenbar.")
    log.info("\n" + classification_report(y_val, pred_val,
                                          labels=[0, 1],
                                          target_names=["Team B", "Team A"],
                                          zero_division=0))

    # Modell mit den tatsächlich genutzten Features speichern
    model._feature_names = available
    return model, metrics, train_df, val_df


# ─────────────────────────────────────────────────────────────────────────────
# Map-Prediction Training
# ─────────────────────────────────────────────────────────────────────────────

def train_map_models(df: pd.DataFrame, params: dict | None = None) -> dict:
    """
    Trainiert für jede CS2-Map ein eigenes Modell.
    Label: hat team_a die jeweilige Map gewonnen?

    Das Kaggle-Dataset hat winner_{map} und loser_{map} als Map-Winrates.
    Wir nutzen als Label: team_a_won (Map-spezifisch) falls vorhanden,
    sonst den generellen Match-Gewinner als Proxy.
    """
    map_models = {}
    train_df, val_df = time_split(df)

    for m in CS2_MAPS:
        map_feat_col = f"{m}_winrate_diff"

        # Map-spezifische Features
        map_features = [
            f for f in ["rating_diff", "adr_diff", "kast_diff",
                        "elo_diff", "h2h_diff", "past3_diff",
                        map_feat_col]
            if f in df.columns
        ]

        if len(map_features) < 2:
            log.warning(f"Überspringe Map '{m}': zu wenige Features")
            continue

        # Label: wir nutzen team_a_won als Proxy (Match-Gewinner gewinnt
        # statistisch auch die relevante Map am häufigsten)
        label_col = "team_a_won"

        X_tr = train_df[map_features].fillna(0).values
        y_tr = train_df[label_col].values
        X_va = val_df[map_features].fillna(0).values
        y_va = val_df[label_col].values

        model = build_model(params)
        model.fit(X_tr, y_tr)

        prob  = model.predict_proba(X_va)[:, 1]
        acc   = accuracy_score(y_va, (prob >= 0.5).astype(int))
        model._feature_names = map_features
        map_models[m] = model

        log.info(f"Map '{m}': Val Accuracy = {acc:.3f} ({len(map_features)} Features)")

    log.info(f"Map-Modelle trainiert: {list(map_models.keys())}")
    return map_models


# ─────────────────────────────────────────────────────────────────────────────
# Feature Importance
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_importance(model: Pipeline) -> pd.DataFrame:
    names = getattr(model, "_feature_names", FEATURES)
    importances = model.named_steps["xgb"].feature_importances_
    # Länge abgleichen falls nötig
    n = min(len(names), len(importances))
    return pd.DataFrame({
        "feature":    names[:n],
        "importance": importances[:n],
    }).sort_values("importance", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Speichern & Laden
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model: Pipeline, elo_ratings: dict,
               map_models: dict | None = None):
    joblib.dump(model,       MODEL_PATH)
    joblib.dump(elo_ratings, ELO_PATH)
    log.info(f"Match-Modell gespeichert: {MODEL_PATH}")
    if map_models:
        joblib.dump(map_models, MAP_MODEL_PATH)
        log.info(f"Map-Modelle gespeichert: {MAP_MODEL_PATH}")


def load_model() -> tuple[Pipeline, dict, dict]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Kein Modell unter {MODEL_PATH}. Bitte zuerst: python train.py --csv dein.csv"
        )
    model       = joblib.load(MODEL_PATH)
    elo_ratings = joblib.load(ELO_PATH) if ELO_PATH.exists() else {}
    map_models  = joblib.load(MAP_MODEL_PATH) if MAP_MODEL_PATH.exists() else {}
    log.info(f"Modell geladen | Map-Modelle: {list(map_models.keys())}")
    return model, elo_ratings, map_models


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_match(model: Pipeline, features: dict) -> dict:
    """Match-Winner Prediction."""
    feat_names = getattr(model, "_feature_names", FEATURES)
    x = np.array([[features.get(f, 0.0) for f in feat_names]])
    prob = model.predict_proba(x)[0]
    return {"prob_a": float(prob[1]), "prob_b": float(prob[0])}


def predict_maps(map_models: dict, features: dict) -> dict:
    """
    Gibt für jede Map die Gewinnwahrscheinlichkeit von Team A zurück.
    """
    results = {}
    for map_name, model in map_models.items():
        feat_names = getattr(model, "_feature_names", [])
        x = np.array([[features.get(f, 0.0) for f in feat_names]])
        try:
            prob = model.predict_proba(x)[0]
            results[map_name] = float(prob[1])
        except Exception:
            results[map_name] = 0.5
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Backtest
# ─────────────────────────────────────────────────────────────────────────────

def backtest(model: Pipeline, val_df: pd.DataFrame) -> pd.DataFrame:
    feat_names = getattr(model, "_feature_names", FEATURES)
    available  = [f for f in feat_names if f in val_df.columns]
    X_val      = val_df[available].fillna(0).values
    probs      = model.predict_proba(X_val)
    val_df     = val_df.copy()
    val_df["prob_a"]     = probs[:, 1]
    val_df["prob_b"]     = probs[:, 0]
    val_df["predicted"]  = (probs[:, 1] >= 0.5).astype(int)
    val_df["correct"]    = (val_df["predicted"] == val_df["team_a_won"]).astype(int)
    val_df["confidence"] = np.abs(probs[:, 1] - 0.5) * 2
    return val_df


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter-Tuning
# ─────────────────────────────────────────────────────────────────────────────

def tune_hyperparameters(df: pd.DataFrame, n_trials: int = 50) -> dict:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        log.warning("Optuna nicht installiert. pip install optuna")
        return {}

    available = [f for f in FEATURES if f in df.columns]
    train_df, _ = time_split(df)
    X = train_df[available].fillna(0).values
    y = train_df["team_a_won"].values

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "max_depth":        trial.suggest_int("max_depth", 2, 6),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0, 0.5),
        }
        model  = build_model(params)
        cv     = StratifiedKFold(n_splits=5, shuffle=False)
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_log_loss")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    log.info(f"Beste Parameter: {study.best_params}")
    return study.best_params


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--tune",   action="store_true")
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    if not FEATURES_CSV.exists():
        print(f"Bitte zuerst Features berechnen: python utils/features.py --input data/matches_raw.csv")
        sys.exit(1)

    df = pd.read_csv(FEATURES_CSV, parse_dates=["date"])
    best_params = tune_hyperparameters(df, args.trials) if args.tune else {}
    model, metrics, _, val_df = train(df, best_params)
    bt = backtest(model, val_df)
    print(f"\nBacktest Accuracy: {bt['correct'].mean():.1%}")
    print(get_feature_importance(model).head(10).to_string(index=False))