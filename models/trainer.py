# models/trainer.py
# Trainiert, evaluiert und speichert das XGBoost-Modell.

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FEATURES, FEATURES_CSV, MODEL_PATH, ELO_PATH, TRAIN_CUTOFF, RANDOM_STATE

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Modell erstellen
# ─────────────────────────────────────────────────────────────────────────────

def build_model(params: dict | None = None) -> Pipeline:
    """
    Gibt eine sklearn-Pipeline zurück:
    StandardScaler → XGBoostClassifier
    """
    default_params = {
        "n_estimators":   300,
        "max_depth":      4,
        "learning_rate":  0.05,
        "subsample":      0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma":          0.1,
        "eval_metric":    "logloss",
        "use_label_encoder": False,
        "random_state":   RANDOM_STATE,
        "n_jobs":         -1,
    }
    if params:
        default_params.update(params)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb",   XGBClassifier(**default_params)),
    ])
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Zeitbasierter Train/Val-Split
# ─────────────────────────────────────────────────────────────────────────────

def time_split(df: pd.DataFrame, cutoff: str = TRAIN_CUTOFF):
    """
    Teilt Daten strikt zeitbasiert.
    WICHTIG: Kein zufälliger Split — würde Data Leakage verursachen!
    """
    train = df[df["date"] < cutoff]
    val   = df[df["date"] >= cutoff]
    log.info(f"Train: {len(train)} Matches (vor {cutoff})")
    log.info(f"Val:   {len(val)} Matches (ab {cutoff})")
    return train, val


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame, params: dict | None = None) -> tuple:
    """
    Trainiert das Modell auf historischen Daten.

    Returns:
        model:    Trainierte Pipeline
        metrics:  Dict mit Accuracy, AUC, Log-Loss
        train_df: Trainings-DataFrame
        val_df:   Validierungs-DataFrame
    """
    train_df, val_df = time_split(df)

    if len(train_df) < 100:
        raise ValueError(f"Zu wenig Trainingsdaten: {len(train_df)}. Mindestens 100 nötig.")

    X_train = train_df[FEATURES].values
    y_train = train_df["team_a_won"].values
    X_val   = val_df[FEATURES].values
    y_val   = val_df["team_a_won"].values

    model = build_model(params)

    log.info("Trainiere Modell ...")
    model.fit(X_train, y_train)

    # ── Metriken ─────────────────────────────────────────────────────────────
    prob_val  = model.predict_proba(X_val)[:, 1]
    pred_val  = (prob_val >= 0.5).astype(int)

    prob_train = model.predict_proba(X_train)[:, 1]
    pred_train = (prob_train >= 0.5).astype(int)

    metrics = {
        "train_accuracy": accuracy_score(y_train, pred_train),
        "val_accuracy":   accuracy_score(y_val,   pred_val),
        "val_auc":        roc_auc_score(y_val,    prob_val),
        "val_logloss":    log_loss(y_val,          prob_val),
        "n_train":        len(train_df),
        "n_val":          len(val_df),
    }

    log.info(f"Train Accuracy : {metrics['train_accuracy']:.3f}")
    log.info(f"Val   Accuracy : {metrics['val_accuracy']:.3f}")
    log.info(f"Val   AUC      : {metrics['val_auc']:.3f}")
    log.info(f"Val   Log-Loss : {metrics['val_logloss']:.3f}")

    # Klassifikationsreport
    log.info("\n" + classification_report(y_val, pred_val,
                                          target_names=["Team B", "Team A"]))

    return model, metrics, train_df, val_df


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter-Tuning mit Optuna
# ─────────────────────────────────────────────────────────────────────────────

def tune_hyperparameters(df: pd.DataFrame, n_trials: int = 50) -> dict:
    """
    Findet beste XGBoost-Hyperparameter via Optuna (Bayesian Optimization).
    Gibt die besten Parameter zurück.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        log.warning("Optuna nicht installiert. Nutze Default-Parameter.")
        return {}

    train_df, _ = time_split(df)
    X = train_df[FEATURES].values
    y = train_df["team_a_won"].values

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "max_depth":         trial.suggest_int("max_depth", 2, 6),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0, 0.5),
        }
        model = build_model(params)
        cv = StratifiedKFold(n_splits=5, shuffle=False)
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_log_loss")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    log.info(f"Beste Parameter gefunden: {best}")
    log.info(f"Bester Score: {study.best_value:.4f}")
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Feature Importance
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_importance(model: Pipeline) -> pd.DataFrame:
    """Gibt Feature-Importance-Tabelle zurück."""
    xgb_step = model.named_steps["xgb"]
    importances = xgb_step.feature_importances_
    return pd.DataFrame({
        "feature":    FEATURES,
        "importance": importances,
    }).sort_values("importance", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Speichern & Laden
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model: Pipeline, elo_ratings: dict):
    """Speichert Modell und ELO-Ratings auf Disk."""
    joblib.dump(model,       MODEL_PATH)
    joblib.dump(elo_ratings, ELO_PATH)
    log.info(f"Modell gespeichert: {MODEL_PATH}")
    log.info(f"ELO gespeichert:    {ELO_PATH}")


def load_model() -> tuple[Pipeline, dict]:
    """Lädt Modell und ELO-Ratings von Disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Kein trainiertes Modell unter {MODEL_PATH}. "
                                 "Bitte zuerst train.py ausführen.")
    model       = joblib.load(MODEL_PATH)
    elo_ratings = joblib.load(ELO_PATH) if ELO_PATH.exists() else {}
    log.info(f"Modell geladen: {MODEL_PATH}")
    return model, elo_ratings


# ─────────────────────────────────────────────────────────────────────────────
# Prediction für einzelnes Match
# ─────────────────────────────────────────────────────────────────────────────

def predict_match(model: Pipeline, features: dict) -> dict:
    """
    Gibt Gewinnwahrscheinlichkeiten für ein Match zurück.

    Args:
        model:    Trainiertes Modell
        features: Feature-Dict aus utils.features.build_prediction_features()

    Returns:
        {"prob_a": float, "prob_b": float, "predicted_winner": str}
    """
    from config import FEATURES as FEAT_COLS
    x = np.array([[features[f] for f in FEAT_COLS]])
    prob = model.predict_proba(x)[0]
    return {
        "prob_a": float(prob[1]),
        "prob_b": float(prob[0]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Backtesting: Simuliere Predictions auf historischen Matches
# ─────────────────────────────────────────────────────────────────────────────

def backtest(model: Pipeline, val_df: pd.DataFrame) -> pd.DataFrame:
    """Fügt Prediction-Spalten zum Validierungs-DataFrame hinzu."""
    X_val = val_df[FEATURES].values
    probs = model.predict_proba(X_val)
    val_df = val_df.copy()
    val_df["prob_a"]     = probs[:, 1]
    val_df["prob_b"]     = probs[:, 0]
    val_df["predicted"]  = (probs[:, 1] >= 0.5).astype(int)
    val_df["correct"]    = (val_df["predicted"] == val_df["team_a_won"]).astype(int)
    val_df["confidence"] = np.abs(probs[:, 1] - 0.5) * 2  # 0=unentschieden, 1=sicher
    return val_df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Modell trainieren")
    parser.add_argument("--tune",   action="store_true", help="Hyperparameter tunen (Optuna)")
    parser.add_argument("--trials", type=int, default=30, help="Anzahl Optuna-Trials")
    args = parser.parse_args()

    if not FEATURES_CSV.exists():
        print(f"Features-CSV nicht gefunden: {FEATURES_CSV}")
        print("Bitte zuerst ausführen: python utils/features.py")
        sys.exit(1)

    df = pd.read_csv(FEATURES_CSV, parse_dates=["date"])

    best_params = {}
    if args.tune:
        log.info(f"Starte Hyperparameter-Tuning ({args.trials} Trials) ...")
        best_params = tune_hyperparameters(df, n_trials=args.trials)

    model, metrics, train_df, val_df = train(df, params=best_params)

    fi = get_feature_importance(model)
    print("\nFeature Importance:")
    print(fi.to_string(index=False))

    bt = backtest(model, val_df)
    print(f"\nBacktest: {bt['correct'].mean():.1%} korrekte Predictions auf Validierungsset")