# models/trainer.py
# Vollständiges Trainer-Modul:
#   - Recency-gewichtetes Training
#   - Stacking Ensemble (XGBoost + RandomForest + LightGBM)
#   - Wahrscheinlichkeits-Kalibrierung
#   - Walk-Forward-Validation
#   - Tier-spezifische Modelle
#   - Betting-Simulation als Evaluation

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (accuracy_score, log_loss, roc_auc_score,
                              classification_report, brier_score_loss)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FEATURES, CS2_MAPS, FEATURES_CSV,
    MODEL_PATH, MAP_MODEL_PATH, ELO_PATH,
    ENSEMBLE_PATH, CALIBRATED_PATH,
    TRAIN_CUTOFF, RANDOM_STATE
)

log = logging.getLogger(__name__)

# LightGBM optional
try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    log.debug("LightGBM nicht installiert — Ensemble ohne LGBM")


# ─────────────────────────────────────────────────────────────────────────────
# Modell-Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_xgb(params: dict | None = None) -> Pipeline:
    default = {
        "n_estimators": 400, "max_depth": 4, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "min_child_weight": 3, "gamma": 0.1,
        "eval_metric": "logloss", "random_state": RANDOM_STATE, "n_jobs": -1,
    }
    if params:
        default.update(params)
    p = Pipeline([("scaler", StandardScaler()),
                  ("xgb",   XGBClassifier(**default))])
    p._model_type = "xgb"
    return p


def build_rf() -> Pipeline:
    p = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])
    p._model_type = "rf"
    return p


def build_lgbm() -> Pipeline | None:
    if not LGBM_AVAILABLE:
        return None
    p = Pipeline([
        ("scaler", StandardScaler()),
        ("lgbm", LGBMClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )),
    ])
    p._model_type = "lgbm"
    return p


def build_ensemble(params: dict | None = None) -> StackingClassifier:
    """
    Stacking Ensemble: XGBoost + RandomForest [+ LightGBM]
    Meta-Learner: Logistic Regression
    """
    estimators = [
        ("xgb", build_xgb(params)),
        ("rf",  build_rf()),
    ]
    lgbm = build_lgbm()
    if lgbm:
        estimators.append(("lgbm", lgbm))

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=1.0, random_state=RANDOM_STATE),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )
    log.info(f"Ensemble: {[e[0] for e in estimators]} + LogReg Meta-Learner")
    return stack


# ─────────────────────────────────────────────────────────────────────────────
# Split
# ─────────────────────────────────────────────────────────────────────────────

def time_split(df: pd.DataFrame, cutoff: str = TRAIN_CUTOFF):
    df = df.sort_values("date").copy()

    def _both(d): return d["team_a_won"].nunique() == 2

    train = df[df["date"] < cutoff]
    val   = df[df["date"] >= cutoff]

    if len(val) < 50 or not _both(val):
        log.warning(f"Cutoff '{cutoff}' → 80/20-Fallback")
        split_idx = int(len(df) * 0.8)
        train = df.iloc[:split_idx]
        val   = df.iloc[split_idx:]

    if not _both(val) or not _both(train):
        from sklearn.model_selection import train_test_split as tts
        train, val = tts(df, test_size=0.2, random_state=42,
                         stratify=df["team_a_won"])
        train = train.sort_values("date")
        val   = val.sort_values("date")

    if not _both(val) or not _both(train):
        raise ValueError(
            f"team_a_won nur eine Klasse:\n"
            f"  rm data/matches_raw.csv data/matches_features.csv\n"
            f"  python train.py --csv dein_dataset.csv"
        )
    log.info(f"Train: {len(train)} | Val: {len(val)} | Cutoff: {val['date'].min().date()}")
    return train, val


# ─────────────────────────────────────────────────────────────────────────────
# Recency-Gewichte
# ─────────────────────────────────────────────────────────────────────────────

def _sample_weights(df: pd.DataFrame) -> np.ndarray:
    from config import RECENCY_HALFLIFE_DAYS
    max_date = df["date"].max()
    days_old = (max_date - df["date"]).dt.days.clip(lower=0)
    return np.exp(-days_old * np.log(2) / RECENCY_HALFLIFE_DAYS).values


# ─────────────────────────────────────────────────────────────────────────────
# Training — Single XGBoost
# ─────────────────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame, params: dict | None = None,
          use_recency_weights: bool = True) -> tuple:
    """Trainiert einzelnes XGBoost-Modell mit Recency-Gewichtung."""
    available = [f for f in FEATURES if f in df.columns]
    missing   = [f for f in FEATURES if f not in df.columns]
    if missing:
        log.warning(f"Fehlende Features: {missing}")

    train_df, val_df = time_split(df)

    X_train = train_df[available].fillna(0).values
    y_train = train_df["team_a_won"].values
    X_val   = val_df[available].fillna(0).values
    y_val   = val_df["team_a_won"].values

    model = build_xgb(params)

    if use_recency_weights:
        weights = _sample_weights(train_df)
        log.info(f"Recency-Gewichtung: min={weights.min():.3f} max={weights.max():.3f}")
        # Pipeline: Gewichte nur an XGB weitergeben
        model.fit(X_train, y_train,
                  xgb__sample_weight=weights)
    else:
        model.fit(X_train, y_train)

    model._feature_names = available
    metrics = _eval(model, X_train, y_train, X_val, y_val, available, train_df, val_df)
    return model, metrics, train_df, val_df


# ─────────────────────────────────────────────────────────────────────────────
# Training — Stacking Ensemble
# ─────────────────────────────────────────────────────────────────────────────

def train_ensemble(df: pd.DataFrame, params: dict | None = None) -> tuple:
    """Trainiert Stacking Ensemble (langsamer, aber besser)."""
    available = [f for f in FEATURES if f in df.columns]
    train_df, val_df = time_split(df)

    X_train = train_df[available].fillna(0).values
    y_train = train_df["team_a_won"].values
    X_val   = val_df[available].fillna(0).values
    y_val   = val_df["team_a_won"].values

    log.info("Trainiere Stacking Ensemble (dauert länger) ...")
    ensemble = build_ensemble(params)
    ensemble.fit(X_train, y_train)
    ensemble._feature_names = available

    metrics = _eval(ensemble, X_train, y_train, X_val, y_val, available, train_df, val_df)
    log.info(f"Ensemble Val Accuracy: {metrics['val_accuracy']:.3f}")
    return ensemble, metrics, train_df, val_df


# ─────────────────────────────────────────────────────────────────────────────
# Kalibrierung
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_model(model, X_val: np.ndarray, y_val: np.ndarray,
                    method: str = "isotonic"):
    """
    Kalibriert Wahrscheinlichkeiten mit Isotonic Regression.
    Wichtig für Value Betting: 70% Modell-Prob soll echte 70% bedeuten.
    Verwendet val-Daten als Kalibrierungs-Set.
    """
    log.info(f"Kalibriere Modell ({method}) ...")

    # Brier Score vor Kalibrierung
    probs_before = model.predict_proba(X_val)[:, 1]
    brier_before = brier_score_loss(y_val, probs_before)

    # Kalibrierung auf Val-Set (cv="prefit" = Modell bereits trainiert)
    calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrated.fit(X_val, y_val)
    calibrated._feature_names = getattr(model, "_feature_names", [])

    probs_after = calibrated.predict_proba(X_val)[:, 1]
    brier_after = brier_score_loss(y_val, probs_after)

    log.info(f"Brier Score: {brier_before:.4f} → {brier_after:.4f} "
             f"({'besser' if brier_after < brier_before else 'schlechter'})")
    return calibrated


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward-Validation
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_eval(df: pd.DataFrame,
                       params: dict | None = None,
                       n_splits: int = 6) -> pd.DataFrame:
    """
    Realistischere Evaluation als einfacher Train/Val-Split.
    Gibt DataFrame mit Accuracy pro Split zurück.
    """
    from utils.features import walk_forward_splits
    splits  = walk_forward_splits(df, n_splits=n_splits)
    available = [f for f in FEATURES if f in df.columns]
    results = []

    for i, (train_df, val_df, cutoff) in enumerate(splits):
        X_tr = train_df[available].fillna(0).values
        y_tr = train_df["team_a_won"].values
        X_va = val_df[available].fillna(0).values
        y_va = val_df["team_a_won"].values

        model = build_xgb(params)
        model.fit(X_tr, y_tr, xgb__sample_weight=_sample_weights(train_df))

        probs = model.predict_proba(X_va)[:, 1]
        acc   = accuracy_score(y_va, (probs >= 0.5).astype(int))
        auc   = roc_auc_score(y_va, probs) if len(set(y_va)) == 2 else float("nan")

        # Betting-Simulation: wette wenn EV > 5% bei Durchschnittsquote 1.9
        avg_odds = 1.9
        ev       = probs * avg_odds - 1
        bet_mask = ev > 0.05
        bet_acc  = accuracy_score(y_va[bet_mask], (probs[bet_mask] >= 0.5).astype(int)) \
                   if bet_mask.sum() > 5 else float("nan")
        roi      = ((probs[bet_mask] >= 0.5).astype(int) * (avg_odds - 1) - \
                    (1 - (probs[bet_mask] >= 0.5).astype(int))).mean() \
                   if bet_mask.sum() > 5 else float("nan")

        results.append({
            "split":          i + 1,
            "cutoff":         cutoff,
            "train_matches":  len(train_df),
            "val_matches":    len(val_df),
            "accuracy":       round(acc, 4),
            "auc":            round(auc, 4),
            "bet_matches":    int(bet_mask.sum()),
            "bet_accuracy":   round(bet_acc, 4) if not np.isnan(bet_acc) else None,
            "roi_sim":        round(roi, 4)      if not np.isnan(roi) else None,
        })
        log.info(f"Split {i+1}/{len(splits)}: "
                 f"Acc={acc:.3f} AUC={auc:.3f} "
                 f"Bet-Acc={bet_acc:.3f if not np.isnan(bet_acc) else '---'}")

    df_res = pd.DataFrame(results)
    log.info(f"\nWalk-Forward Ø Accuracy: {df_res['accuracy'].mean():.3f} "
             f"± {df_res['accuracy'].std():.3f}")
    return df_res


# ─────────────────────────────────────────────────────────────────────────────
# Tier-spezifische Modelle
# ─────────────────────────────────────────────────────────────────────────────

def train_tier_models(df: pd.DataFrame,
                       params: dict | None = None) -> dict:
    """
    Trainiert separate Modelle für:
    - Tier 1/Major (event_tier >= 2)
    - Alle Matches (Fallback)
    """
    tier_models = {}
    available   = [f for f in FEATURES if f in df.columns]

    for tier_name, mask in [
        ("tier1_major", df.get("event_tier", pd.Series(1, index=df.index)) >= 2),
        ("all",         pd.Series(True, index=df.index)),
    ]:
        sub = df.loc[mask]
        if len(sub) < 200:
            log.warning(f"Tier-Modell '{tier_name}': nur {len(sub)} Matches — übersprungen")
            continue

        try:
            train_df, val_df = time_split(sub)
            X_tr = train_df[available].fillna(0).values
            y_tr = train_df["team_a_won"].values
            X_va = val_df[available].fillna(0).values
            y_va = val_df["team_a_won"].values

            model = build_xgb(params)
            model.fit(X_tr, y_tr, xgb__sample_weight=_sample_weights(train_df))
            model._feature_names = available

            acc = accuracy_score(y_va, (model.predict_proba(X_va)[:, 1] >= 0.5).astype(int))
            tier_models[tier_name] = model
            log.info(f"Tier-Modell '{tier_name}': {len(sub)} Matches, Val-Acc={acc:.3f}")
        except Exception as e:
            log.warning(f"Tier-Modell '{tier_name}' fehlgeschlagen: {e}")

    return tier_models


# ─────────────────────────────────────────────────────────────────────────────
# Map-Modelle
# ─────────────────────────────────────────────────────────────────────────────

def train_map_models(df: pd.DataFrame, params: dict | None = None) -> dict:
    map_models = {}
    train_df, val_df = time_split(df)

    for m in CS2_MAPS:
        map_features = [f for f in [
            "rating_diff", "adr_diff", "kast_diff",
            "elo_diff", "team1_h2h_pct", "past3_diff",
            f"{m}_winrate_diff", "is_lan", "event_tier",
            "momentum_diff", "quality_winrate_diff",
        ] if f in df.columns]

        if len(map_features) < 2:
            continue

        X_tr = train_df[map_features].fillna(0).values
        y_tr = train_df["team_a_won"].values
        X_va = val_df[map_features].fillna(0).values
        y_va = val_df["team_a_won"].values

        model = build_xgb(params)
        model.fit(X_tr, y_tr, xgb__sample_weight=_sample_weights(train_df))
        model._feature_names = map_features

        acc = accuracy_score(y_va, (model.predict_proba(X_va)[:, 1] >= 0.5).astype(int))
        map_models[m] = model
        log.info(f"Map '{m}': Val-Acc={acc:.3f}")

    return map_models


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _eval(model, X_train, y_train, X_val, y_val,
          available, train_df, val_df) -> dict:
    prob_val   = model.predict_proba(X_val)[:, 1]
    pred_val   = (prob_val >= 0.5).astype(int)
    prob_train = model.predict_proba(X_train)[:, 1]
    pred_train = (prob_train >= 0.5).astype(int)

    n2 = len(set(y_val))
    metrics = {
        "train_accuracy": accuracy_score(y_train, pred_train),
        "val_accuracy":   accuracy_score(y_val,   pred_val),
        "val_auc":        roc_auc_score(y_val, prob_val) if n2 == 2 else float("nan"),
        "val_logloss":    log_loss(y_val, prob_val, labels=[0,1]) if n2 == 2 else float("nan"),
        "val_brier":      brier_score_loss(y_val, prob_val) if n2 == 2 else float("nan"),
        "n_train":        len(train_df),
        "n_val":          len(val_df),
        "features_used":  available,
    }
    log.info(f"Train Acc: {metrics['train_accuracy']:.3f} | "
             f"Val Acc: {metrics['val_accuracy']:.3f} | "
             f"AUC: {metrics['val_auc']:.3f} | "
             f"Brier: {metrics['val_brier']:.4f}")
    if n2 == 2:
        log.info("\n" + classification_report(y_val, pred_val,
                                              labels=[0,1],
                                              target_names=["Team B","Team A"],
                                              zero_division=0))
    return metrics


def get_feature_importance(model) -> pd.DataFrame:
    if hasattr(model, "named_steps") and "xgb" in model.named_steps:
        names = getattr(model, "_feature_names", FEATURES)
        imp   = model.named_steps["xgb"].feature_importances_
        n = min(len(names), len(imp))
        return pd.DataFrame({"feature": names[:n], "importance": imp[:n]}) \
                 .sort_values("importance", ascending=False)
    # Ensemble: Feature Importance aus XGB-Basis-Modell
    if hasattr(model, "estimators_"):
        for name, est in model.estimators_:
            if "xgb" in name:
                return get_feature_importance(est)
    return pd.DataFrame({"feature": [], "importance": []})


def backtest(model, val_df: pd.DataFrame) -> pd.DataFrame:
    feat   = getattr(model, "_feature_names", FEATURES)
    avail  = [f for f in feat if f in val_df.columns]
    probs  = model.predict_proba(val_df[avail].fillna(0).values)
    val_df = val_df.copy()
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
        log.warning("Optuna fehlt: pip install optuna")
        return {}

    available = [f for f in FEATURES if f in df.columns]
    train_df, _ = time_split(df)
    X = train_df[available].fillna(0).values
    y = train_df["team_a_won"].values
    w = _sample_weights(train_df)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "max_depth":        trial.suggest_int("max_depth", 2, 6),
            "learning_rate":    trial.suggest_float("lr", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("mcw", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0, 0.5),
        }
        model  = build_xgb(params)
        cv     = StratifiedKFold(n_splits=5, shuffle=False)
        scores = cross_val_score(model, X, y, cv=cv,
                                 scoring="neg_log_loss",
                                 params={"xgb__sample_weight": w})
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    log.info(f"Beste Parameter: {study.best_params}")
    return study.best_params


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_match(model, features: dict) -> dict:
    feat_names = getattr(model, "_feature_names", FEATURES)
    x    = np.array([[features.get(f, 0.0) for f in feat_names]])
    prob = model.predict_proba(x)[0]
    return {"prob_a": float(prob[1]), "prob_b": float(prob[0])}


def predict_maps(map_models: dict, features: dict) -> dict:
    results = {}
    for map_name, model in map_models.items():
        feat_names = getattr(model, "_feature_names", [])
        x = np.array([[features.get(f, 0.0) for f in feat_names]])
        try:
            results[map_name] = float(model.predict_proba(x)[0][1])
        except Exception:
            results[map_name] = 0.5
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Speichern & Laden
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model, elo_ratings: dict, map_models: dict | None = None,
               calibrated=None, ensemble=None):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(elo_ratings, ELO_PATH)
    if map_models:
        joblib.dump(map_models, MAP_MODEL_PATH)
    if calibrated:
        joblib.dump(calibrated, CALIBRATED_PATH)
    if ensemble:
        joblib.dump(ensemble, ENSEMBLE_PATH)
    log.info(f"Modelle gespeichert: {MODEL_PATH.parent}")


def load_model() -> tuple:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Kein Modell: {MODEL_PATH}")
    model       = joblib.load(MODEL_PATH)
    elo_ratings = joblib.load(ELO_PATH) if ELO_PATH.exists() else {}
    map_models  = joblib.load(MAP_MODEL_PATH) if MAP_MODEL_PATH.exists() else {}
    # Kalibriertes Modell laden falls vorhanden
    if CALIBRATED_PATH.exists():
        calibrated = joblib.load(CALIBRATED_PATH)
        log.info("Kalibriertes Modell geladen")
        calibrated._feature_names = getattr(model, "_feature_names", [])
        return calibrated, elo_ratings, map_models
    return model, elo_ratings, map_models