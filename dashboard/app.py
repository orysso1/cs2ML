# dashboard/app.py
# CS2 Match Prediction Dashboard — Streamlit
#
# Starten mit:  streamlit run dashboard/app.py

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PAGE_TITLE, PAGE_ICON, FEATURES_CSV, RAW_CSV,
    MODEL_PATH, FEATURES
)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Seiten-Konfiguration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0a0f1a; }
    .metric-card {
        background: #0d1b2a;
        border: 1px solid #1a3a5c;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .prob-bar-a { background: linear-gradient(90deg, #00d4ff, #0077aa); border-radius: 4px; }
    .prob-bar-b { background: linear-gradient(90deg, #ff6b00, #aa3300); border-radius: 4px; }
    h1, h2, h3 { color: #e0f0ff; }
    .stSelectbox label { color: #a0c0d0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Daten & Modell laden (gecacht)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Lade Modell ...")
def load_model_cached():
    """Lädt Modell + ELO — nur einmal beim Start."""
    try:
        from models.trainer import load_model
        return load_model()
    except FileNotFoundError:
        return None, {}


@st.cache_data(show_spinner="Lade Match-Daten ...")
def load_data():
    """Lädt Feature-Daten und Rohdaten."""
    df_feat = None
    df_raw  = None

    if FEATURES_CSV.exists():
        df_feat = pd.read_csv(FEATURES_CSV, parse_dates=["date"])

    if RAW_CSV.exists():
        df_raw = pd.read_csv(RAW_CSV, parse_dates=["date"])

    return df_feat, df_raw


def get_teams(df_feat, df_raw) -> list[str]:
    """Alle Teams aus den Daten."""
    teams = set()
    for df in [df_feat, df_raw]:
        if df is not None and "team_a" in df.columns:
            teams.update(df["team_a"].dropna().unique())
            teams.update(df["team_b"].dropna().unique())
    return sorted(teams)


# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

def confidence_label(prob: float) -> tuple[str, str]:
    """Gibt Label und Farbe für Konfidenz zurück."""
    diff = abs(prob - 0.5)
    if diff >= 0.25:
        return "Sehr sicher", "#00ff9f"
    elif diff >= 0.15:
        return "Sicher",      "#00d4ff"
    elif diff >= 0.08:
        return "Leicht",      "#ffcc00"
    else:
        return "Unentschieden", "#ff6b00"


def make_gauge(prob_a: float, team_a: str, team_b: str) -> go.Figure:
    """Erstellt Gauge-Chart für Gewinnwahrscheinlichkeit."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob_a * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "color": "#00d4ff"}},
        gauge={
            "axis":       {"range": [0, 100], "tickcolor": "#2a4a6a"},
            "bar":        {"color": "#00d4ff", "thickness": 0.3},
            "bgcolor":    "#0d1b2a",
            "bordercolor": "#1a3a5c",
            "steps": [
                {"range": [0, 40],   "color": "#1a0a00"},
                {"range": [40, 60],  "color": "#0a1a0a"},
                {"range": [60, 100], "color": "#0a1a2a"},
            ],
            "threshold": {
                "line":      {"color": "#ff6b00", "width": 3},
                "thickness": 0.8,
                "value":     50,
            },
        },
        title={"text": f"{team_a} vs {team_b}", "font": {"color": "#a0c0d0", "size": 14}},
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="#0a0f1a",
        font_color="#c0d8e8",
    )
    return fig


def accuracy_over_time_chart(bt_df: pd.DataFrame) -> go.Figure:
    """Rolling Accuracy über Zeit."""
    bt_df = bt_df.sort_values("date").copy()
    bt_df["rolling_acc"] = bt_df["correct"].rolling(50, min_periods=10).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bt_df["date"], y=bt_df["rolling_acc"],
        mode="lines", name="Rolling Accuracy (50 Matches)",
        line=dict(color="#00d4ff", width=2),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.08)",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#ff6b00",
                  annotation_text="Baseline (50%)", annotation_position="bottom right")
    fig.update_layout(
        title="Modell-Accuracy über Zeit",
        xaxis_title="Datum",
        yaxis_title="Accuracy (Rolling 50)",
        yaxis=dict(range=[0.3, 0.9], tickformat=".0%"),
        paper_bgcolor="#0a0f1a",
        plot_bgcolor="#0d1b2a",
        font_color="#a0c0d0",
        height=300,
    )
    return fig


def feature_importance_chart(fi_df: pd.DataFrame) -> go.Figure:
    """Horizontales Feature-Importance-Balkendiagramm."""
    fig = go.Figure(go.Bar(
        x=fi_df["importance"],
        y=fi_df["feature"],
        orientation="h",
        marker_color=["#00d4ff"] * 3 + ["#0077aa"] * max(0, len(fi_df) - 3),
    ))
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance",
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="#0a0f1a",
        plot_bgcolor="#0d1b2a",
        font_color="#a0c0d0",
        height=300,
        margin=dict(l=140, r=20),
    )
    return fig


def winrate_history_chart(df: pd.DataFrame, team: str) -> go.Figure:
    """Winrate-Verlauf eines Teams."""
    mask = (df["team_a"] == team) | (df["team_b"] == team)
    sub = df.loc[mask].copy().sort_values("date")
    sub["won"] = (sub["winner"] == team).astype(int)
    sub["rolling_wr"] = sub["won"].rolling(20, min_periods=5).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["date"], y=sub["rolling_wr"],
        mode="lines", name="Winrate (Rolling 20)",
        line=dict(color="#00ff9f", width=2),
        fill="tozeroy", fillcolor="rgba(0,255,159,0.07)",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#555")
    fig.update_layout(
        title=f"{team} — Winrate-Verlauf",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        paper_bgcolor="#0a0f1a",
        plot_bgcolor="#0d1b2a",
        font_color="#a0c0d0",
        height=260,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Hauptseite
# ─────────────────────────────────────────────────────────────────────────────

def main():
    model, elo_ratings = load_model_cached()
    df_feat, df_raw    = load_data()
    teams              = get_teams(df_feat, df_raw)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🎯 CS2 Predictor")
        st.markdown("---")

        page = st.radio("Navigation", [
            "🏠 Prediction",
            "📊 Model-Performance",
            "👥 Team-Analyse",
            "📋 Match-Historie",
            "⚙️ Setup & Info",
        ])

        st.markdown("---")
        if MODEL_PATH.exists():
            st.success("✅ Modell geladen")
        else:
            st.error("❌ Kein Modell. Bitte `python train.py --demo` ausführen.")

        if df_feat is not None:
            st.info(f"📦 {len(df_feat)} Matches geladen")

    # ─────────────────────────────────────────────────────────────────────────
    # SEITE: Prediction
    # ─────────────────────────────────────────────────────────────────────────
    if page == "🏠 Prediction":
        st.title("🎯 CS2 Match Prediction")
        st.markdown("Wähle zwei Teams und erhalte eine ML-basierte Gewinnwahrscheinlichkeit.")

        if model is None:
            st.warning("⚠️ Kein trainiertes Modell gefunden. Führe zunächst `python train.py --demo` aus.")
            st.code("python train.py --demo", language="bash")
            return

        col1, col2 = st.columns(2)
        with col1:
            team_a = st.selectbox("🟦 Team A", teams, index=0 if teams else None)
        with col2:
            remaining = [t for t in teams if t != team_a]
            team_b = st.selectbox("🟥 Team B", remaining, index=min(1, len(remaining)-1))

        predict_btn = st.button("⚡ Prediction berechnen", type="primary", use_container_width=True)

        if predict_btn and team_a and team_b:
            if df_raw is None:
                st.error("Keine historischen Daten vorhanden.")
                return

            with st.spinner("Berechne Features ..."):
                from utils.features import build_prediction_features
                from models.trainer import predict_match

                feats  = build_prediction_features(team_a, team_b, df_raw, elo_ratings)
                result = predict_match(model, feats)

            prob_a = result["prob_a"]
            prob_b = result["prob_b"]
            conf_label, conf_color = confidence_label(prob_a)

            winner = team_a if prob_a >= prob_b else team_b
            winner_prob = max(prob_a, prob_b)

            # ── Hauptergebnis ─────────────────────────────────────────────
            st.markdown("---")
            c1, c2, c3 = st.columns([2, 3, 2])
            with c1:
                st.metric(f"🟦 {team_a}", f"{prob_a:.1%}")
            with c2:
                st.plotly_chart(make_gauge(prob_a, team_a, team_b),
                                use_container_width=True)
            with c3:
                st.metric(f"🟥 {team_b}", f"{prob_b:.1%}")

            st.markdown("---")
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("🏆 Vorhergesagter Sieger", winner)
            cc2.metric("📊 Gewinnchance",           f"{winner_prob:.1%}")
            cc3.metric("🎯 Konfidenz",              conf_label)

            # ── Feature-Details ───────────────────────────────────────────
            with st.expander("🔍 Feature-Details anzeigen"):
                feat_df = pd.DataFrame([
                    {"Feature": k, "Team A - Team B": f"{v:+.3f}"}
                    for k, v in feats.items()
                ])
                st.dataframe(feat_df, use_container_width=True, hide_index=True)

            # ── Letzte H2H-Matches ────────────────────────────────────────
            if df_raw is not None:
                mask = (
                    ((df_raw["team_a"] == team_a) & (df_raw["team_b"] == team_b)) |
                    ((df_raw["team_a"] == team_b) & (df_raw["team_b"] == team_a))
                )
                h2h_df = df_raw.loc[mask].sort_values("date", ascending=False).head(10)
                if len(h2h_df) > 0:
                    with st.expander(f"📋 Letzte H2H-Matches ({len(h2h_df)})"):
                        h2h_show = h2h_df[["date", "team_a", "score_a", "score_b", "team_b", "winner", "event"]].copy()
                        h2h_show["date"] = h2h_show["date"].dt.strftime("%Y-%m-%d")
                        st.dataframe(h2h_show, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SEITE: Model-Performance
    # ─────────────────────────────────────────────────────────────────────────
    elif page == "📊 Model-Performance":
        st.title("📊 Model-Performance")

        bt_path = Path("data/backtest_results.csv")
        if not bt_path.exists():
            st.warning("Keine Backtest-Daten. Führe `python train.py` aus.")
            return

        bt = pd.read_csv(bt_path, parse_dates=["date"])

        # Kennzahlen
        acc     = bt["correct"].mean()
        n_total = len(bt)
        n_high  = len(bt[bt["confidence"] >= 0.5])
        acc_high = bt[bt["confidence"] >= 0.5]["correct"].mean() if n_high > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Gesamt-Accuracy",       f"{acc:.1%}")
        c2.metric("Matches (Val.)",        f"{n_total}")
        c3.metric("High-Conf Accuracy",    f"{acc_high:.1%}")
        c4.metric("High-Conf Matches",     f"{n_high}")

        st.plotly_chart(accuracy_over_time_chart(bt), use_container_width=True)

        # Feature Importance
        if model is not None:
            from models.trainer import get_feature_importance
            fi = get_feature_importance(model)
            st.plotly_chart(feature_importance_chart(fi), use_container_width=True)

        # Accuracy nach Confidence-Level
        st.subheader("Accuracy nach Konfidenz-Level")
        rows = []
        for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
            sub = bt[bt["confidence"] >= t]
            if len(sub) > 0:
                rows.append({
                    "Min. Konfidenz": f"{t:.0%}",
                    "Matches":        len(sub),
                    "Accuracy":       f"{sub['correct'].mean():.1%}",
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SEITE: Team-Analyse
    # ─────────────────────────────────────────────────────────────────────────
    elif page == "👥 Team-Analyse":
        st.title("👥 Team-Analyse")

        if not teams:
            st.warning("Keine Teams gefunden. Daten laden.")
            return

        selected_team = st.selectbox("Team auswählen", teams)

        df_src = df_raw if df_raw is not None else df_feat
        if df_src is None:
            st.warning("Keine Daten vorhanden.")
            return

        mask = (df_src["team_a"] == selected_team) | (df_src["team_b"] == selected_team)
        team_matches = df_src.loc[mask].copy()

        if len(team_matches) == 0:
            st.info("Keine Matches für dieses Team.")
            return

        team_matches["won"] = (team_matches["winner"] == selected_team).astype(int)

        # Kennzahlen
        total_m   = len(team_matches)
        total_w   = team_matches["won"].sum()
        winrate   = total_w / total_m if total_m > 0 else 0
        elo_curr  = elo_ratings.get(selected_team, 1500)

        last_5 = team_matches.tail(5)["won"].tolist()
        form_str = " ".join(["✅" if w else "❌" for w in reversed(last_5)])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Gesamtmatches",  total_m)
        c2.metric("Gesamtsiege",    int(total_w))
        c3.metric("Winrate",        f"{winrate:.1%}")
        c4.metric("ELO Rating",     f"{elo_curr:.0f}")

        st.markdown(f"**Letzte 5 Matches:** {form_str}")

        # Winrate-Chart
        st.plotly_chart(winrate_history_chart(df_src, selected_team), use_container_width=True)

        # Stärkste Gegner
        st.subheader("Matches vs. Gegner (Top 10)")
        opponents = []
        for opp in df_src["team_a"].unique():
            if opp == selected_team:
                continue
            opp_mask = mask & ((df_src["team_a"] == opp) | (df_src["team_b"] == opp))
            sub = df_src.loc[opp_mask]
            if len(sub) < 2:
                continue
            wins = (sub["winner"] == selected_team).sum()
            opponents.append({
                "Gegner":   opp,
                "Matches":  len(sub),
                "Siege":    int(wins),
                "Winrate":  f"{wins/len(sub):.1%}",
            })

        if opponents:
            opp_df = pd.DataFrame(opponents).sort_values("Matches", ascending=False).head(10)
            st.dataframe(opp_df, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SEITE: Match-Historie
    # ─────────────────────────────────────────────────────────────────────────
    elif page == "📋 Match-Historie":
        st.title("📋 Match-Historie")

        df_src = df_raw if df_raw is not None else df_feat
        if df_src is None:
            st.warning("Keine Daten vorhanden.")
            return

        col1, col2, col3 = st.columns(3)
        with col1:
            search_team = st.text_input("Team-Filter", placeholder="z.B. Natus Vincere")
        with col2:
            date_from = st.date_input("Von", value=None)
        with col3:
            date_to = st.date_input("Bis", value=None)

        filtered = df_src.copy()
        if search_team:
            filtered = filtered[
                filtered["team_a"].str.contains(search_team, case=False, na=False) |
                filtered["team_b"].str.contains(search_team, case=False, na=False)
            ]
        if date_from:
            filtered = filtered[filtered["date"] >= pd.Timestamp(date_from)]
        if date_to:
            filtered = filtered[filtered["date"] <= pd.Timestamp(date_to)]

        filtered = filtered.sort_values("date", ascending=False)
        st.info(f"{len(filtered)} Matches gefunden")

        show_cols = [c for c in ["date", "team_a", "score_a", "score_b", "team_b", "winner", "event"]
                     if c in filtered.columns]
        display = filtered[show_cols].head(200).copy()
        if "date" in display.columns:
            display["date"] = display["date"].dt.strftime("%Y-%m-%d")

        st.dataframe(display, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SEITE: Setup & Info
    # ─────────────────────────────────────────────────────────────────────────
    elif page == "⚙️ Setup & Info":
        st.title("⚙️ Setup & Info")

        st.subheader("📦 Schnellstart")
        st.code("""
# 1. Abhängigkeiten installieren
pip install -r requirements.txt

# 2. Demo-Daten generieren und Modell trainieren
python train.py --demo

# 3. Dashboard starten
streamlit run dashboard/app.py
        """, language="bash")

        st.subheader("🌐 Echte HLTV-Daten")
        st.code("""
# Option A: Von HLTV scrapen (langsam, ~15-20 Minuten)
python train.py --scrape

# Option B: Kaggle-Dataset (empfohlen für den Start)
# 1. kaggle.com → "CS2 HLTV Professional Match Statistics" herunterladen
# 2. CSV in den data/-Ordner legen
python train.py --csv data/dein_dataset.csv
        """, language="bash")

        st.subheader("🔧 Mit Hyperparameter-Tuning")
        st.code("""
python train.py --demo --tune --trials 50
        """, language="bash")

        st.subheader("📊 Projekt-Struktur")
        st.code("""
cs2predictor/
├── config.py              ← Zentrale Einstellungen
├── train.py               ← Haupt-Pipeline (hier starten!)
├── requirements.txt
├── data/
│   ├── scraper.py         ← HLTV-Scraper
│   ├── kaggle_loader.py   ← CSV/Kaggle-Loader + Demo-Daten
│   ├── matches_raw.csv    ← Rohdaten (automatisch erstellt)
│   └── matches_features.csv ← Features (automatisch erstellt)
├── models/
│   ├── trainer.py         ← Training, Evaluation, Tuning
│   ├── xgb_model.joblib   ← Trainiertes Modell (nach Training)
│   └── elo_ratings.joblib ← ELO-Ratings (nach Training)
├── utils/
│   └── features.py        ← ELO, Winrate, Form, H2H, ...
└── dashboard/
    └── app.py             ← Streamlit Dashboard (diese Datei)
        """, language="text")

        st.subheader("⚠️ Hinweise")
        st.info("""
**HLTV-Scraping:** HLTV.org hat Bot-Schutz (Cloudflare).
Das Scraping kann 15–30 Minuten dauern. Nutze VPN oder Proxys falls Probleme auftreten.

**Accuracy:** Realistisch sind 55–65%. Esport-Ergebnisse haben hohe Varianz.
Ein gutes Modell schlägt dauerhaft die 50%-Baseline.

**Data Leakage:** Der zeitbasierte Split ist KRITISCH.
Niemals zukünftige Matches ins Training mischen!
        """)

        st.subheader("📈 Status")
        status_rows = [
            {"Datei":            str(RAW_CSV),      "Status": "✅" if RAW_CSV.exists()      else "❌ fehlt"},
            {"Datei":            str(FEATURES_CSV),  "Status": "✅" if FEATURES_CSV.exists() else "❌ fehlt"},
            {"Datei":            str(MODEL_PATH),    "Status": "✅" if MODEL_PATH.exists()   else "❌ fehlt"},
        ]
        st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()