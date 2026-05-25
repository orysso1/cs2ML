# dashboard/app.py — CS2 Match Prediction Dashboard
# streamlit run dashboard/app.py

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PAGE_TITLE, PAGE_ICON, FEATURES_CSV, RAW_CSV, MODEL_PATH, CS2_MAPS

log = logging.getLogger(__name__)

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON,
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="metric-container"] { background:#0d1b2a; border:1px solid #1a3a5c;
        border-radius:8px; padding:12px 16px; }
    h1,h2,h3 { color:#e0f0ff; }
</style>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cache-Loader
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Lade Modell ...")
def load_model_cached():
    try:
        from models.trainer import load_model
        return load_model()
    except FileNotFoundError:
        return None, {}, {}


def _check_model_updated() -> bool:
    """Prüft ob auto_updater.py das Modell seit dem letzten Reload aktualisiert hat."""
    from config import DATA_DIR
    flag = DATA_DIR / ".model_updated"
    if not flag.exists():
        return False
    session_ts = st.session_state.get("model_load_time", 0)
    flag_ts    = flag.stat().st_mtime
    return flag_ts > session_ts


def _load_last_update_time():
    """Liest den Zeitpunkt des letzten erfolgreichen Updates aus."""
    import json, datetime as _dt
    from config import DATA_DIR
    status_file = DATA_DIR / "updater_status.json"
    if not status_file.exists():
        return None
    try:
        data = json.loads(status_file.read_text())
        ts   = data.get("timestamp")
        if ts:
            return _dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        pass
    return None


def _run_dashboard_update(days_back: int):
    """Führt den Update-Prozess direkt im Dashboard aus (mit Fortschritts-Anzeige)."""
    import datetime as _dt
    from config import DATA_DIR

    progress = st.sidebar.progress(0, text="Starte Update ...")
    status   = st.sidebar.empty()

    try:
        # Schritt 1: Neue Matches holen
        progress.progress(10, text="📡 Hole neue Matches ...")
        new_matches = []

        try:
            from data.api_fetcher import (
                fetch_past_matches, fetch_grid_stats,
                PANDASCORE_KEY, GRID_KEY
            )
            if PANDASCORE_KEY:
                status.info(f"PandaScore: lade letzte {days_back} Tage ...")
                ps = fetch_past_matches(days_back=days_back)
                new_matches.extend(ps)
            if GRID_KEY:
                status.info("GRID: lade aktuelle Stats ...")
                gr = fetch_grid_stats(max_series=30)
                new_matches.extend(gr)
        except Exception as e:
            status.warning(f"API-Fehler: {e} — fahre mit lokalen Daten fort")

        progress.progress(35, text=f"✅ {len(new_matches)} Matches gefunden")

        # Schritt 2: Mergen
        if new_matches:
            progress.progress(45, text="💾 Füge neue Matches ein ...")
            from data.api_fetcher import merge_into_csv
            from config import RAW_CSV
            import pandas as pd

            before = len(pd.read_csv(RAW_CSV)) if RAW_CSV.exists() else 0
            merge_into_csv(new_matches, RAW_CSV)
            after  = len(pd.read_csv(RAW_CSV)) if RAW_CSV.exists() else 0
            added  = after - before
            status.info(f"{added} neue Matches hinzugefügt (gesamt: {after})")
        else:
            added = 0
            status.info("Keine neuen Matches von API — Re-Training mit vorhandenen Daten")

        progress.progress(55, text="⚙️ Berechne Features ...")

        # Schritt 3: Features + Training
        from config import RAW_CSV, FEATURES_CSV
        import pandas as pd

        if not RAW_CSV.exists():
            status.error("Keine Rohdaten vorhanden.")
            progress.empty()
            return

        df_raw = pd.read_csv(RAW_CSV, parse_dates=["date"])
        if pd.api.types.is_datetime64tz_dtype(df_raw["date"]):
            df_raw["date"] = df_raw["date"].dt.tz_convert(None)

        FEATURES_CSV.unlink(missing_ok=True)
        from utils.features import build_features
        df_feat, elo_ratings = build_features(df_raw, save=True)

        progress.progress(75, text="🧠 Trainiere Modell ...")
        from models.trainer import train, train_map_models, save_model, backtest
        model, metrics, _, val_df = train(df_feat)
        map_models = train_map_models(df_feat)
        bt = backtest(model, val_df)
        bt.to_csv(DATA_DIR / "backtest_results.csv", index=False)
        save_model(model, elo_ratings, map_models)

        progress.progress(90, text="📅 Lade Upcoming Matches ...")
        try:
            from data.api_fetcher import fetch_upcoming_matches, save_upcoming, PANDASCORE_KEY
            if PANDASCORE_KEY:
                upcoming = fetch_upcoming_matches()
                save_upcoming(upcoming)
        except Exception:
            pass

        # Timestamp speichern
        import json
        status_data = {
            "timestamp":   _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "new_matches": added,
            "retrained":   True,
            "val_accuracy": metrics.get("val_accuracy", 0),
            "errors":      [],
        }
        (DATA_DIR / "updater_status.json").write_text(json.dumps(status_data, indent=2))
        (DATA_DIR / ".model_updated").write_text(_dt.datetime.now(_dt.timezone.utc).isoformat())

        progress.progress(100, text="✅ Update abgeschlossen!")
        acc = metrics.get("val_accuracy", 0)
        status.success(
            f"✅ Fertig! {added} neue Matches · "
            f"Val-Accuracy: {acc:.1%}"
        )

        # Cache leeren damit Dashboard neue Daten lädt
        st.cache_resource.clear()
        st.cache_data.clear()
        import time as _time
        _time.sleep(1.5)
        st.rerun()

    except Exception as e:
        progress.empty()
        status.error(f"❌ Update fehlgeschlagen: {e}")
        import traceback
        st.sidebar.code(traceback.format_exc(), language="text")


@st.cache_data(show_spinner="Lade Daten ...")
def load_data():
    df_feat = pd.read_csv(FEATURES_CSV, parse_dates=["date"]) if FEATURES_CSV.exists() else None
    df_raw  = pd.read_csv(RAW_CSV,      parse_dates=["date"]) if RAW_CSV.exists()      else None
    return df_feat, df_raw


def get_teams(df_feat, df_raw):
    teams = set()
    for df in [df_feat, df_raw]:
        if df is not None:
            teams.update(df["team_a"].dropna().unique())
            teams.update(df["team_b"].dropna().unique())
    return sorted(teams)


# ─────────────────────────────────────────────────────────────────────────────
# Chart-Helpers
# ─────────────────────────────────────────────────────────────────────────────

DARK = "#0a0f1a"
PANEL = "#0d1b2a"
BLUE  = "#00d4ff"
ORANGE= "#ff6b00"
GREEN = "#00ff9f"

def _layout(fig, h=300, ml=20):
    fig.update_layout(height=h, margin=dict(l=ml,r=20,t=40,b=20),
                      paper_bgcolor=DARK, plot_bgcolor=PANEL,
                      font_color="#a0c0d0")
    return fig


def gauge_chart(prob_a, team_a, team_b):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob_a * 100, 1),
        number={"suffix": "%", "font": {"size": 40, "color": BLUE}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#2a4a6a"},
            "bar":  {"color": BLUE, "thickness": 0.28},
            "bgcolor": PANEL, "bordercolor": "#1a3a5c",
            "steps": [
                {"range": [0,  40], "color": "#1a0a00"},
                {"range": [40, 60], "color": "#0a1a0a"},
                {"range": [60,100], "color": "#0a1a2a"},
            ],
            "threshold": {"line": {"color": ORANGE, "width": 3},
                          "thickness": 0.8, "value": 50},
        },
        title={"text": f"{team_a} <br><span style='font-size:11px'>vs</span><br> {team_b}",
               "font": {"color": "#a0c0d0", "size": 13}},
    ))
    fig.update_layout(height=260, margin=dict(l=20,r=20,t=50,b=10),
                      paper_bgcolor=DARK, font_color="#c0d8e8")
    return fig


def map_bar_chart(map_probs: dict, team_a: str, team_b: str):
    maps  = list(map_probs.keys())
    probs = [map_probs[m] for m in maps]

    colors = [GREEN if p > 0.55 else (ORANGE if p < 0.45 else "#ffcc00") for p in probs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=maps, y=probs, marker_color=colors, name=team_a,
        text=[f"{p*100:.1f}%" for p in probs], textposition="outside",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#555",
                  annotation_text="50%", annotation_position="right")
    fig.update_layout(
        title=f"Map-Gewinnwahrscheinlichkeit — {team_a}",
        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
        showlegend=False,
    )
    return _layout(fig, h=320)


def accuracy_chart(bt):
    bt = bt.sort_values("date").copy()
    bt["rolling_acc"] = bt["correct"].rolling(50, min_periods=10).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt["date"], y=bt["rolling_acc"], mode="lines",
                             line=dict(color=BLUE, width=2),
                             fill="tozeroy", fillcolor="rgba(0,212,255,0.07)"))
    fig.add_hline(y=0.5, line_dash="dash", line_color=ORANGE,
                  annotation_text="Baseline 50%")
    fig.update_layout(title="Modell-Accuracy über Zeit (Rolling 50)",
                      yaxis=dict(tickformat=".0%", range=[0.3, 0.9]))
    return _layout(fig, h=300)


def fi_chart(fi_df):
    fig = go.Figure(go.Bar(
        x=fi_df["importance"], y=fi_df["feature"],
        orientation="h",
        marker_color=[BLUE]*3 + ["#0077aa"]*max(0, len(fi_df)-3),
    ))
    fig.update_layout(title="Feature Importance", yaxis=dict(autorange="reversed"))
    return _layout(fig, h=350, ml=180)


def winrate_chart(df, team):
    mask = (df["team_a"] == team) | (df["team_b"] == team)
    sub  = df.loc[mask].sort_values("date").copy()
    # winner enthält echte Teamnamen (nach kaggle_loader-Fix)
    # Fallback: wenn winner noch "team1"/"team2" enthält → via team_a_won
    if "team_a_won" in sub.columns and sub["winner"].isin(["team1","team2"]).mean() > 0.5:
        sub["won"] = np.where(
            sub["team_a"] == team, sub["team_a_won"], 1 - sub["team_a_won"]
        ).astype(int)
    else:
        sub["won"] = (sub["winner"] == team).astype(int)
    sub["rolling_wr"] = sub["won"].rolling(20, min_periods=5).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["rolling_wr"], mode="lines",
                             line=dict(color=GREEN, width=2),
                             fill="tozeroy", fillcolor="rgba(0,255,159,0.06)"))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#555")
    fig.update_layout(title=f"{team} — Winrate (Rolling 20)",
                      yaxis=dict(tickformat=".0%", range=[0, 1]))
    return _layout(fig, h=260)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    model, elo_ratings, map_models = load_model_cached()
    df_feat, df_raw                = load_data()
    teams                          = get_teams(df_feat, df_raw)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🎯 CS2 Predictor")
        st.markdown("---")
        page = st.radio("Navigation", [
            "🏠 Match Prediction",
            "🗺️ Map Prediction",
            "📅 Upcoming Matches",
            "📊 Model Performance",
            "👥 Team Analyse",
            "📋 Match Historie",
            "⚙️ Setup",
        ])
        st.markdown("---")
        st.session_state.setdefault("model_load_time", __import__("time").time())

        # ── Update-Button ─────────────────────────────────────────────────────
        last_update_ts = _load_last_update_time()
        if last_update_ts:
            import datetime as _dt
            last_str = last_update_ts.strftime("%d.%m.%Y %H:%M")
            days_since = (_dt.datetime.now() - last_update_ts).days
            st.caption(f"🕐 Letztes Update: {last_str}")
        else:
            days_since = 30
            st.caption("🕐 Noch kein Update")

        if st.button("⬆️ Daten & Modell aktualisieren", type="primary", use_container_width=True):
            _run_dashboard_update(days_since if days_since > 0 else 1)

        # Modell-Update-Banner
        if _check_model_updated():
            st.warning("⚡ Neues Modell verfügbar!")
            if st.button("🔄 Modell neu laden", use_container_width=True):
                st.cache_resource.clear()
                st.cache_data.clear()
                st.session_state["model_load_time"] = __import__("time").time()
                st.rerun()

        st.markdown("---")
        if model:
            st.success("✅ Match-Modell geladen")
            if map_models:
                st.success(f"✅ Map-Modelle: {len(map_models)}")
            else:
                st.warning("⚠️ Keine Map-Modelle")
        else:
            st.error("❌ Kein Modell – bitte train.py ausführen")
        if df_feat is not None:
            st.info(f"📦 {len(df_feat)} Matches | {len(teams)} Teams")

    # ═════════════════════════════════════════════════════════════════════════
    # SEITE: Match Prediction
    # ═════════════════════════════════════════════════════════════════════════
    if page == "🏠 Match Prediction":
        st.title("🎯 Match Prediction")

        if model is None:
            st.error("Kein Modell gefunden. Führe aus:")
            st.code("python train.py --csv dein_dataset.csv")
            return

        c1, c2 = st.columns(2)
        team_a = c1.selectbox("🟦 Team A", teams, index=0)
        team_b = c2.selectbox("🟥 Team B", [t for t in teams if t != team_a], index=0)

        if st.button("⚡ Prediction berechnen", type="primary", use_container_width=True):
            df_src = df_raw if df_raw is not None else df_feat
            with st.spinner("Berechne ..."):
                from utils.features import build_prediction_features
                from models.trainer import predict_match
                feats  = build_prediction_features(team_a, team_b, df_src, elo_ratings)
                result = predict_match(model, feats)

            prob_a = result["prob_a"]
            prob_b = result["prob_b"]
            winner = team_a if prob_a >= prob_b else team_b
            conf   = abs(prob_a - 0.5) * 2

            conf_label = (
                "🟢 Sehr sicher" if conf >= 0.5 else
                "🔵 Sicher"      if conf >= 0.3 else
                "🟡 Leicht"      if conf >= 0.15 else
                "⚪ Unentschieden"
            )

            st.markdown("---")
            g1, g2, g3 = st.columns([2, 3, 2])
            g1.metric(f"🟦 {team_a}", f"{prob_a:.1%}")
            g2.plotly_chart(gauge_chart(prob_a, team_a, team_b), use_container_width=True)
            g3.metric(f"🟥 {team_b}", f"{prob_b:.1%}")

            m1, m2, m3 = st.columns(3)
            m1.metric("🏆 Vorhergesagter Sieger", winner)
            m2.metric("📊 Gewinnchance",           f"{max(prob_a, prob_b):.1%}")
            m3.metric("🎯 Konfidenz",              conf_label)

            with st.expander("🔍 Feature-Details"):
                rows = [{"Feature": k, "Wert (Team A − B)": f"{v:+.4f}"}
                        for k, v in sorted(feats.items())]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # H2H Tabelle
            if df_src is not None:
                mask = (
                    ((df_src["team_a"] == team_a) & (df_src["team_b"] == team_b)) |
                    ((df_src["team_a"] == team_b) & (df_src["team_b"] == team_a))
                )
                h2h = df_src.loc[mask].sort_values("date", ascending=False).head(8)
                if len(h2h) > 0:
                    with st.expander(f"📋 Letzte H2H-Matches ({len(h2h)})"):
                        show_cols = [c for c in ["date","team_a","score_a","score_b","team_b","winner","event"]
                                     if c in h2h.columns]
                        d = h2h[show_cols].copy()
                        if "date" in d.columns:
                            d["date"] = d["date"].dt.strftime("%Y-%m-%d")
                        st.dataframe(d, use_container_width=True, hide_index=True)

    # ═════════════════════════════════════════════════════════════════════════
    # SEITE: Map Prediction
    # ═════════════════════════════════════════════════════════════════════════
    elif page == "🗺️ Map Prediction":
        st.title("🗺️ Map Prediction")
        st.markdown("Welches Team hat auf welcher Map die bessere Gewinnchance?")

        if model is None:
            st.error("Kein Modell. Bitte train.py ausführen.")
            return
        if not map_models:
            st.warning("Keine Map-Modelle gefunden. Trainiere mit: `python train.py --csv dein.csv`")
            st.info("Das Map-Modell wird automatisch mittrainiert, solange `--no-map` nicht gesetzt ist.")
            return

        c1, c2 = st.columns(2)
        team_a = c1.selectbox("🟦 Team A", teams, key="map_ta")
        team_b = c2.selectbox("🟥 Team B", [t for t in teams if t != team_a], key="map_tb")

        if st.button("🗺️ Map-Predictions berechnen", type="primary", use_container_width=True):
            df_src = df_raw if df_raw is not None else df_feat
            with st.spinner("Berechne Map-Wahrscheinlichkeiten ..."):
                from utils.features import build_prediction_features
                from models.trainer import predict_match, predict_maps
                feats      = build_prediction_features(team_a, team_b, df_src, elo_ratings)
                match_res  = predict_match(model, feats)
                map_probs  = predict_maps(map_models, feats)

            st.markdown("---")

            # Match-Übersicht
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric(f"🟦 {team_a} (Match)", f"{match_res['prob_a']:.1%}")
            mc2.metric("Favoritenstatus",
                       f"{'🟦 ' + team_a if match_res['prob_a'] > 0.5 else '🟥 ' + team_b}")
            mc3.metric(f"🟥 {team_b} (Match)", f"{match_res['prob_b']:.1%}")

            # Map-Balkendiagramm
            st.plotly_chart(map_bar_chart(map_probs, team_a, team_b),
                            use_container_width=True)

            # Map-Tabelle
            st.subheader("Map-Übersicht")
            map_rows = []
            for m, p in sorted(map_probs.items(), key=lambda x: -x[1]):
                fav  = f"🟦 {team_a}" if p > 0.5 else (f"🟥 {team_b}" if p < 0.5 else "Equal")
                diff = abs(p - 0.5) * 2
                sig  = "⚡ Stark" if diff >= 0.4 else ("✅ Deutlich" if diff >= 0.2 else "~Neutral")
                map_rows.append({
                    "Map":         m.capitalize(),
                    f"{team_a}":   f"{p:.1%}",
                    f"{team_b}":   f"{1-p:.1%}",
                    "Favorit":     fav,
                    "Vorteil":     sig,
                })
            st.dataframe(pd.DataFrame(map_rows), use_container_width=True, hide_index=True)

            # Beste/Schlechteste Map für Team A
            best_map  = max(map_probs, key=map_probs.get)
            worst_map = min(map_probs, key=map_probs.get)
            i1, i2 = st.columns(2)
            i1.info(f"✅ Beste Map für **{team_a}**: **{best_map.capitalize()}** ({map_probs[best_map]:.1%})")
            i2.warning(f"⚠️ Schwächste Map für **{team_a}**: **{worst_map.capitalize()}** ({map_probs[worst_map]:.1%})")

            # Map-Winrate-Historie
            df_src2 = df_raw if df_raw is not None else df_feat
            if df_src2 is not None:
                with st.expander("📊 Map-Winrate aus Historischen Daten"):
                    map_hist_rows = []
                    for m in CS2_MAPS:
                        col_a = f"team1_{m}_winrate"
                        col_b = f"team2_{m}_winrate"
                        if col_a in df_src2.columns:
                            wr_a = df_src2[df_src2["team_a"] == team_a][col_a].dropna()
                            wr_b = df_src2[df_src2["team_a"] == team_b][col_b].dropna()
                            if len(wr_a) > 0 or len(wr_b) > 0:
                                def _fmt(v):
                                    if len(v) == 0: return "n/a"
                                    m_val = v.mean()
                                    return f"{m_val/100:.1%}" if m_val > 1 else f"{m_val:.1%}"
                                map_hist_rows.append({
                                    "Map": m.capitalize(),
                                    f"{team_a} Hist. Winrate": _fmt(wr_a),
                                    f"{team_b} Hist. Winrate": _fmt(wr_b),
                                })
                    if map_hist_rows:
                        st.dataframe(pd.DataFrame(map_hist_rows), use_container_width=True, hide_index=True)

    # ═════════════════════════════════════════════════════════════════════════
    # SEITE: Upcoming Matches
    # ═════════════════════════════════════════════════════════════════════════
    elif page == "📅 Upcoming Matches":
        st.title("📅 Upcoming Matches")
        st.markdown("Kommende CS2 Pro-Matches mit ML-Prediction.")

        from config import DATA_DIR
        upcoming_csv = DATA_DIR / "upcoming_matches.csv"

        if st.button("🔄 Daten aktualisieren (PandaScore)", use_container_width=False):
            with st.spinner("Lade von PandaScore ..."):
                try:
                    from data.api_fetcher import fetch_upcoming_matches, save_upcoming
                    upcoming_new = fetch_upcoming_matches()
                    save_upcoming(upcoming_new)
                    st.success(f"{len(upcoming_new)} Matches geladen!")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Fehler: {e}. Prüfe PANDASCORE_API_KEY in .env")

        if not upcoming_csv.exists():
            st.info("Noch keine Upcoming-Daten.")
            st.code("python data/api_fetcher.py --fetch-upcoming")
        else:
            df_up = pd.read_csv(upcoming_csv, parse_dates=["date"])
            df_up = df_up[df_up["date"] >= pd.Timestamp.now() - pd.Timedelta(hours=2)]
            df_up = df_up.sort_values("date")

            if len(df_up) == 0:
                st.info("Keine kommenden Matches. Bitte aktualisieren.")
            else:
                st.info(f"{len(df_up)} kommende Matches")
                pred_rows = []
                df_src = df_raw if df_raw is not None else df_feat

                for _, row in df_up.iterrows():
                    ta, tb = str(row.get("team_a","?")), str(row.get("team_b","?"))
                    prob_a, prob_b = 0.5, 0.5
                    if model is not None and ta not in ("TBD","?") and tb not in ("TBD","?") and df_src is not None:
                        try:
                            from utils.features import build_prediction_features
                            from models.trainer import predict_match as _pm
                            feats  = build_prediction_features(ta, tb, df_src, elo_ratings)
                            res    = _pm(model, feats)
                            prob_a, prob_b = res["prob_a"], res["prob_b"]
                        except Exception:
                            pass

                    conf  = abs(prob_a - 0.5) * 2
                    fav   = ta if prob_a >= prob_b else tb
                    cl    = ("⚡ Sehr sicher" if conf >= 0.5 else
                             "✅ Sicher"      if conf >= 0.3 else "~ Neutral")
                    pred_rows.append({
                        "Datum":     row["date"].strftime("%m-%d %H:%M") if pd.notna(row["date"]) else "",
                        "Team A":    ta,
                        "Win% A":    f"{prob_a:.1%}",
                        "Team B":    tb,
                        "Win% B":    f"{prob_b:.1%}",
                        "Favorit":   fav,
                        "Chance":    f"{max(prob_a,prob_b):.1%}",
                        "Konfidenz": cl,
                        "Event":     str(row.get("event",""))[:30],
                    })

                st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)

    # ═════════════════════════════════════════════════════════════════════════
    # SEITE: Model Performance
    # ═════════════════════════════════════════════════════════════════════════
    elif page == "📊 Model Performance":
        st.title("📊 Model Performance")

        bt_path = Path("data/backtest_results.csv")
        if not bt_path.exists():
            st.warning("Kein Backtest gefunden. Führe `python train.py --csv ...` aus.")
            return

        bt = pd.read_csv(bt_path, parse_dates=["date"])
        acc   = bt["correct"].mean()
        n_hc  = len(bt[bt["confidence"] >= 0.5])
        acc_hc= bt[bt["confidence"] >= 0.5]["correct"].mean() if n_hc > 0 else 0

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Gesamt-Accuracy",    f"{acc:.1%}")
        c2.metric("Val.-Matches",       len(bt))
        c3.metric("High-Conf Accuracy", f"{acc_hc:.1%}")
        c4.metric("High-Conf Matches",  n_hc)

        st.plotly_chart(accuracy_chart(bt), use_container_width=True)

        if model:
            from models.trainer import get_feature_importance
            st.plotly_chart(fi_chart(get_feature_importance(model)), use_container_width=True)

        st.subheader("Accuracy nach Konfidenz-Level")
        rows = []
        for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            sub = bt[bt["confidence"] >= t]
            if len(sub) > 0:
                rows.append({"Min. Konfidenz": f"{t:.0%}", "Matches": len(sub),
                             "Accuracy": f"{sub['correct'].mean():.1%}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ═════════════════════════════════════════════════════════════════════════
    # SEITE: Team Analyse
    # ═════════════════════════════════════════════════════════════════════════
    elif page == "👥 Team Analyse":
        st.title("👥 Team Analyse")
        if not teams:
            st.warning("Keine Teams gefunden.")
            return

        team = st.selectbox("Team", teams)
        df_s = df_raw if df_raw is not None else df_feat
        if df_s is None:
            return

        mask = (df_s["team_a"] == team) | (df_s["team_b"] == team)
        tm   = df_s.loc[mask].copy()
        if "team_a_won" in tm.columns and tm["winner"].isin(["team1","team2"]).mean() > 0.5:
            tm["won"] = np.where(
                tm["team_a"] == team, tm["team_a_won"], 1 - tm["team_a_won"]
            ).astype(int)
        else:
            tm["won"] = (tm["winner"] == team).astype(int)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Matches",  len(tm))
        c2.metric("Siege",    int(tm["won"].sum()))
        c3.metric("Winrate",  f"{tm['won'].mean():.1%}" if len(tm) > 0 else "n/a")
        c4.metric("ELO",      f"{elo_ratings.get(team, 1500):.0f}")

        last5 = tm.tail(5)["won"].tolist()
        st.markdown("**Letzte 5:** " + " ".join(["✅" if w else "❌" for w in reversed(last5)]))

        st.plotly_chart(winrate_chart(df_s, team), use_container_width=True)

        # Map-Winrates
        map_cols = [f"team1_{m}_winrate" for m in CS2_MAPS if f"team1_{m}_winrate" in df_s.columns]
        if map_cols:
            st.subheader("Map-Winrates (historisch)")
            team_rows = df_s[df_s["team_a"] == team]
            map_data  = {}
            for m in CS2_MAPS:
                col = f"team1_{m}_winrate"
                if col in team_rows.columns:
                    v = team_rows[col].dropna()
                    if len(v) > 0:
                        val = v.mean()
                        # Normalisieren: Werte > 1 sind in Prozent (0-100), auf 0-1 umrechnen
                        if val > 1:
                            val = val / 100
                        map_data[m.capitalize()] = f"{val:.1%}"
            if map_data:
                st.dataframe(
                    pd.DataFrame(map_data.items(), columns=["Map", "Winrate"]),
                    use_container_width=True, hide_index=True
                )

        # Top-Gegner
        st.subheader("Gegner-Statistik (Top 10)")
        rows = []
        for opp in df_s["team_a"].unique():
            if opp == team: continue
            sub = df_s.loc[mask & ((df_s["team_a"] == opp) | (df_s["team_b"] == opp))]
            if len(sub) < 2: continue
            if "team_a_won" in sub.columns and sub["winner"].isin(["team1","team2"]).mean() > 0.5:
                w = np.where(sub["team_a"] == team, sub["team_a_won"], 1 - sub["team_a_won"]).sum()
            else:
                w = (sub["winner"] == team).sum()
            rows.append({"Gegner": opp, "Matches": len(sub),
                         "Siege": int(w), "Winrate": f"{w/len(sub):.1%}"})
        if rows:
            odf = pd.DataFrame(rows).sort_values("Matches", ascending=False).head(10)
            st.dataframe(odf, use_container_width=True, hide_index=True)

    # ═════════════════════════════════════════════════════════════════════════
    # SEITE: Match Historie
    # ═════════════════════════════════════════════════════════════════════════
    elif page == "📋 Match Historie":
        st.title("📋 Match Historie")
        df_s = df_raw if df_raw is not None else df_feat
        if df_s is None:
            st.warning("Keine Daten.")
            return

        c1,c2,c3 = st.columns(3)
        search = c1.text_input("Team-Filter")
        d_from = c2.date_input("Von", value=None)
        d_to   = c3.date_input("Bis", value=None)

        f = df_s.copy()
        if search:
            f = f[f["team_a"].str.contains(search, case=False, na=False) |
                  f["team_b"].str.contains(search, case=False, na=False)]
        if d_from: f = f[f["date"] >= pd.Timestamp(d_from)]
        if d_to:   f = f[f["date"] <= pd.Timestamp(d_to)]
        f = f.sort_values("date", ascending=False)

        st.info(f"{len(f)} Matches")
        cols = [c for c in ["date","team_a","score_a","score_b","team_b","winner","event","match_type"]
                if c in f.columns]
        d = f[cols].head(300).copy()
        if "date" in d.columns:
            d["date"] = d["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(d, use_container_width=True, hide_index=True)

    # ═════════════════════════════════════════════════════════════════════════
    # SEITE: Setup
    # ═════════════════════════════════════════════════════════════════════════
    elif page == "⚙️ Setup":
        st.title("⚙️ Setup")

        st.subheader("Schnellstart mit Kaggle-CSV")
        st.code("""
# 1. Abhängigkeiten
pip install -r requirements.txt

# 2. Kaggle-CSV laden und Modell trainieren (inkl. Map-Modelle)
python train.py --csv dein_kaggle_dataset.csv

# 3. Mit Hyperparameter-Tuning (besser, dauert länger)
python train.py --csv dein_kaggle_dataset.csv --tune --trials 50

# 4. Dashboard starten
streamlit run dashboard/app.py
        """, language="bash")

        st.subheader("Datei-Status")
        from config import MODEL_PATH, MAP_MODEL_PATH, ELO_PATH
        status = [
            {"Datei": str(RAW_CSV),          "Status": "✅" if RAW_CSV.exists()          else "❌"},
            {"Datei": str(FEATURES_CSV),      "Status": "✅" if FEATURES_CSV.exists()     else "❌"},
            {"Datei": str(MODEL_PATH),        "Status": "✅" if MODEL_PATH.exists()       else "❌"},
            {"Datei": str(MAP_MODEL_PATH),    "Status": "✅" if MAP_MODEL_PATH.exists()   else "❌"},
            {"Datei": str(ELO_PATH),          "Status": "✅" if ELO_PATH.exists()         else "❌"},
        ]
        st.dataframe(pd.DataFrame(status), use_container_width=True, hide_index=True)

        st.subheader("Erwartete Spalten im Kaggle-CSV (Auszug)")
        st.code("""
match_id, date, tournament, winner,
team1_name, team2_name,
score_team1, score_team2,
rating_diff, adr_diff, kast_diff, kpr_diff, dpr_diff,
team1_avg_RATING, team2_avg_RATING,
winner_head2head_percentage, loser_head2head_percentage,
winner_past3, loser_past3,
winner_mirage, loser_mirage,   # Map-Winrates
winner_inferno, loser_inferno,
team1_overall_winrate, team2_overall_winrate,
team1_lan_winrate, team2_lan_winrate,
star_player_advantage, weakest_link_advantage,
team1_rating_std, team2_rating_std, consistency_advantage
        """)


if __name__ == "__main__":
    main()