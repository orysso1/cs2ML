# dashboard/value_betting.py
# Value-Betting-Modul für das CS2-Dashboard.
#
# Konzept:
#   Value Bet = wenn unsere Modell-Wahrscheinlichkeit HÖHER ist als die
#               implizite Wahrscheinlichkeit der Buchmacher-Quote
#
#   Implizite Wahrscheinlichkeit = 1 / Quote
#   Expected Value (EV) = (prob_model * quote) - 1
#   EV > 0  → Value Bet (lohnt sich langfristig)
#   EV > 0.05 → Gute Value Bet
#   EV > 0.10 → Sehr gute Value Bet
#
# Kelly-Kriterium (optimale Einsatzgröße):
#   f = (prob * quote - 1) / (quote - 1)
#   Empfehlung: 1/4 Kelly (konservativer)

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Konstanten
# ─────────────────────────────────────────────────────────────────────────────

BETS_FILE = Path(__file__).parent.parent / "data" / "bets_log.json"

EV_THRESHOLDS = {
    "Sehr gut":  0.10,
    "Gut":       0.05,
    "Schwach":   0.02,
    "Kein Value": 0.0,
}

BOOKMAKERS = [
    "Bet365", "Unibet", "Bwin", "Betway",
    "William Hill", "Pinnacle", "1xBet", "Sonstige"
]

DARK  = "#0a0f1a"
PANEL = "#0d1b2a"
GREEN = "#00ff9f"
RED   = "#ff4444"
BLUE  = "#00d4ff"
ORANGE= "#ff6b00"
WARN  = "#ffcc00"


# ─────────────────────────────────────────────────────────────────────────────
# Kernberechnungen
# ─────────────────────────────────────────────────────────────────────────────

def implied_prob(odds: float) -> float:
    """Implizite Wahrscheinlichkeit aus europäischer Dezimalquote."""
    if odds <= 1.0:
        return 1.0
    return 1.0 / odds


def expected_value(prob_model: float, odds: float) -> float:
    """
    EV = (Modell-Wahrscheinlichkeit × Quote) - 1
    EV > 0 = Value Bet (Buchmacher unterschätzt das Team)
    """
    return (prob_model * odds) - 1.0


def kelly_fraction(prob_model: float, odds: float, fraction: float = 0.25) -> float:
    """
    Kelly-Kriterium: optimale Einsatzgröße als Anteil des Bankrolls.
    fraction=0.25 = Quarter Kelly (konservativ, empfohlen).
    """
    if odds <= 1.0:
        return 0.0
    b = odds - 1.0          # Nettogewinn pro eingesetzter Einheit
    q = 1.0 - prob_model    # Verlustwahrscheinlichkeit
    k = (prob_model * b - q) / b
    return max(0.0, k * fraction)


def ev_label(ev: float) -> tuple[str, str]:
    """Gibt (Label, Farbe) für einen EV-Wert zurück."""
    if ev >= EV_THRESHOLDS["Sehr gut"]:
        return "🟢 Sehr gut", GREEN
    elif ev >= EV_THRESHOLDS["Gut"]:
        return "🔵 Gut", BLUE
    elif ev >= EV_THRESHOLDS["Schwach"]:
        return "🟡 Schwach", WARN
    else:
        return "🔴 Kein Value", RED


def analyze_odds(
    team_a: str, team_b: str,
    prob_a: float, prob_b: float,
    odds_a: float, odds_b: float,
    bankroll: float = 100.0,
) -> dict:
    """
    Vollständige Value-Analyse für ein Match.
    Gibt Dict mit allen relevanten Kennzahlen zurück.
    """
    imp_a = implied_prob(odds_a)
    imp_b = implied_prob(odds_b)
    margin = (imp_a + imp_b - 1.0)   # Buchmacher-Margin (Vig)

    ev_a = expected_value(prob_a, odds_a)
    ev_b = expected_value(prob_b, odds_b)

    kelly_a = kelly_fraction(prob_a, odds_a)
    kelly_b = kelly_fraction(prob_b, odds_b)

    stake_a = bankroll * kelly_a
    stake_b = bankroll * kelly_b

    label_a, color_a = ev_label(ev_a)
    label_b, color_b = ev_label(ev_b)

    # Beste Wette bestimmen
    if ev_a > ev_b and ev_a > 0:
        best_team  = team_a
        best_ev    = ev_a
        best_odds  = odds_a
        best_stake = stake_a
        best_kelly = kelly_a
    elif ev_b > ev_a and ev_b > 0:
        best_team  = team_b
        best_ev    = ev_b
        best_odds  = odds_b
        best_stake = stake_b
        best_kelly = kelly_b
    else:
        best_team  = None
        best_ev    = max(ev_a, ev_b)
        best_odds  = None
        best_stake = 0.0
        best_kelly = 0.0

    return {
        "team_a": team_a, "team_b": team_b,
        "prob_a": prob_a, "prob_b": prob_b,
        "odds_a": odds_a, "odds_b": odds_b,
        "imp_a":  imp_a,  "imp_b":  imp_b,
        "margin": margin,
        "ev_a":   ev_a,   "ev_b":   ev_b,
        "label_a": label_a, "color_a": color_a,
        "label_b": label_b, "color_b": color_b,
        "kelly_a": kelly_a, "kelly_b": kelly_b,
        "stake_a": stake_a, "stake_b": stake_b,
        "best_team":  best_team,
        "best_ev":    best_ev,
        "best_odds":  best_odds,
        "best_stake": best_stake,
        "best_kelly": best_kelly,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Wetten-Logbuch
# ─────────────────────────────────────────────────────────────────────────────

def load_bets() -> list[dict]:
    if not BETS_FILE.exists():
        return []
    try:
        return json.loads(BETS_FILE.read_text())
    except Exception:
        return []


def save_bet(bet: dict):
    bets = load_bets()
    bet["id"]        = len(bets) + 1
    bet["timestamp"] = datetime.now().isoformat()
    bet["result"]    = "offen"
    bets.append(bet)
    BETS_FILE.write_text(json.dumps(bets, indent=2, ensure_ascii=False))


def update_bet_result(bet_id: int, result: str, actual_winner: str = ""):
    bets = load_bets()
    for b in bets:
        if b["id"] == bet_id:
            b["result"]        = result   # "gewonnen" / "verloren"
            b["actual_winner"] = actual_winner
            b["closed_at"]     = datetime.now().isoformat()
            # P&L berechnen
            if result == "gewonnen":
                b["pnl"] = b["stake"] * (b["odds"] - 1)
            elif result == "verloren":
                b["pnl"] = -b["stake"]
            else:
                b["pnl"] = 0
            break
    BETS_FILE.write_text(json.dumps(bets, indent=2, ensure_ascii=False))


def delete_bet(bet_id: int):
    """Löscht eine Wette aus dem Log."""
    bets = load_bets()
    bets = [b for b in bets if b["id"] != bet_id]
    BETS_FILE.write_text(json.dumps(bets, indent=2, ensure_ascii=False))


def bet_stats(bets: list[dict]) -> dict:
    closed = [b for b in bets if b["result"] in ("gewonnen", "verloren")]
    if not closed:
        return {"total": len(bets), "closed": 0, "won": 0, "lost": 0,
                "roi": 0.0, "pnl": 0.0, "winrate": 0.0, "avg_ev": 0.0}
    won    = sum(1 for b in closed if b["result"] == "gewonnen")
    pnl    = sum(b.get("pnl", 0) for b in closed)
    staked = sum(b.get("stake", 0) for b in closed)
    avg_ev = np.mean([b.get("ev", 0) for b in closed]) if closed else 0
    return {
        "total":   len(bets),
        "closed":  len(closed),
        "open":    len(bets) - len(closed),
        "won":     won,
        "lost":    len(closed) - won,
        "pnl":     pnl,
        "roi":     (pnl / staked * 100) if staked > 0 else 0.0,
        "winrate": won / len(closed) if closed else 0.0,
        "avg_ev":  avg_ev,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────

def ev_gauge(ev: float, team: str) -> go.Figure:
    color = GREEN if ev >= 0.05 else (WARN if ev >= 0 else RED)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(ev * 100, 2),
        number={"suffix": "%", "font": {"size": 32, "color": color}},
        delta={"reference": 0, "valueformat": ".1f"},
        gauge={
            "axis":    {"range": [-30, 30], "tickcolor": "#2a4a6a"},
            "bar":     {"color": color, "thickness": 0.3},
            "bgcolor": PANEL, "bordercolor": "#1a3a5c",
            "steps": [
                {"range": [-30,  0], "color": "#1a0505"},
                {"range": [  0,  5], "color": "#0a1a0a"},
                {"range": [  5, 30], "color": "#0a2a0a"},
            ],
            "threshold": {"line": {"color": "#fff", "width": 2},
                          "thickness": 0.8, "value": 5},
        },
        title={"text": f"EV — {team}", "font": {"color": "#a0c0d0", "size": 13}},
    ))
    fig.update_layout(height=220, margin=dict(l=20,r=20,t=40,b=10),
                      paper_bgcolor=DARK, font_color="#c0d8e8")
    return fig


def pnl_chart(bets: list[dict]) -> go.Figure:
    closed = sorted(
        [b for b in bets if b["result"] in ("gewonnen", "verloren")],
        key=lambda x: x.get("timestamp", "")
    )
    if not closed:
        return None

    cumulative = 0
    dates, values, colors = [], [], []
    for b in closed:
        cumulative += b.get("pnl", 0)
        dates.append(b.get("timestamp", "")[:10])
        values.append(cumulative)
        colors.append(GREEN if cumulative >= 0 else RED)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=values, mode="lines+markers",
        line=dict(color=BLUE, width=2),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.07)",
        marker=dict(color=colors, size=8),
        name="Kumulativer P&L",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#555")
    fig.update_layout(
        title="Kumulativer Gewinn/Verlust",
        yaxis_title="P&L (€)",
        paper_bgcolor=DARK, plot_bgcolor=PANEL,
        font_color="#a0c0d0", height=280,
        margin=dict(l=20,r=20,t=40,b=20),
    )
    return fig


def odds_comparison_chart(result: dict) -> go.Figure:
    """Vergleicht Modell-Wahrscheinlichkeit vs. implizite Buchmacher-Wahrscheinlichkeit."""
    teams    = [result["team_a"], result["team_b"]]
    model    = [result["prob_a"] * 100, result["prob_b"] * 100]
    implied  = [result["imp_a"]  * 100, result["imp_b"]  * 100]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Modell",      x=teams, y=model,
                         marker_color=BLUE,   text=[f"{v:.1f}%" for v in model],
                         textposition="outside"))
    fig.add_trace(go.Bar(name="Buchmacher", x=teams, y=implied,
                         marker_color=ORANGE, text=[f"{v:.1f}%" for v in implied],
                         textposition="outside"))
    fig.update_layout(
        title="Modell vs. Buchmacher Wahrscheinlichkeit",
        barmode="group",
        yaxis=dict(range=[0, 110], ticksuffix="%"),
        paper_bgcolor=DARK, plot_bgcolor=PANEL,
        font_color="#a0c0d0", height=280,
        margin=dict(l=20,r=20,t=40,b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Hauptseite (wird von app.py aufgerufen)
# ─────────────────────────────────────────────────────────────────────────────

def render(model, elo_ratings, df_raw, df_feat, teams):
    st.title("💰 Value Betting")
    st.markdown(
        "Gib die aktuellen Buchmacher-Quoten ein. Das Modell berechnet "
        "automatisch ob eine **Value Bet** vorliegt."
    )

    st.info(
        "**Was ist eine Value Bet?** Eine Wette bei der die Buchmacher-Quote "
        "höher ist als sie sein müsste — d.h. der Buchmacher unterschätzt die "
        "Gewinnchance. Langfristig profitabel wenn EV > 0."
    )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_analyse, tab_log, tab_stats = st.tabs([
        "🔍 Analyse", "📋 Wetten-Log", "📈 Statistik"
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1: Analyse
    # ═══════════════════════════════════════════════════════════════════════
    with tab_analyse:
        if model is None:
            st.error("Kein Modell geladen. Bitte `python train.py` ausführen.")
            return

        # Bankroll-Einstellung
        with st.expander("⚙️ Einstellungen", expanded=False):
            c1, c2, c3 = st.columns(3)
            bankroll    = c1.number_input("Bankroll (€)", min_value=10.0,
                                          max_value=100000.0, value=100.0, step=10.0)
            kelly_frac  = c2.select_slider("Kelly-Fraktion",
                                           options=[0.1, 0.25, 0.5, 1.0],
                                           value=0.25,
                                           format_func=lambda x: f"{x:.0%} Kelly")
            min_ev      = c3.slider("Min. EV für Empfehlung", 0.0, 0.20, 0.05, 0.01,
                                    format="%.0f%%")

        st.markdown("---")

        # ── Match-Auswahl: Upcoming oder manuell ──────────────────────────────
        upcoming_csv = Path(__file__).parent.parent / "data" / "upcoming_matches.csv"
        df_up = None
        if upcoming_csv.exists():
            try:
                _df = pd.read_csv(upcoming_csv, parse_dates=["date"])
                _df = _df[
                    (_df["team_a"].notna()) & (_df["team_b"].notna()) &
                    (_df["team_a"] != "TBD")  & (_df["team_b"] != "TBD") &
                    (_df["date"] >= pd.Timestamp.now() - pd.Timedelta(hours=3))
                ].sort_values("date")
                if len(_df) > 0:
                    df_up = _df
            except Exception:
                pass

        # Toggle: Upcoming vs. Manuell
        if df_up is not None:
            source_mode = st.radio(
                "Match auswählen",
                ["📅 Aus Upcoming Matches", "✏️ Manuell"],
                horizontal=True,
                key="vb_source_mode",
            )
        else:
            source_mode = "✏️ Manuell"

        team_a, team_b = None, None

        if source_mode == "📅 Aus Upcoming Matches" and df_up is not None:
            # Upcoming-Dropdown
            match_options = {
                f"{row['date'].strftime('%m-%d %H:%M')}  —  {row['team_a']}  vs  {row['team_b']}"
                f"  ({row.get('event','')})": (row["team_a"], row["team_b"])
                for _, row in df_up.iterrows()
            }
            selected = st.selectbox(
                "Kommendes Match",
                list(match_options.keys()),
                key="vb_upcoming_select",
            )
            team_a, team_b = match_options[selected]
            st.markdown(
                f"**🟦 {team_a}** &nbsp;&nbsp; vs &nbsp;&nbsp; **🟥 {team_b}**"
            )
        else:
            # Manuelle Auswahl
            c1, c2 = st.columns(2)
            team_a = c1.selectbox("🟦 Team A", teams, key="vb_ta")
            team_b = c2.selectbox("🟥 Team B",
                                  [t for t in teams if t != team_a], key="vb_tb")

        # ── Quoten-Eingabe ─────────────────────────────────────────────────────
        st.markdown("#### Buchmacher-Quoten eingeben")
        st.caption("Dezimalquoten (z.B. 1.85 bedeutet: 1€ einsetzen → 1.85€ zurück bei Sieg)")

        c1, c2, c3 = st.columns([3, 3, 2])
        odds_a     = c1.number_input(f"Quote {team_a}", min_value=1.01,
                                     max_value=50.0, value=1.90, step=0.05,
                                     key="odds_a")
        odds_b     = c2.number_input(f"Quote {team_b}", min_value=1.01,
                                     max_value=50.0, value=1.95, step=0.05,
                                     key="odds_b")
        bookmaker  = c3.selectbox("Buchmacher", BOOKMAKERS, key="bm")

        if st.button("📊 Value analysieren", type="primary", use_container_width=True):

            df_src = df_raw if df_raw is not None else df_feat
            with st.spinner("Berechne Modell-Prediction ..."):
                try:
                    from utils.features import build_prediction_features
                    from models.trainer import predict_match
                    feats  = build_prediction_features(team_a, team_b,
                                                       df_src, elo_ratings)
                    res    = predict_match(model, feats)
                    prob_a = res["prob_a"]
                    prob_b = res["prob_b"]
                except Exception as e:
                    st.error(f"Prediction-Fehler: {e}")
                    return

            result = analyze_odds(
                team_a, team_b, prob_a, prob_b,
                odds_a, odds_b, bankroll
            )
            # Kelly-Fraktion aus Einstellung übernehmen
            result["kelly_a"] = kelly_fraction(prob_a, odds_a, kelly_frac)
            result["kelly_b"] = kelly_fraction(prob_b, odds_b, kelly_frac)
            result["stake_a"] = bankroll * result["kelly_a"]
            result["stake_b"] = bankroll * result["kelly_b"]

            st.session_state["vb_result"]    = result
            st.session_state["vb_bookmaker"] = bookmaker
            st.session_state["vb_min_ev"]    = min_ev
            st.session_state["vb_bankroll"]  = bankroll

        # Ergebnis anzeigen
        if "vb_result" in st.session_state:
            result    = st.session_state["vb_result"]
            min_ev    = st.session_state.get("vb_min_ev", 0.05)
            bankroll  = st.session_state.get("vb_bankroll", 100.0)
            bookmaker = st.session_state.get("vb_bookmaker", "")

            st.markdown("---")

            # ── Hauptergebnis ──────────────────────────────────────────────
            if result["best_team"] and result["best_ev"] >= min_ev:
                st.success(
                    f"✅ **VALUE BET gefunden!** "
                    f"Setze auf **{result['best_team']}** "
                    f"@ {result['best_odds']:.2f} "
                    f"(EV: +{result['best_ev']*100:.1f}%)"
                )
            elif result["best_ev"] >= 0:
                st.warning(
                    f"⚠️ Schwache Value Bet auf **{result['best_team']}** "
                    f"(EV: +{result['best_ev']*100:.1f}%) — "
                    f"unter deinem Mindest-EV von {min_ev*100:.0f}%"
                )
            else:
                st.error(
                    "❌ Keine Value Bet — Buchmacher-Quoten bieten keinen Vorteil. "
                    "Nicht wetten."
                )

            # ── Kennzahlen ─────────────────────────────────────────────────
            st.markdown("#### Detailanalyse")
            cols = st.columns(4)
            cols[0].metric("Modell: " + result["team_a"],
                           f"{result['prob_a']:.1%}",
                           f"Impl.: {result['imp_a']:.1%}")
            cols[1].metric("Modell: " + result["team_b"],
                           f"{result['prob_b']:.1%}",
                           f"Impl.: {result['imp_b']:.1%}")
            cols[2].metric("Buchmacher-Margin",
                           f"{result['margin']*100:.1f}%",
                           help="Vig/Overround — je niedriger desto besser")
            cols[3].metric("Bestes EV",
                           f"{result['best_ev']*100:+.1f}%",
                           help="Expected Value — über 5% ist gut")

            # ── EV-Gauges ──────────────────────────────────────────────────
            g1, g2 = st.columns(2)
            g1.plotly_chart(ev_gauge(result["ev_a"], result["team_a"]),
                            use_container_width=True)
            g2.plotly_chart(ev_gauge(result["ev_b"], result["team_b"]),
                            use_container_width=True)

            # ── Vergleichs-Chart ───────────────────────────────────────────
            st.plotly_chart(odds_comparison_chart(result), use_container_width=True)

            # ── Kelly-Tabelle ──────────────────────────────────────────────
            st.markdown("#### Empfohlene Einsätze (Quarter Kelly)")
            kelly_rows = []
            for team, ev, odds, kelly, stake in [
                (result["team_a"], result["ev_a"], result["odds_a"],
                 result["kelly_a"], result["stake_a"]),
                (result["team_b"], result["ev_b"], result["odds_b"],
                 result["kelly_b"], result["stake_b"]),
            ]:
                lbl, _ = ev_label(ev)
                kelly_rows.append({
                    "Team":           team,
                    "Quote":          f"{odds:.2f}",
                    "Modell-Prob.":   f"{result['prob_a' if team==result['team_a'] else 'prob_b']:.1%}",
                    "Impl. Prob.":    f"{result['imp_a' if team==result['team_a'] else 'imp_b']:.1%}",
                    "EV":             f"{ev*100:+.1f}%",
                    "Kelly-Einsatz":  f"{stake:.2f}€" if stake > 0.5 else "—",
                    "Bewertung":      lbl,
                })
            st.dataframe(pd.DataFrame(kelly_rows), use_container_width=True,
                         hide_index=True)

            # ── Wette speichern ────────────────────────────────────────────
            if result["best_team"] and result["best_ev"] >= min_ev:
                st.markdown("---")
                st.markdown("#### Wette im Log speichern")

                custom_stake = st.number_input(
                    "Einsatz (€)",
                    min_value=0.5,
                    max_value=float(bankroll),
                    value=round(result["best_stake"], 2),
                    step=0.5,
                    help="Vorausgefüllt mit Kelly-Empfehlung"
                )

                if st.button("💾 Wette speichern", type="primary"):
                    bet = {
                        "match":      f"{result['team_a']} vs {result['team_b']}",
                        "team":       result["best_team"],
                        "odds":       result["best_odds"],
                        "stake":      custom_stake,
                        "ev":         result["best_ev"],
                        "prob_model": result["prob_a"] if result["best_team"] == result["team_a"] else result["prob_b"],
                        "bookmaker":  bookmaker,
                        "date":       datetime.now().strftime("%Y-%m-%d"),
                    }
                    save_bet(bet)
                    st.success(f"✅ Wette gespeichert: {custom_stake:.2f}€ auf {result['best_team']} @ {result['best_odds']:.2f}")
                    st.cache_data.clear()

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2: Wetten-Log
    # ═══════════════════════════════════════════════════════════════════════
    with tab_log:
        bets = load_bets()

        if not bets:
            st.info("Noch keine Wetten gespeichert.")
        else:
            # Offene Wetten abschließen
            open_bets = [b for b in bets if b["result"] == "offen"]
            if open_bets:
                st.markdown(f"#### Offene Wetten ({len(open_bets)})")
                for bet in open_bets:
                    with st.expander(f"#{bet['id']} — {bet['match']} — {bet['team']} @ {bet['odds']:.2f} — {bet['stake']:.2f}€"):
                        c1, c2, c3 = st.columns(3)
                        c1.write(f"**Datum:** {bet.get('date','')}")
                        c1.write(f"**Buchmacher:** {bet.get('bookmaker','')}")
                        c2.write(f"**EV:** {bet.get('ev',0)*100:+.1f}%")
                        c2.write(f"**Modell-Prob.:** {bet.get('prob_model',0):.1%}")

                        col_w, col_l, col_d = c3.columns(3)
                        if col_w.button("✅ Gewonnen", key=f"open_won_{bet['id']}"):
                            update_bet_result(bet["id"], "gewonnen", bet["team"])
                            st.rerun()
                        if col_l.button("❌ Verloren", key=f"open_lost_{bet['id']}"):
                            update_bet_result(bet["id"], "verloren")
                            st.rerun()
                        if col_d.button("🗑️", key=f"open_del_{bet['id']}",
                                        help="Wette löschen"):
                            delete_bet(bet["id"])
                            st.rerun()

            # Alle Wetten als Tabelle mit Lösch-Option
            st.markdown("#### Alle Wetten")

            # Abgeschlossene Wetten einzeln mit Lösch-Button
            closed_bets = [b for b in bets if b["result"] != "offen"]
            if closed_bets:
                with st.expander(f"Abgeschlossene Wetten ({len(closed_bets)}) — löschen"):
                    for bet in reversed(closed_bets):
                        pnl = bet.get("pnl", 0)
                        pnl_str = f"{pnl:+.2f}€" if isinstance(pnl, (int,float)) else "—"
                        icon = "✅" if bet["result"] == "gewonnen" else "❌"
                        c1, c2 = st.columns([8, 1])
                        c1.write(
                            f"{icon} **#{bet['id']}** — {bet['match']} — "
                            f"{bet['team']} @ {bet['odds']:.2f} — "
                            f"{bet['stake']:.2f}€ — **{pnl_str}**"
                        )
                        if c2.button("🗑️", key=f"del_closed_{bet['id']}",
                                     help="Wette löschen"):
                            delete_bet(bet["id"])
                            st.rerun()

            # Übersichtstabelle
            rows = []
            for b in reversed(bets):
                pnl = b.get("pnl", "—")
                rows.append({
                    "#":          b["id"],
                    "Datum":      b.get("date",""),
                    "Match":      b["match"],
                    "Team":       b["team"],
                    "Quote":      f"{b['odds']:.2f}",
                    "Einsatz":    f"{b['stake']:.2f}€",
                    "EV":         f"{b.get('ev',0)*100:+.1f}%",
                    "Ergebnis":   b["result"],
                    "P&L":        f"{pnl:+.2f}€" if isinstance(pnl, (int,float)) else "—",
                    "Buchmacher": b.get("bookmaker",""),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Alle löschen
            st.markdown("---")
            if st.button("🗑️ Alle Wetten löschen", type="secondary"):
                if st.session_state.get("confirm_delete_all"):
                    BETS_FILE.unlink(missing_ok=True)
                    st.session_state.pop("confirm_delete_all", None)
                    st.success("Alle Wetten gelöscht.")
                    st.rerun()
                else:
                    st.session_state["confirm_delete_all"] = True
                    st.warning("Nochmal klicken zum Bestätigen.")

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 3: Statistik
    # ═══════════════════════════════════════════════════════════════════════
    with tab_stats:
        bets  = load_bets()
        stats = bet_stats(bets)

        if stats["total"] == 0:
            st.info("Noch keine Wetten. Fang mit der Analyse-Tab an.")
            return

        # Kennzahlen
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Gesamt Wetten",  stats["total"])
        c2.metric("Offen",          stats.get("open", 0))
        c3.metric("Winrate",        f"{stats['winrate']:.1%}" if stats["closed"] else "—")
        c4.metric("ROI",            f"{stats['roi']:+.1f}%"   if stats["closed"] else "—")
        c5.metric("Gesamt P&L",     f"{stats['pnl']:+.2f}€"   if stats["closed"] else "—")

        # P&L Chart
        fig = pnl_chart(bets)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # EV vs. tatsächliche Ergebnisse
        closed = [b for b in bets if b["result"] in ("gewonnen","verloren")]
        if len(closed) >= 5:
            st.markdown("#### EV-Kalibrierung")
            st.caption("Zeigt ob der Expected Value mit tatsächlichen Gewinnen übereinstimmt")
            ev_vals  = [b.get("ev",0)*100 for b in closed]
            outcomes = [1 if b["result"]=="gewonnen" else 0 for b in closed]
            cal_df   = pd.DataFrame({"ev": ev_vals, "gewonnen": outcomes})
            cal_df["ev_bucket"] = pd.cut(cal_df["ev"], bins=5)
            cal_grp = cal_df.groupby("ev_bucket")["gewonnen"].agg(["mean","count"]).reset_index()
            cal_grp.columns = ["EV-Bereich", "Tatsächliche Winrate", "Anzahl"]
            st.dataframe(cal_grp, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.caption(
            "⚠️ Hinweis: Wetten birgt finanzielle Risiken. "
            "Setze nur Beträge ein die du bereit bist zu verlieren. "
            "Das Modell ist kein Garant für Gewinne."
        )