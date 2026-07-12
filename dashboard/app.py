# =============================================================
#  A.E.D.A.S. — Dashboard HMI Streamlit
#  dashboard/app.py
#
#  Interface simulant le tableau de bord d'un véhicule :
#  - Probabilité de détection en temps réel
#  - Direction d'approche (radar polaire)
#  - Log des messages CAN
#  - Commandes de simulation
# =============================================================

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
import datetime

from config import SAMPLE_RATE, N_SAMPLES, REPORTS_DIR
from inference import AEDASInference, DetectionResult
from tdoa import simulate_mic_signals, estimate_direction, TDOAResult
from can_bus import VirtualCANBus, encode_siren_detection, encode_tdoa_direction

# ── Configuration Streamlit ───────────────────────────────────

st.set_page_config(
    page_title="A.E.D.A.S. — Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé ──────────────────────────────────────────

st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .metric-card {
        background: linear-gradient(135deg, #1A3557 0%, #2563A8 100%);
        padding: 20px; border-radius: 12px;
        text-align: center; margin: 5px;
    }
    .metric-value { font-size: 2.5em; font-weight: bold; color: white; }
    .metric-label { font-size: 0.9em; color: #AAC4E0; margin-top: 5px; }
    .alert-active {
        background: linear-gradient(135deg, #8B0000 0%, #E84040 100%);
        padding: 15px; border-radius: 10px;
        text-align: center; animation: pulse 1s infinite;
    }
    .alert-clear {
        background: linear-gradient(135deg, #1A4A2A 0%, #2E8B57 100%);
        padding: 15px; border-radius: 10px; text-align: center;
    }
    .alert-text { font-size: 1.8em; font-weight: bold; color: white; }
    .can-log {
        font-family: 'Courier New', monospace;
        font-size: 0.75em; color: #00FF41;
        background: #0D0D0D; padding: 10px;
        border-radius: 6px; border: 1px solid #1A7A1A;
    }
    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(232,64,64,0.7); }
        70%  { box-shadow: 0 0 0 10px rgba(232,64,64,0); }
        100% { box-shadow: 0 0 0 0 rgba(232,64,64,0); }
    }
</style>
""", unsafe_allow_html=True)

# ── Initialisation session state ──────────────────────────────

def init_session_state():
    defaults = {
        "engine":          None,
        "can_bus":         None,
        "history_prob":    [],
        "history_time":    [],
        "history_angle":   [],
        "can_log":         [],
        "is_siren":        False,
        "current_prob":    0.0,
        "current_angle":   0.0,
        "current_quadrant":"FRONT",
        "volume":          100,
        "frame_count":     0,
        "detections":      0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# ── Chargement des modèles ────────────────────────────────────

@st.cache_resource
def load_engine():
    return AEDASInference(threshold=0.82)

@st.cache_resource
def load_can_bus():
    return VirtualCANBus(bitrate=500_000)

# ── Sidebar ───────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.shields.io/badge/A.E.D.A.S.-v1.0-blue?style=for-the-badge")
    st.title("⚙️ Paramètres de simulation")

    st.subheader("🎯 Source sonore")
    sim_mode = st.radio(
        "Mode",
        ["Sirène simulée", "Bruit aléatoire", "Silence"],
        index=0,
    )

    source_angle = st.slider(
        "Angle d'approche (°)",
        min_value=0, max_value=359, value=45, step=5,
        help="0° = avant du véhicule, 90° = gauche"
    )

    source_distance = st.slider(
        "Distance estimée (m)",
        min_value=10, max_value=200, value=50, step=10,
    )

    snr_db = st.slider(
        "SNR (dB) — qualité du signal",
        min_value=0, max_value=30, value=15, step=5,
        help="Plus élevé = signal plus propre"
    )

    st.subheader("🎛️ Seuil de détection")
    threshold = st.slider(
        "Seuil P(siren)",
        min_value=0.50, max_value=0.95, value=0.82, step=0.01,
    )

    st.subheader("⏱️ Simulation")
    sim_speed = st.selectbox(
        "Vitesse", ["Temps réel (1s)", "Rapide (0.3s)", "Manuel"]
    )

    auto_run = st.toggle("▶ Simulation automatique", value=False)

    st.divider()
    if st.button("🔄 Réinitialiser", type="secondary", use_container_width=True):
        for k in ["history_prob", "history_time", "history_angle",
                  "can_log", "frame_count", "detections"]:
            st.session_state[k] = [] if isinstance(st.session_state[k], list) else 0
        st.session_state["is_siren"]  = False
        st.session_state["volume"]    = 100
        st.rerun()

# ── Header ────────────────────────────────────────────────────

st.markdown("""
<div style='text-align:center; padding:10px 0 20px 0'>
    <h1 style='color:#2563A8; font-size:2.2em; margin:0'>
        🚨 A.E.D.A.S.
    </h1>
    <p style='color:#AAC4E0; font-size:1em; margin:0'>
        Acoustic Emergency Detection & Alert System — Dashboard HMI
    </p>
</div>
""", unsafe_allow_html=True)

# ── Statut alerte ─────────────────────────────────────────────

alert_col, info_col = st.columns([2, 3])

with alert_col:
    if st.session_state["is_siren"]:
        st.markdown(f"""
        <div class='alert-active'>
            <div class='alert-text'>🚨 SIRÈNE DÉTECTÉE</div>
            <div style='color:#FFD0D0; margin-top:8px'>
                Direction : {st.session_state['current_quadrant']} —
                {st.session_state['current_angle']:.1f}°
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='alert-clear'>
            <div class='alert-text'>✅ Aucune sirène</div>
            <div style='color:#C0FFD0; margin-top:8px'>Système actif — monitoring en cours</div>
        </div>
        """, unsafe_allow_html=True)

with info_col:
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("P(siren)", f"{st.session_state['current_prob']:.3f}",
                  delta=None)
    with m2:
        st.metric("Volume audio", f"{st.session_state['volume']}%",
                  delta=f"{st.session_state['volume']-100}%"
                  if st.session_state['volume'] < 100 else None,
                  delta_color="inverse")
    with m3:
        st.metric("Frames analysées", st.session_state["frame_count"])
    with m4:
        st.metric("Détections", st.session_state["detections"])

st.divider()

# ── Graphiques principaux ─────────────────────────────────────

col_prob, col_radar = st.columns([3, 2])

with col_prob:
    st.subheader("📈 Probabilité P(siren) en temps réel")

    if st.session_state["history_prob"]:
        fig_prob = go.Figure()

        # Zone de seuil
        fig_prob.add_hrect(
            y0=threshold, y1=1.0,
            fillcolor="rgba(232,64,64,0.15)",
            line_width=0,
        )
        fig_prob.add_hline(
            y=threshold, line_dash="dash",
            line_color="#E84040", line_width=1.5,
            annotation_text=f"Seuil = {threshold:.2f}",
            annotation_position="right",
        )

        # Courbe de probabilité
        colors = ["#E84040" if p >= threshold else "#2563A8"
                  for p in st.session_state["history_prob"]]

        fig_prob.add_trace(go.Scatter(
            x=list(range(len(st.session_state["history_prob"]))),
            y=st.session_state["history_prob"],
            mode="lines+markers",
            line=dict(color="#2563A8", width=2),
            marker=dict(color=colors, size=6),
            fill="tozeroy",
            fillcolor="rgba(37,99,168,0.1)",
            name="P(siren)",
        ))

        fig_prob.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=20, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(14,17,23,0.8)",
            xaxis=dict(title="Frame", color="#AAC4E0",
                       gridcolor="#1E2A3A"),
            yaxis=dict(title="Probabilité", range=[0, 1.05],
                       color="#AAC4E0", gridcolor="#1E2A3A"),
            showlegend=False,
        )
        st.plotly_chart(fig_prob, use_container_width=True)
    else:
        st.info("Lance la simulation pour voir les données en temps réel.")

with col_radar:
    st.subheader("🧭 Direction d'approche")

    angle_rad = np.radians(st.session_state["current_angle"])

    fig_polar = go.Figure()

    # Cercles de référence
    for r_val, label in [(0.33, "PROCHE"), (0.66, "MOYEN"), (1.0, "LOIN")]:
        theta_circle = np.linspace(0, 2*np.pi, 100)
        fig_polar.add_trace(go.Scatterpolar(
            r=[r_val]*100,
            theta=np.degrees(theta_circle),
            mode="lines",
            line=dict(color="#1E2A3A", width=1),
            showlegend=False,
        ))

    # Indicateur de direction
    if st.session_state["is_siren"]:
        fig_polar.add_trace(go.Scatterpolar(
            r=[0, 0.9],
            theta=[0, st.session_state["current_angle"]],
            mode="lines+markers",
            line=dict(color="#E84040", width=3),
            marker=dict(size=[0, 14], color="#E84040",
                        symbol=["circle", "arrow"]),
            name="Source",
            showlegend=False,
        ))

    # Position du véhicule
    fig_polar.add_trace(go.Scatterpolar(
        r=[0], theta=[0],
        mode="markers",
        marker=dict(size=16, color="#2563A8", symbol="square"),
        name="Véhicule",
        showlegend=False,
    ))

    fig_polar.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(
                tickmode="array",
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=["AVANT", "AV-G", "GAUCHE", "AR-G",
                          "ARRIÈRE", "AR-D", "DROITE", "AV-D"],
                direction="clockwise",
                rotation=90,
                color="#AAC4E0",
            ),
            bgcolor="rgba(14,17,23,0.9)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=40, r=40, t=20, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig_polar, use_container_width=True)

# ── Log CAN ───────────────────────────────────────────────────

st.subheader("📡 Log Bus CAN Virtuel")
if st.session_state["can_log"]:
    log_text = "\n".join(st.session_state["can_log"][-12:])
    st.markdown(f"<div class='can-log'>{log_text}</div>",
                unsafe_allow_html=True)
else:
    st.markdown("<div class='can-log'>En attente de messages CAN...</div>",
                unsafe_allow_html=True)

# ── Bouton manuel / simulation ────────────────────────────────

st.divider()
run_col, step_col = st.columns([1, 3])

with run_col:
    manual_trigger = st.button(
        "▶ Analyser une frame",
        type="primary",
        use_container_width=True,
        disabled=auto_run,
    )

with step_col:
    if st.session_state["history_prob"]:
        hist_df = pd.DataFrame({
            "Frame": range(len(st.session_state["history_prob"])),
            "P(siren)": st.session_state["history_prob"],
            "Siren": ["🚨" if p >= threshold else "✅"
                      for p in st.session_state["history_prob"]],
        })
        st.dataframe(hist_df.tail(5), hide_index=True,
                     use_container_width=True, height=120)


# ── Logique de simulation ─────────────────────────────────────

def run_one_frame():
    """Simule une frame d'analyse audio complète."""
    try:
        engine  = load_engine()
        can_bus = load_can_bus()
    except Exception as e:
        st.error(f"Erreur chargement modèle : {e}")
        return

    np.random.seed(int(time.time() * 1000) % 2**32)

    # Génération du signal audio selon le mode
    if sim_mode == "Sirène simulée":
        mic_signals = simulate_mic_signals(
            source_angle_deg=source_angle,
            source_distance_m=source_distance,
            duration_s=4.0,
            snr_db=snr_db,
        )
        audio = mic_signals["FL"]

    elif sim_mode == "Bruit aléatoire":
        audio = np.random.randn(N_SAMPLES).astype(np.float32) * 0.1

    else:  # Silence
        audio = np.zeros(N_SAMPLES, dtype=np.float32)

    # Inférence
    engine.threshold = threshold
    result = engine.predict(audio)

    # TDOA si sirène détectée
    tdoa = None
    if result.is_siren and sim_mode == "Sirène simulée":
        mic_signals = simulate_mic_signals(
            source_angle_deg=source_angle,
            source_distance_m=source_distance,
            duration_s=1.0, snr_db=snr_db,
        )
        tdoa = estimate_direction(mic_signals)

    # Mise à jour session state
    st.session_state["current_prob"]    = result.probability
    st.session_state["is_siren"]        = result.is_siren
    st.session_state["frame_count"]    += 1

    if result.is_siren:
        st.session_state["detections"]  += 1
        st.session_state["volume"]       = 30
        if tdoa:
            st.session_state["current_angle"]   = tdoa.angle_deg
            st.session_state["current_quadrant"] = tdoa.quadrant
    else:
        st.session_state["volume"]       = 100
        st.session_state["current_angle"] = 0.0

    # Historique (max 50 points)
    st.session_state["history_prob"].append(result.probability)
    if len(st.session_state["history_prob"]) > 50:
        st.session_state["history_prob"].pop(0)

    # Log CAN
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    status = "SIREN" if result.is_siren else "CLEAR"
    log_entry = (f"[{ts}] ID=0x1A0 {status} "
                 f"P={result.probability:.4f} "
                 f"{'| DIR=' + tdoa.quadrant if tdoa else ''}")
    st.session_state["can_log"].append(log_entry)
    if len(st.session_state["can_log"]) > 50:
        st.session_state["can_log"].pop(0)


# ── Déclenchement ─────────────────────────────────────────────

if manual_trigger:
    run_one_frame()
    st.rerun()

if auto_run:
    delay = {"Temps réel (1s)": 1.0, "Rapide (0.3s)": 0.3}.get(sim_speed, 0.5)
    run_one_frame()
    time.sleep(delay)
    st.rerun()