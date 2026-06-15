import os
import sys
import time
import random
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ── local modules ────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from dynamic_router import dynamic_router

# ── Chart export & Formatting Layout ───────────────────────────
CHARTS_DIR = os.path.join(BASE_DIR, "exported_charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# تعديل: تركيز التكبير والتغميق بشكل صارم على محاور الفواصل والتراتيب (xaxis & yaxis) داخل الرسومات
EXPORT_LAYOUT = dict(
    font=dict(family="Arial Black, Arial, sans-serif", size=26, color="#000000"),
    title_font=dict(size=30, color="#000000", family="Arial Black, Arial, sans-serif"),
    xaxis=dict(
        color="#000000",
        tickfont=dict(color="#000000", size=26, family="Arial Black, sans-serif"), # تكبير وتغميق أرقام/نصوص محور الفواصل
        title_font=dict(color="#000000", size=30, family="Arial Black, sans-serif"), # تكبير وتغميق عنوان محور الفواصل
        linecolor="#000000", linewidth=4, showline=True,
        mirror=True, ticks="outside", tickwidth=4, tickcolor="#000000"
    ),
    yaxis=dict(
        color="#000000",
        tickfont=dict(color="#000000", size=26, family="Arial Black, sans-serif"), # تكبير وتغميق أرقام/نصوص محور التراتيب
        title_font=dict(color="#000000", size=30, family="Arial Black, sans-serif"), # تكبير وتغميق عنوان محور التراتيب
        linecolor="#000000", linewidth=4, showline=True,
        mirror=True, ticks="outside", tickwidth=4, tickcolor="#000000"
    ),
    legend=dict(
        font=dict(family="Arial Black, Arial, sans-serif", size=24, color="#000000"),
        bordercolor="#000000", borderwidth=2
    ),
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=140, r=80, t=100, b=140), # هوامش كافية لمنع تداخل نصوص وعناوين المحاور الكبيرة جداً
)

def apply_strict_black_theme(fig):
    fig.update_layout(**EXPORT_LAYOUT)
    fig.update_annotations(font=dict(family="Arial Black, Arial, sans-serif", size=24, color="#000000"))
    
    fig.update_layout(
        uniformtext=dict(mode="hide", minsize=22),
        font=dict(color="#000000")
    )
    if hasattr(fig, 'data'):
        for trace in fig.data:
            if trace.type == 'pie':
                trace.update(
                    textfont=dict(family="Arial Black, Arial, sans-serif", size=24, color="#000000"),
                    textinfo="label+percent"
                )
    return fig

def save_all_charts(df: pd.DataFrame, drift_scores: list) -> int:
    saved = 0
    try:
        if df.empty:
            return 0

        dec_map = {"High": 2, "Medium": 1, "Low": 0}
        df = df.copy()
        df["dec_num"] = df["decision"].map(dec_map)

        figs = {}

        # 1. Timeline
        figs["01_irrigation_timeline"] = px.line(
            df, x="timestamp", y="dec_num",
            title="Irrigation Level Timeline (0=Low, 1=Medium, 2=High)",
            color_discrete_sequence=["#000000"]) 
        figs["01_irrigation_timeline"].update_yaxes(
            tickvals=[0,1,2], ticktext=["Low","Medium","High"])

        # 2. Distribution
        dec_counts = df["decision"].value_counts().reset_index()
        dec_counts.columns = ["decision", "count"]
        figs["02_irrigation_distribution"] = px.pie(
            dec_counts, names="decision", values="count",
            title="Irrigation Level Distribution", color="decision",
            color_discrete_map={"High":"#da3633","Medium":"#e3b341","Low":"#1a7f37"})

        # 3. Edge vs Cloud
        layer_counts = df["layer"].value_counts().reset_index()
        layer_counts.columns = ["layer", "count"]
        figs["03_edge_vs_cloud"] = px.bar(
            layer_counts, x="layer", y="count",
            title="Edge vs Cloud Decisions", color="layer",
            color_discrete_map={"Edge":"#1f6feb","Cloud":"#388bfd"})

        # 4. Latency
        figs["04_inference_latency"] = px.area(
            df, x="timestamp", y="latency_ms",
            title="Inference Latency (ms)",
            color_discrete_sequence=["#1a7f37"])

        # 5. Network latency
        if "network_latency" in df.columns:
            figs["05_network_latency"] = px.line(
                df, x="timestamp", y="network_latency",
                title="Network Latency — Dynamic Router (ms)",
                color_discrete_sequence=["#000000"])
            figs["05_network_latency"].add_hline(y=120, line_dash="dash", line_color="#000000", line_width=4,
                annotation_text="Cloud Threshold (120 ms)", annotation_font=dict(color="#000000", size=22, family="Arial Black"))

        # 6. Edge CPU
        if "edge_cpu" in df.columns:
            figs["06_edge_cpu"] = px.area(
                df, x="timestamp", y="edge_cpu",
                title="Edge CPU Usage (%)",
                color_discrete_sequence=["#ffa657"])
            figs["06_edge_cpu"].add_hline(y=80, line_dash="dash", line_color="#000000", line_width=4,
                annotation_text="Max Edge CPU (80%)", annotation_font=dict(color="#000000", size=22, family="Arial Black"))

        # 7. Adaptive Threshold
        if "adaptive_thresh" in df.columns:
            figs["07_adaptive_threshold"] = px.line(
                df, x="timestamp", y="adaptive_thresh",
                title="Adaptive Edge Threshold Over Time",
                color_discrete_sequence=["#1a7f37"])
            figs["07_adaptive_threshold"].add_hline(y=0.70, line_dash="dash", line_color="#000000", line_width=4,
                annotation_text="Original Fixed Threshold (0.70)", annotation_font=dict(color="#000000", size=22, family="Arial Black"))

        # 8. Buffer Size
        if "buffer_size" in df.columns:
            figs["08_buffer_size"] = px.area(
                df, x="timestamp", y="buffer_size",
                title="Sliding Buffer Size (Recent IoT Samples)",
                color_discrete_sequence=["#8250df"])
            figs["08_buffer_size"].add_hline(y=50, line_dash="dash", line_color="#000000", line_width=4,
                annotation_text="Min for Retrain (50)", annotation_font=dict(color="#000000", size=22, family="Arial Black"))

        # 9. Drift Indicator
        if len(drift_scores) >= 3:
            drift_df = pd.DataFrame({"reading": range(len(drift_scores)), "confidence": drift_scores})
            figs["09_drift_indicator"] = px.line(
                drift_df, x="reading", y="confidence",
                title="Sliding Confidence Score — Drift Indicator",
                color_discrete_sequence=["#da3633"])
            figs["09_drift_indicator"].add_hline(y=75, line_dash="dash", line_color="#000000", line_width=4,
                annotation_text="Drift Threshold (75%)", annotation_font=dict(color="#000000", size=22, family="Arial Black"))

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        for name, fig in figs.items():
            fig = apply_strict_black_theme(fig)
            path = os.path.join(CHARTS_DIR, f"{ts}_{name}.png")
            fig.write_image(path, format="png", width=1800, height=800, scale=2)
            saved += 1

    except Exception as e:
        print(f"[save_all_charts] Error: {e}")

    return saved

MQTT_AVAILABLE = False
mqtt_bridge    = None
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Irrigation AIoT",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS (تعديل لتكبير وتغميق نصوص الواجهة والبطاقات فقط دون تغيير نمط أو ألوان الواجهة الأصلية) ──
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { font-size: 20px; font-weight: 700; color: #000000; }
  [data-testid="stSidebar"] * { font-size: 19px !important; font-weight: 800 !important; color: #000000 !important; }
  [data-testid="stSidebar"] .stMarkdown p { font-size: 19px !important; font-weight: 800; }
  h1 { font-size: 2.6rem !important; font-weight: 900 !important; color: #000000 !important; }
  h2 { font-size: 2.1rem !important; font-weight: 900 !important; color: #000000 !important; }
  h3 { font-size: 1.7rem !important; font-weight: 800 !important; color: #000000 !important; }
  [data-testid="stMarkdownContainer"] h3 { font-size: 1.7rem !important; font-weight: 800 !important; color: #000000 !important; }
  [data-testid="stMetric"] label { font-size: 20px !important; font-weight: 800 !important; color: #000000 !important; }
  [data-testid="stMetricValue"] { font-size: 2.6rem !important; font-weight: 900 !important; color: #000000 !important; }
  [data-testid="stMetricDelta"] { font-size: 18px !important; font-weight: 800 !important; }
  .edge-badge   { padding:6px 16px; border-radius:20px; font-size:18px; font-weight:900; }
  .cloud-badge  { padding:6px 16px; border-radius:20px; font-size:18px; font-weight:900; }
  .high-badge   { padding:6px 16px; border-radius:20px; font-size:18px; font-weight:900; }
  .medium-badge { padding:6px 16px; border-radius:20px; font-size:18px; font-weight:900; }
  .low-badge    { padding:6px 16px; border-radius:20px; font-size:18px; font-weight:900; }
  .router-info  { padding:14px 20px; border-radius:8px; font-size:18px; font-weight:800; }
  .thresh-info  { padding:14px 20px; border-radius:8px; font-size:18px; font-weight:800; }
  .buffer-info  { padding:14px 20px; border-radius:8px; font-size:18px; font-weight:800; }
  .drift-warn   { padding:16px 22px; border-radius:10px; font-size:18px; font-weight:900; }
  .retrain-ok   { padding:16px 22px; border-radius:10px; font-size:18px; font-weight:900; }
  b { font-size: 19px; font-weight: 900; color: #000000; }
  code { font-size: 19px !important; font-weight: 800 !important; color: #000000 !important; padding: 4px 10px; border-radius: 4px; }
  [data-testid="stDataFrame"] * { font-size: 18px !important; font-weight: 700 !important; color: #000000 !important; }
  [data-testid="stTabs"] button { font-size: 21px !important; font-weight: 900 !important; color: #000000 !important; }
  .stButton > button { font-size: 19px !important; font-weight: 800 !important; padding: 12px 28px !important; color: #000000 !important; }
  [data-testid="stSlider"] label { font-size: 18px !important; font-weight: 800 !important; }
  [data-testid="stToggle"] label { font-size: 18px !important; font-weight: 800 !important; }
</style>
""", unsafe_allow_html=True)

# ── load artifacts ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    m   = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    enc = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    te  = joblib.load(os.path.join(MODEL_DIR, "target_encoder.pkl"))
    mt  = joblib.load(os.path.join(MODEL_DIR, "model_meta.pkl"))
    return m, enc, te, mt

model, encoders, target_encoder, meta = load_artifacts()
FEATURE_COLS = meta["features"]
CLASSES      = meta["classes"]

# ── Drift Retrainer ────────────────────────────────────────────
@st.cache_resource
def load_retrainer(_model, _te):
    try:
        from Drift_retrainer import DriftRetrainer
        return DriftRetrainer(
            model=_model, feature_cols=FEATURE_COLS, target_encoder=_te,
            buffer_size=500, window_size=20, threshold=0.75,
            min_label_conf=0.80, min_buffer_retrain=50, min_retrain_gap=30,
        )
    except Exception as e:
        st.warning(f"DriftRetrainer error: {e}")
        return None

retrainer = load_retrainer(model, target_encoder)

CATEGORICAL_POOLS = {
    "Soil_Type"         : ["Clay", "Loamy", "Sandy", "Silt"],
    "Crop_Type"         : ["Cotton", "Maize", "Potato", "Rice", "Sugarcane", "Wheat"],
    "Crop_Growth_Stage" : ["Flowering", "Harvest", "Sowing", "Vegetative"],
    "Season"            : ["Kharif", "Rabi", "Zaid"],
    "Water_Source"      : ["Groundwater", "Rainwater", "Reservoir", "River"],
    "Mulching_Used"     : ["Yes", "No"],
    "Region"            : ["Central", "East", "North", "South", "West"],
}

NUMERIC_RANGES = {
    "Soil_Moisture"          : (8.0,  65.0),
    "Temperature_C"          : (12.0, 42.0),
    "Humidity"               : (25.0, 95.0),
    "Rainfall_mm"            : (0.0,  150.0),
    "Sunlight_Hours"         : (4.0,  11.0),
    "Wind_Speed_kmh"         : (0.5,  20.0),
    "Organic_Carbon"         : (0.30, 1.60),
    "Electrical_Conductivity": (0.10, 3.50),
    "Soil_pH"                : (4.80, 8.20),
    "Field_Area_hectare"     : (0.30, 15.0),
    "Previous_Irrigation_mm" : (0.02, 120.0),
}

DECISION_COLORS = {"High": "#da3633", "Medium": "#e3b341", "Low": "#1a7f37"}
DECISION_ICONS  = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}

def simulate_iot(drift: bool = False) -> dict:
    r = {}
    for feat in FEATURE_COLS:
        if feat in CATEGORICAL_POOLS:
            r[feat] = random.choice(CATEGORICAL_POOLS[feat])
        elif feat in NUMERIC_RANGES:
            lo, hi = NUMERIC_RANGES[feat]
            v = random.uniform(lo, hi)
            if drift:
                v += random.gauss(0, (hi - lo) * 0.25)
                v  = max(lo * 0.5, min(hi * 1.5, v))
            r[feat] = round(v, 2)
        else:
            r[feat] = round(random.uniform(0, 1), 2)
    return r

def encode_reading(raw: dict) -> pd.DataFrame:
    enc = {}
    for feat in FEATURE_COLS:
        val = str(raw.get(feat, ""))
        if feat in encoders:
            le = encoders[feat]
            enc[feat] = int(le.transform([val])[0]) if val in le.classes_ else 0
        else:
            try:    enc[feat] = float(raw[feat])
            except: enc[feat] = 0.0
    return pd.DataFrame([enc])[FEATURE_COLS]

def edge_infer(raw: dict, min_conf_thresh: float = 0.70):
    m = raw.get("Soil_Moisture", 50)
    r = raw.get("Rainfall_mm", 500)
    t = raw.get("Temperature_C", 25)
    h = raw.get("Humidity", 60)
    w = raw.get("Wind_Speed_kmh", 10)
    p = raw.get("Previous_Irrigation_mm", 30)
    candidates = []
    if m <= 22 and r <= 30 and t >= 30: candidates.append(("High", 0.93))
    if m <= 30 and t >= 35 and h <= 40: candidates.append(("High", 0.89))
    if m <= 25 and w >= 15 and r <= 40: candidates.append(("High", 0.85))
    if m >= 50 or r >= 100: candidates.append(("Low", 0.91))
    if m >= 40 and h >= 80 and p >= 50: candidates.append(("Low", 0.87))
    if 22 < m < 40 and 10 < r < 80: candidates.append(("Medium", 0.78))
    if 30 <= m <= 45 and 20 <= t <= 30 and h >= 50: candidates.append(("Medium", 0.75))
    for label, conf in sorted(candidates, key=lambda x: -x[1]):
        if conf >= min_conf_thresh:
            return label, conf
    return None, 0.0

def cloud_infer(raw: dict):
    current_model = retrainer.get_model() if retrainer else model
    X     = encode_reading(raw)
    proba = current_model.predict_proba(X)[0]
    idx   = int(np.argmax(proba))
    conf  = float(proba[idx])
    label = str(target_encoder.inverse_transform([idx])[0])
    return label, conf

def hybrid_predict(raw: dict, edge_conf_thresh: float = 0.70) -> dict:
    t0 = time.perf_counter()
    ed, ec = edge_infer(raw, min_conf_thresh=edge_conf_thresh)
    route  = dynamic_router(ec if ed is not None else 0.0)
    if route["use_edge"] and ed is not None:
        layer, decision, confidence = "Edge", ed, ec
    else:
        layer, decision, confidence = "Cloud", *cloud_infer(raw)
    lat = round((time.perf_counter() - t0) * 1000, 2)
    if retrainer:
        retrainer.observe(confidence * 100)
        if confidence >= retrainer._min_label_conf:
            try:
                x_encoded       = encode_reading(raw).iloc[0]
                y_label_encoded = int(target_encoder.transform([decision])[0])
                retrainer.add_sample(x_encoded, y_label_encoded)
            except Exception:
                pass
    return {
        "timestamp"      : datetime.now().strftime("%H:%M:%S"),
        "layer"          : layer,
        "decision"       : decision,
        "confidence"     : round(confidence * 100, 1),
        "latency_ms"     : lat,
        "moisture"       : raw.get("Soil_Moisture", 0),
        "rainfall"       : raw.get("Rainfall_mm", 0),
        "temp"           : raw.get("Temperature_C", 0),
        "router_reason"  : route["reason"],
        "network_latency": route["network_latency"],
        "edge_cpu"       : round(route["edge_cpu"], 1),
        "adaptive_thresh": route.get("adaptive_thresh", 0.70),
        "buffer_size"    : retrainer.get_status()["buffer_size"] if retrainer else 0,
    }

# ── session state ──────────────────────────────────────────────
if "history"        not in st.session_state: st.session_state.history        = []
if "drift_scores"   not in st.session_state: st.session_state.drift_scores   = []
if "running"        not in st.session_state: st.session_state.running        = False
if "last_save_msg"  not in st.session_state: st.session_state.last_save_msg  = ""
if "simulate_drift" not in st.session_state: st.session_state.simulate_drift = False
if "stream_interval" not in st.session_state: st.session_state.stream_interval = 1.0
if "max_history"     not in st.session_state: st.session_state.max_history     = 60
if "edge_thresh"     not in st.session_state: st.session_state.edge_thresh     = 0.70

# ── sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.session_state.simulate_drift  = st.toggle("🔀 Simulate Concept Drift", st.session_state.simulate_drift)
    st.session_state.stream_interval = st.slider("⏱ Sensor Interval (sec)", 0.5, 5.0, st.session_state.stream_interval, 0.5)
    st.session_state.max_history     = st.slider("📋 History Size", 10, 200, st.session_state.max_history)
    st.session_state.edge_thresh     = st.slider("🔵 Edge Confidence Threshold", 0.5, 0.95, st.session_state.edge_thresh, 0.05)

    st.divider()
    st.markdown("<b>🟠 MQTT: Simulated (broker offline)</b>", unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Start", use_container_width=True):
            st.session_state.running = True
    with col2:
        if st.button("⏹ Stop", use_container_width=True):
            st.session_state.running = False
            if st.session_state.history:
                df_save = pd.DataFrame(st.session_state.history)
                n = save_all_charts(df_save, st.session_state.drift_scores)
                if n > 0:
                    st.session_state.last_save_msg = f"✅ Saved {n} high-contrast charts to exported_charts directory."
                else:
                    st.session_state.last_save_msg = "⚠️ Error saving — check kaleido package version."

    if st.session_state.running:
        st.success("🟢 Simulation Active")
    else:
        st.info("⏸ Simulation Stopped")
        if st.session_state.get("last_save_msg"):
            if "✅" in st.session_state.last_save_msg: st.success(st.session_state.last_save_msg)
            else: st.warning(st.session_state.last_save_msg)

    st.divider()
    model_name = meta.get("best_model", "Model")
    st.markdown(f"**Model:** `{meta.get('version', '—')}`")
    st.markdown(f"**Best:** `{model_name}`")
    st.markdown(f"**F1:** `{meta['f1_score']*100:.1f}%`")
    st.markdown(f"**Accuracy:** `{meta['accuracy']*100:.1f}%`")
    st.markdown(f"**Trained:** `{meta['trained_at'][:10]}`")

simulate_drift  = st.session_state.simulate_drift
stream_interval = st.session_state.stream_interval
max_history     = st.session_state.max_history
edge_thresh     = st.session_state.edge_thresh

if st.session_state.running:
    raw    = simulate_iot(drift=simulate_drift)
    result = hybrid_predict(raw, edge_conf_thresh=edge_thresh)
    st.session_state.history.append(result)
    st.session_state.drift_scores.append(result["confidence"])
    if len(st.session_state.history) > max_history:
        st.session_state.history      = st.session_state.history[-max_history:]
        st.session_state.drift_scores = st.session_state.drift_scores[-max_history:]

# ── tabs ───────────────────────────────────────────────────────
tab_main, tab_compare = st.tabs(["📡 Live Dashboard", "📊 Model Comparison"])

with tab_main:
    st.markdown("# 🌱 Smart Irrigation AIoT Dashboard")
    st.markdown("**Hybrid Edge–Cloud | Real-Time IoT Sensing | Adaptive Sliding-Window Retraining**")
    st.divider()

    hist    = st.session_state.history
    total   = len(hist)
    high_n  = sum(1 for h in hist if h["decision"] == "High")
    med_n   = sum(1 for h in hist if h["decision"] == "Medium")
    low_n   = sum(1 for h in hist if h["decision"] == "Low")
    edge_n  = sum(1 for h in hist if h["layer"]    == "Edge")
    cloud_n = total - edge_n
    avg_lat = round(np.mean([h["latency_ms"] for h in hist]), 2) if hist else 0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("📡 Total",           total)
    k2.metric("🔴 High",            high_n)
    k3.metric("🟡 Medium",          med_n)
    k4.metric("🟢 Low",             low_n)
    k5.metric("🔵 Edge / ☁️ Cloud", f"{edge_n} / {cloud_n}")
    k6.metric("⚡ Avg Latency",     f"{avg_lat} ms")

    st.divider()

    st.subheader("📡 Live Sensor Stream")
    if hist:
        latest = hist[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🌡 Temperature °C", latest["temp"])
        c2.metric("💧 Soil Moisture",  latest["moisture"])
        c3.metric("🌧 Rainfall mm",    latest["rainfall"])
        c4.metric("⚡ Latency ms",     latest["latency_ms"])

        dec         = latest["decision"]
        badge_class = f"{dec.lower()}-badge"
        layer_badge = "edge-badge" if latest["layer"] == "Edge" else "cloud-badge"
        st.markdown(
            f'<b>Decision:</b> <span class="{badge_class}">{DECISION_ICONS.get(dec,"")} {dec} Irrigation</span> &nbsp;|&nbsp; '
            f'<b>Layer:</b> <span class="{layer_badge}">{latest["layer"]}</span> &nbsp;|&nbsp; '
            f'<b>Confidence:</b> <code>{latest["confidence"]}%</code>',
            unsafe_allow_html=True,
        )

    df = pd.DataFrame(hist[-max_history:]) if hist else pd.DataFrame()

    if len(df) >= 3:
        st.subheader("📊 Real-Time Analytics")
        ch1, ch2 = st.columns(2)
        dec_map = {"High": 2, "Medium": 1, "Low": 0}
        df["dec_num"] = df["decision"].map(dec_map)

        # 1. Timeline Chart
        fig_time = px.line(df, x="timestamp", y="dec_num",
            title="Irrigation Level Timeline (0=Low, 1=Medium, 2=High)", color_discrete_sequence=["#000000"])
        fig_time.update_yaxes(tickvals=[0,1,2], ticktext=["Low","Medium","High"])
        fig_time = apply_strict_black_theme(fig_time)
        ch1.plotly_chart(fig_time, use_container_width=True)

        # 2. Pie Chart
        dec_counts = df["decision"].value_counts().reset_index()
        dec_counts.columns = ["decision", "count"]
        fig_pie = px.pie(dec_counts, names="decision", values="count",
            title="Irrigation Level Distribution", color="decision", color_discrete_map=DECISION_COLORS)
        fig_pie = apply_strict_black_theme(fig_pie)
        ch2.plotly_chart(fig_pie, use_container_width=True)

        ch3, ch4 = st.columns(2)
        # 3. Bar Chart Layer
        layer_counts = df["layer"].value_counts().reset_index()
        layer_counts.columns = ["layer", "count"]
        fig_layer = px.bar(layer_counts, x="layer", y="count", title="Edge vs Cloud Decisions",
            color="layer", color_discrete_map={"Edge": "#1f6feb", "Cloud": "#388bfd"})
        fig_layer = apply_strict_black_theme(fig_layer)
        ch3.plotly_chart(fig_layer, use_container_width=True)

        # 4. Latency Chart
        fig_lat = px.area(df, x="timestamp", y="latency_ms", title="Inference Latency (ms)", color_discrete_sequence=["#1a7f37"])
        fig_lat = apply_strict_black_theme(fig_lat)
        ch4.plotly_chart(fig_lat, use_container_width=True)

        if "network_latency" in df.columns:
            ch5, ch6 = st.columns(2)
            fig_net = px.line(df, x="timestamp", y="network_latency", title="Network Latency — Dynamic Router (ms)", color_discrete_sequence=["#000000"])
            fig_net.add_hline(y=120, line_dash="dash", line_color="#000000", line_width=4,
                annotation_text="Cloud Threshold (120 ms)", annotation_font=dict(color="#000000", size=22, family="Arial Black"))
            fig_net = apply_strict_black_theme(fig_net)
            ch5.plotly_chart(fig_net, use_container_width=True)

            fig_cpu = px.area(df, x="timestamp", y="edge_cpu", title="Edge CPU Usage (%)", color_discrete_sequence=["#ffa657"])
            fig_cpu.add_hline(y=80, line_dash="dash", line_color="#000000", line_width=4,
                annotation_text="Max Edge CPU (80%)", annotation_font=dict(color="#000000", size=22, family="Arial Black"))
            fig_cpu = apply_strict_black_theme(fig_cpu)
            ch6.plotly_chart(fig_cpu, use_container_width=True)

        if "adaptive_thresh" in df.columns:
            ch7, ch8 = st.columns(2)
            fig_thresh = px.line(df, x="timestamp", y="adaptive_thresh", title="Adaptive Edge Threshold Over Time", color_discrete_sequence=["#1a7f37"])
            fig_thresh.add_hline(y=0.70, line_dash="dash", line_color="#000000", line_width=4,
                annotation_text="Original Fixed Threshold (0.70)", annotation_font=dict(color="#000000", size=22, family="Arial Black"))
            fig_thresh.update_yaxes(range=[0.40, 1.0])
            fig_thresh = apply_strict_black_theme(fig_thresh)
            ch7.plotly_chart(fig_thresh, use_container_width=True)

            if "buffer_size" in df.columns:
                fig_buf = px.area(df, x="timestamp", y="buffer_size", title="Sliding Buffer Size (Recent IoT Samples)", color_discrete_sequence=["#8250df"])
                fig_buf.add_hline(y=50, line_dash="dash", line_color="#000000", line_width=4,
                    annotation_text="Min for Retrain (50)", annotation_font=dict(color="#000000", size=22, family="Arial Black"))
                fig_buf = apply_strict_black_theme(fig_buf)
                ch8.plotly_chart(fig_buf, use_container_width=True)

    st.subheader("🔄 Concept Drift Monitor")
    ds = st.session_state.drift_scores
    if len(ds) >= 3:
        drift_df  = pd.DataFrame({"reading": range(len(ds)), "confidence": ds})
        fig_drift = px.line(drift_df, x="reading", y="confidence", title="Sliding Confidence Score — Drift Indicator", color_discrete_sequence=["#da3633"])
        fig_drift.add_hline(y=75, line_dash="dash", line_color="#000000", line_width=4,
            annotation_text="Drift Threshold (75%)", annotation_font=dict(color="#000000", size=22, family="Arial Black"))
        fig_drift = apply_strict_black_theme(fig_drift)
        st.plotly_chart(fig_drift, use_container_width=True)

    st.subheader("📈 Feature Importance")
    current_model = retrainer.get_model() if retrainer else model
    model_label   = meta.get("best_model", "Model")
    fi_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": current_model.feature_importances_}).sort_values("Importance", ascending=True)

    fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h", title=f"{model_label} — Feature Importances", color="Importance",
        color_continuous_scale=[[0.0,"#a8c8f0"],[1.0,"#0a2d6b"]])
    fig_fi = apply_strict_black_theme(fig_fi)
    st.plotly_chart(fig_fi, use_container_width=True)

    if st.session_state.running:
        time.sleep(stream_interval)
        st.rerun()

with tab_compare:
    st.markdown("# 📊 Model Comparison")
    st.divider()

    csv_path = os.path.join(MODEL_DIR, "model_comparison_table.csv")
    cm_path  = os.path.join(MODEL_DIR, "confusion_matrix.png")

    fig_comp = fig_all = fig_lat2 = None
    available = []

    if os.path.exists(csv_path):
        comp_df  = pd.read_csv(csv_path)
        best_idx = comp_df["F1"].idxmax()

        fig_comp = px.bar(comp_df, x="Model", y="F1", title="Model Comparison — F1 Score (Weighted)",
            color="F1", color_continuous_scale="Blues", text=comp_df["F1"].apply(lambda x: f"{x:.3f}"))
        fig_comp = apply_strict_black_theme(fig_comp)
        st.plotly_chart(fig_comp, use_container_width=True)

        if "High_F1" in comp_df.columns:
            fig_high = px.bar(comp_df, x="Model", y="High_F1", title="F1 Score — 'High' Irrigation Class (Critical)",
                color="High_F1", color_continuous_scale="Reds", text=comp_df["High_F1"].apply(lambda x: f"{x:.3f}"))
            fig_high = apply_strict_black_theme(fig_high)
            st.plotly_chart(fig_high, use_container_width=True)

        metrics   = ["Accuracy", "Precision", "Recall", "F1"]
        available = [m for m in metrics if m in comp_df.columns]
        if available:
            melted = comp_df.melt(id_vars="Model", value_vars=available, var_name="Metric", value_name="Score")
            fig_all = px.bar(melted, x="Model", y="Score", color="Metric", barmode="group", title="All Metrics — Model Comparison",
                color_discrete_sequence=["#58a6ff","#3fb950","#f78166","#d2a8ff"])
            fig_all = apply_strict_black_theme(fig_all)
            st.plotly_chart(fig_all, use_container_width=True)

        if "Latency_ms" in comp_df.columns:
            fig_lat2 = px.bar(comp_df, x="Model", y="Latency_ms", title="Training Latency (ms)", color="Latency_ms", color_continuous_scale="Reds")
            fig_lat2 = apply_strict_black_theme(fig_lat2)
            st.plotly_chart(fig_lat2, use_container_width=True)

        st.subheader("📋 Full Results Table")
        styled = comp_df.style.highlight_max(subset=available, color="#1a3a1a").format({m: "{:.4f}" for m in available})
        st.dataframe(styled, use_container_width=True)
    else:
        st.info("Run `model_comparison.py` first to generate comparison results.")

    if os.path.exists(cm_path):
        st.subheader("🔲 Confusion Matrix — Best Model")
        st.image(cm_path, width=500)

    st.divider()
    st.subheader("💾 Export Charts")

    def fig_to_html(fig):
        return fig.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")

    col_dl1, col_dl2, col_dl3 = st.columns(3)
    with col_dl1:
        if fig_comp is not None:
            st.download_button("📥 F1 Score Chart (.html)", fig_to_html(fig_comp), "model_comparison_f1.html", "text/html", use_container_width=True)
    with col_dl2:
        if fig_all is not None and available:
            st.download_button("📥 All Metrics Chart (.html)", fig_to_html(fig_all), "model_comparison_all_metrics.html", "text/html", use_container_width=True)
    with col_dl3:
        if fig_lat2 is not None:
            st.download_button("📥 Latency Chart (.html)", fig_to_html(fig_lat2), "model_comparison_latency.html", "text/html", use_container_width=True)
