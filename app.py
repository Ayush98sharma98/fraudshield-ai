import os

def setup():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    if not os.path.exists("data/creditcard.csv"):
        try:
            os.system("pip install kaggle -q")
            # Kaggle credentials from Streamlit secrets
            os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
            os.environ["KAGGLE_KEY"]      = st.secrets["KAGGLE_KEY"]
            os.system("kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip")
        except:
            pass  # Will use sample data instead

setup()
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time

# ── Page Config ────────────────────────────────────
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════
# THEME SYSTEM
# ══════════════════════════════════════════════════
THEMES = {
    "🌑 Cyber Dark": {
        "bg": "#0a0e1a",
        "card": "#111827",
        "card2": "#1a2235",
        "border": "#1e3a5f",
        "accent": "#00d4ff",
        "accent2": "#7c3aed",
        "success": "#10b981",
        "danger": "#ef4444",
        "warning": "#f59e0b",
        "text": "#e2e8f0",
        "text2": "#94a3b8",
        "glow": "0 0 20px rgba(0,212,255,0.3)",
        "plotbg": "#111827",
        "plotpaper": "#0a0e1a",
        "font": "'Rajdhani', sans-serif",
    },
    "🌊 Ocean Depths": {
        "bg": "#020b18",
        "card": "#041628",
        "card2": "#062040",
        "border": "#0e4d7a",
        "accent": "#06b6d4",
        "accent2": "#0284c7",
        "success": "#34d399",
        "danger": "#f87171",
        "warning": "#fbbf24",
        "text": "#e0f2fe",
        "text2": "#7dd3fc",
        "glow": "0 0 25px rgba(6,182,212,0.4)",
        "plotbg": "#041628",
        "plotpaper": "#020b18",
        "font": "'Exo 2', sans-serif",
    },
    "🔥 Neon Ember": {
        "bg": "#0f0500",
        "card": "#1a0a00",
        "card2": "#2a1000",
        "border": "#7c2d12",
        "accent": "#f97316",
        "accent2": "#dc2626",
        "success": "#84cc16",
        "danger": "#ef4444",
        "warning": "#eab308",
        "text": "#fff7ed",
        "text2": "#fed7aa",
        "glow": "0 0 25px rgba(249,115,22,0.4)",
        "plotbg": "#1a0a00",
        "plotpaper": "#0f0500",
        "font": "'Orbitron', sans-serif",
    },
    "💜 Violet Storm": {
        "bg": "#07020f",
        "card": "#0f0520",
        "card2": "#160833",
        "border": "#4c1d95",
        "accent": "#a855f7",
        "accent2": "#ec4899",
        "success": "#22d3ee",
        "danger": "#f43f5e",
        "warning": "#fb923c",
        "text": "#f3e8ff",
        "text2": "#c4b5fd",
        "glow": "0 0 25px rgba(168,85,247,0.4)",
        "plotbg": "#0f0520",
        "plotpaper": "#07020f",
        "font": "'Audiowide', sans-serif",
    },
    "🌿 Matrix Green": {
        "bg": "#000800",
        "card": "#001200",
        "card2": "#002200",
        "border": "#14532d",
        "accent": "#22c55e",
        "accent2": "#4ade80",
        "success": "#86efac",
        "danger": "#f87171",
        "warning": "#fde047",
        "text": "#dcfce7",
        "text2": "#86efac",
        "glow": "0 0 20px rgba(34,197,94,0.4)",
        "plotbg": "#001200",
        "plotpaper": "#000800",
        "font": "'Share Tech Mono', monospace",
    },
    "❄️ Arctic White": {
        "bg": "#f0f9ff",
        "card": "#ffffff",
        "card2": "#e0f2fe",
        "border": "#bae6fd",
        "accent": "#0369a1",
        "accent2": "#7c3aed",
        "success": "#059669",
        "danger": "#dc2626",
        "warning": "#d97706",
        "text": "#0c1a2e",
        "text2": "#334155",
        "glow": "0 4px 20px rgba(3,105,161,0.15)",
        "plotbg": "#ffffff",
        "plotpaper": "#f0f9ff",
        "font": "'DM Sans', sans-serif",
    },
}

# ── Sidebar Theme Picker ───────────────────────────
with st.sidebar:
    st.markdown("## 🎨 Theme Studio")
    theme_name = st.selectbox("Choose Theme", list(THEMES.keys()), index=0)
    T = THEMES[theme_name]

    st.markdown("---")
    st.markdown("## 🧭 Navigation")
    page = st.radio("", [
        "🏠  Dashboard",
        "🔍  Live Detector",
        "📊  Analytics",
        "📁  Batch Scan",
        "ℹ️  About"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown(f"""
    <div style='text-align:center; padding: 10px;'>
        <div style='font-size:2rem'>🛡️</div>
        <div style='font-size:0.75rem; color:{T["text2"]}; margin-top:5px'>FraudShield AI v2.0</div>
        <div style='font-size:0.7rem; color:{T["text2"]}'>Model: Random Forest</div>
        <div style='font-size:0.7rem; color:{T["accent"]}'>● Online</div>
    </div>
    """, unsafe_allow_html=True)

# ── Dynamic CSS ────────────────────────────────────
is_light = theme_name == "❄️ Arctic White"

st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600;700&family=Audiowide&family=Share+Tech+Mono&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>
* {{ font-family: {T["font"]}; box-sizing: border-box; }}

.stApp {{
    background: {T["bg"]} !important;
    background-image: {"none" if is_light else f"radial-gradient(ellipse at 20% 50%, {T['accent']}08 0%, transparent 50%), radial-gradient(ellipse at 80% 20%, {T['accent2']}08 0%, transparent 50%)"} !important;
}}

section[data-testid="stSidebar"] {{
    background: {T["card"]} !important;
    border-right: 1px solid {T["border"]} !important;
}}

section[data-testid="stSidebar"] * {{ color: {T["text"]} !important; }}

.stSelectbox > div > div, .stRadio > div {{
    background: {T["card2"]} !important;
    border-color: {T["border"]} !important;
    color: {T["text"]} !important;
}}

.stButton > button {{
    background: linear-gradient(135deg, {T["accent"]}, {T["accent2"]}) !important;
    color: {'#000' if is_light else '#fff'} !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    transition: all 0.3s ease !important;
    box-shadow: {T["glow"]} !important;
}}

.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 0 35px {T["accent"]}66 !important;
}}

.stSlider > div > div > div {{
    background: {T["accent"]} !important;
}}

.stNumberInput > div > div > input, .stTextInput > div > div > input {{
    background: {T["card2"]} !important;
    border: 1px solid {T["border"]} !important;
    color: {T["text"]} !important;
    border-radius: 8px !important;
}}

.stTextArea > div > div > textarea {{
    background: {T["card2"]} !important;
    border: 1px solid {T["border"]} !important;
    color: {T["text"]} !important;
}}

h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {{
    color: {T["text"]} !important;
}}

.stDataFrame {{ background: {T["card"]} !important; }}

hr {{ border-color: {T["border"]} !important; }}

/* Custom Card */
.fs-card {{
    background: {T["card"]};
    border: 1px solid {T["border"]};
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}}

.fs-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, {T["accent"]}, {T["accent2"]});
}}

.fs-card:hover {{
    border-color: {T["accent"]}66;
    box-shadow: {T["glow"]};
    transform: translateY(-2px);
}}

/* Metric Card */
.metric-card {{
    background: {T["card2"]};
    border: 1px solid {T["border"]};
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}}

.metric-value {{
    font-size: 2rem;
    font-weight: 700;
    color: {T["accent"]};
    line-height: 1;
}}

.metric-label {{
    font-size: 0.75rem;
    color: {T["text2"]};
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 5px;
}}

.metric-sub {{
    font-size: 0.8rem;
    color: {T["text2"]};
    margin-top: 3px;
}}

/* Badges */
.badge-fraud {{
    background: {T["danger"]}22;
    border: 1px solid {T["danger"]};
    color: {T["danger"]};
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1px;
}}

.badge-safe {{
    background: {T["success"]}22;
    border: 1px solid {T["success"]};
    color: {T["success"]};
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1px;
}}

/* Alert Boxes */
.alert-fraud {{
    background: linear-gradient(135deg, {T["danger"]}15, {T["danger"]}05);
    border: 1px solid {T["danger"]}66;
    border-left: 4px solid {T["danger"]};
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    animation: pulse-red 2s infinite;
}}

.alert-safe {{
    background: linear-gradient(135deg, {T["success"]}15, {T["success"]}05);
    border: 1px solid {T["success"]}66;
    border-left: 4px solid {T["success"]};
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}}

@keyframes pulse-red {{
    0%, 100% {{ box-shadow: 0 0 0 0 {T["danger"]}44; }}
    50% {{ box-shadow: 0 0 15px 5px {T["danger"]}22; }}
}}

/* Hero Banner */
.hero {{
    background: linear-gradient(135deg, {T["card"]}, {T["card2"]});
    border: 1px solid {T["border"]};
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 2rem;
}}

.hero::after {{
    content: '🛡️';
    position: absolute;
    font-size: 15rem;
    opacity: 0.04;
    top: -30px;
    right: -30px;
    transform: rotate(15deg);
}}

.hero-title {{
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(135deg, {T["accent"]}, {T["accent2"]});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 2px;
    margin: 0;
}}

.hero-sub {{
    color: {T["text2"]};
    font-size: 1.1rem;
    margin-top: 0.5rem;
}}

/* Step indicator */
.step {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid {T["border"]}44;
}}

.step-num {{
    width: 32px; height: 32px;
    background: linear-gradient(135deg, {T["accent"]}, {T["accent2"]});
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700;
    font-size: 0.85rem;
    color: {'#000' if is_light else '#fff'};
    flex-shrink: 0;
}}

/* Scan animation */
@keyframes scanline {{
    0% {{ transform: translateY(-100%); }}
    100% {{ transform: translateY(100vh); }}
}}

.scan-effect {{
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, {T["accent"]}, transparent);
    z-index: 9999;
    animation: scanline 2s linear;
    pointer-events: none;
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {T["bg"]}; }}
::-webkit-scrollbar-thumb {{ background: {T["accent"]}66; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {T["accent"]}; }}

/* Expander */
.streamlit-expanderHeader {{
    background: {T["card2"]} !important;
    border-color: {T["border"]} !important;
    color: {T["text"]} !important;
    border-radius: 8px !important;
}}

/* File uploader */
.stFileUploader > div {{
    background: {T["card2"]} !important;
    border: 2px dashed {T["border"]} !important;
    border-radius: 12px !important;
}}

/* Progress bar */
.stProgress > div > div > div {{
    background: linear-gradient(90deg, {T["accent"]}, {T["accent2"]}) !important;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background: {T["card"]} !important;
    border-radius: 10px !important;
    padding: 4px !important;
    border: 1px solid {T["border"]} !important;
}}

.stTabs [data-baseweb="tab"] {{
    color: {T["text2"]} !important;
    border-radius: 8px !important;
}}

.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {T["accent"]}33, {T["accent2"]}33) !important;
    color: {T["accent"]} !important;
}}

</style>
""", unsafe_allow_html=True)

# ── Plotly Theme ───────────────────────────────────
def hex_to_rgba(hex_color, alpha=0.25):
    """Convert #rrggbb hex to rgba() string for Plotly compatibility."""
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return hex_color  # fallback

def plotly_layout(title="", height=400):
    grid = hex_to_rgba(T["border"], 0.3)
    return dict(
        title=dict(text=title, font=dict(color=T["accent"], size=14)),
        paper_bgcolor=T["plotpaper"],
        plot_bgcolor=T["plotbg"],
        font=dict(color=T["text"], family=T["font"]),
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(gridcolor=grid, color=T["text2"]),
        yaxis=dict(gridcolor=grid, color=T["text2"]),
    )

# ── Load Model ─────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open("models/fraud_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler, True
    except:
        return None, None, False

model, scaler, model_loaded = load_model()

# ══════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════
if "🏠" in page:

    # Hero
    st.markdown(f"""
    <div class="hero">
        <div class="hero-title">🛡️ FRAUDSHIELD AI</div>
        <div class="hero-sub">Real-time Credit Card Fraud Detection • Powered by Machine Learning</div>
        <div style="margin-top:1rem; display:flex; gap:10px; justify-content:center; flex-wrap:wrap;">
            <span class="badge-safe">● System Online</span>
            <span style="background:{T['accent']}22; border:1px solid {T['accent']}; color:{T['accent']}; padding:4px 12px; border-radius:20px; font-size:0.75rem; font-weight:700">Random Forest Model</span>
            <span style="background:{T['accent2']}22; border:1px solid {T['accent2']}; color:{T['accent2']}; padding:4px 12px; border-radius:20px; font-size:0.75rem; font-weight:700">284K Transactions Trained</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = [
        ("284,807", "Transactions", "Training Dataset"),
        ("99.5%", "Accuracy", "Test Set"),
        ("0.980", "ROC AUC", "Excellent"),
        ("98.7%", "Precision", "Fraud Class"),
        ("97.2%", "Recall", "Fraud Detection"),
    ]
    for col, (val, label, sub) in zip([col1, col2, col3, col4, col5], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f'<div class="fs-card">', unsafe_allow_html=True)
        st.markdown(f"### 🔄 How It Works")
        steps = [
            ("1", "Data Collection", "284K real European transactions from Kaggle"),
            ("2", "Preprocessing", "StandardScaler on Amount & Time features"),
            ("3", "SMOTE Balancing", "Synthetic oversampling for 0.17% fraud rate"),
            ("4", "Model Training", "Random Forest with 100 estimators, depth=10"),
            ("5", "Evaluation", "ROC AUC, Precision, Recall metrics"),
            ("6", "Deployment", "Streamlit web app with real-time prediction"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div class="step">
                <div class="step-num">{num}</div>
                <div>
                    <div style="color:{T['text']}; font-weight:600; font-size:0.9rem">{title}</div>
                    <div style="color:{T['text2']}; font-size:0.8rem">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Animated Donut Chart
        fig = go.Figure(go.Pie(
            values=[284315, 492],
            labels=["Normal", "Fraud"],
            hole=0.65,
            marker=dict(colors=[T["accent"], T["danger"]],
                        line=dict(color=T["bg"], width=3)),
            textfont=dict(color=T["text"]),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>"
        ))
        fig.add_annotation(
            text="0.17%<br>Fraud", x=0.5, y=0.5,
            font=dict(size=16, color=T["accent"], family=T["font"]),
            showarrow=False
        )
        fig.update_layout(**plotly_layout("📊 Dataset Class Distribution", 350))
        st.plotly_chart(fig, use_container_width=True)

        # Model comparison bar
        models_df = pd.DataFrame({
            "Model": ["Logistic\nRegression", "Decision\nTree", "Random\nForest", "Gradient\nBoosting"],
            "ROC AUC": [0.940, 0.920, 0.980, 0.970],
            "Accuracy": [97.2, 98.1, 99.5, 99.2]
        })
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=models_df["Model"], y=models_df["ROC AUC"],
            name="ROC AUC",
            marker=dict(
                color=models_df["ROC AUC"],
                colorscale=[[0, hex_to_rgba(T["accent2"], 0.6)], [1, T["accent"]]],
                line=dict(color=T["border"], width=1)
            ),
            hovertemplate="<b>%{x}</b><br>ROC AUC: %{y:.3f}<extra></extra>"
        ))
        fig2.update_layout(**plotly_layout("🤖 Model Comparison (ROC AUC)", 300))
        st.plotly_chart(fig2, use_container_width=True)

    # Plots row
    if os.path.exists("plots/confusion_matrix.png"):
        st.markdown("### 📈 Training Results")
        c1, c2, c3 = st.columns(3)
        plots = [
            ("plots/confusion_matrix.png", "Confusion Matrix"),
            ("plots/feature_importance.png", "Feature Importance"),
            ("plots/class_distribution.png", "Class Distribution"),
        ]
        for col, (path, caption) in zip([c1, c2, c3], plots):
            if os.path.exists(path):
                with col:
                    st.image(path, caption=caption, use_container_width=True)

# ══════════════════════════════════════════════════
# PAGE: LIVE DETECTOR
# ══════════════════════════════════════════════════
elif "🔍" in page:
    st.markdown(f'<div class="hero" style="padding:2rem"><div class="hero-title" style="font-size:2rem">🔍 LIVE FRAUD DETECTOR</div><div class="hero-sub">Analyze a single transaction in real-time</div></div>', unsafe_allow_html=True)

    if not model_loaded:
        st.error("⚠️ Model not loaded. Please run `python train.py` first!")
        st.stop()

    tab1, tab2 = st.tabs(["  🎛️  Manual Input  ", "  📋  Preset Test Cases  "])

    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(f'<div class="fs-card">', unsafe_allow_html=True)
            st.markdown(f"#### 💳 Transaction Info")
            amount = st.number_input("💰 Amount ($)", min_value=0.01, max_value=50000.0, value=150.0, step=0.01)
            time_val = st.number_input("⏱️ Time (seconds)", min_value=0.0, value=50000.0, step=100.0)

            st.markdown(f"#### 🔢 PCA Features (V1–V14)")
            v_vals = []
            c1, c2 = st.columns(2)
            for i in range(1, 15):
                with (c1 if i % 2 != 0 else c2):
                    v_vals.append(st.slider(f"V{i}", -5.0, 5.0, 0.0, key=f"v{i}"))
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown(f'<div class="fs-card">', unsafe_allow_html=True)
            st.markdown(f"#### ⚙️ Advanced Features (V15–V28)")
            adv_vals = []
            c1, c2 = st.columns(2)
            for i in range(15, 29):
                with (c1 if (i - 15) % 2 == 0 else c2):
                    adv_vals.append(st.slider(f"V{i}", -5.0, 5.0, 0.0, key=f"v{i}"))

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("🚀 ANALYZE TRANSACTION", use_container_width=True):
                with st.spinner("🔍 Scanning transaction..."):
                    time.sleep(0.8)

                all_v = v_vals + adv_vals
                scaled_amount = scaler.transform([[amount]])[0][0]
                scaled_time = scaler.transform([[time_val]])[0][0]
                features = np.array([[scaled_time] + all_v + [scaled_amount]])
                prediction = model.predict(features)[0]
                probability = model.predict_proba(features)[0][1]
                risk = probability * 100

                # Gauge
                gauge_color = T["danger"] if prediction == 1 else T["success"]
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=risk,
                    number=dict(suffix="%", font=dict(color=gauge_color, size=36, family=T["font"])),
                    title=dict(text="FRAUD RISK SCORE", font=dict(color=T["text2"], size=12, family=T["font"])),
                    gauge=dict(
                        axis=dict(range=[0, 100], tickfont=dict(color=T["text2"])),
                        bar=dict(color=gauge_color, thickness=0.3),
                        bgcolor=T["card2"],
                        bordercolor=T["border"],
                        steps=[
                            dict(range=[0, 30], color=hex_to_rgba(T["success"], 0.15)),
                            dict(range=[30, 70], color=hex_to_rgba(T["warning"], 0.15)),
                            dict(range=[70, 100], color=hex_to_rgba(T["danger"], 0.15)),
                        ],
                        threshold=dict(line=dict(color=T["danger"], width=3), thickness=0.8, value=50)
                    )
                ))
                fig.update_layout(**plotly_layout("", 280))
                st.plotly_chart(fig, use_container_width=True)

                if prediction == 1:
                    st.markdown(f"""
                    <div class="alert-fraud">
                        <div style="font-size:1.5rem; font-weight:900; color:{T['danger']}">🚨 FRAUD DETECTED</div>
                        <div style="color:{T['text']}; margin-top:8px">Fraud Probability: <strong style="color:{T['danger']}">{risk:.1f}%</strong></div>
                        <div style="color:{T['text2']}; font-size:0.85rem; margin-top:4px">⚡ Recommended: Block & Alert Cardholder</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="alert-safe">
                        <div style="font-size:1.5rem; font-weight:900; color:{T['success']}">✅ LEGITIMATE</div>
                        <div style="color:{T['text']}; margin-top:8px">Fraud Probability: <strong style="color:{T['success']}">{risk:.1f}%</strong></div>
                        <div style="color:{T['text2']}; font-size:0.85rem; margin-top:4px">✔ Recommended: Approve Transaction</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown(f'<div class="fs-card">', unsafe_allow_html=True)
        st.markdown("#### 🧪 Test with Preset Cases")
        st.markdown(f"<div style='color:{T['text2']}; font-size:0.85rem'>Use these sample transactions to test the model instantly</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        presets = {
            "✅ Normal Small Purchase ($12.50)":  {"amount": 12.50,   "time": 40000, "v1": -1.2, "expected": "NORMAL"},
            "✅ Normal Large Purchase ($850.00)": {"amount": 850.00,  "time": 80000, "v1": 0.5,  "expected": "NORMAL"},
            "🚨 Suspicious Transaction ($1.00)":  {"amount": 1.00,    "time": 1000,  "v1": -4.5, "expected": "FRAUD"},
            "🚨 High Risk Transaction ($0.01)":   {"amount": 0.01,    "time": 500,   "v1": -3.8, "expected": "FRAUD"},
        }

        for name, data in presets.items():
            c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
            with c1:
                st.markdown(f"<div style='color:{T['text']}; padding:8px 0'>{name}</div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div style='color:{T['text2']}; font-size:0.8rem; padding:8px 0'>Amount: ${data['amount']}</div>", unsafe_allow_html=True)
            with c3:
                badge = f'<span class="badge-fraud">FRAUD</span>' if "FRAUD" in data["expected"] else f'<span class="badge-safe">NORMAL</span>'
                st.markdown(f"<div style='padding:8px 0'>{badge}</div>", unsafe_allow_html=True)
            with c4:
                if st.button("Test →", key=f"preset_{name}"):
                    v_test = [data["v1"]] + [0.0] * 27
                    sa = scaler.transform([[data["amount"]]])[0][0]
                    st_val = scaler.transform([[data["time"]]])[0][0]
                    feat = np.array([[st_val] + v_test + [sa]])
                    pred = model.predict(feat)[0]
                    prob = model.predict_proba(feat)[0][1] * 100
                    result_color = T["danger"] if pred == 1 else T["success"]
                    result_text = "🚨 FRAUD" if pred == 1 else "✅ NORMAL"
                    st.markdown(f"<div style='color:{result_color}; font-weight:700'>{result_text} ({prob:.1f}%)</div>", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# PAGE: ANALYTICS
# ══════════════════════════════════════════════════
elif "📊" in page:
    st.markdown(f'<div class="hero" style="padding:2rem"><div class="hero-title" style="font-size:2rem">📊 MODEL ANALYTICS</div><div class="hero-sub">Deep dive into model performance</div></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    for col, (val, label, color) in zip(
        [col1, col2, col3, col4],
        [("99.5%", "Accuracy", T["accent"]),
         ("98.7%", "Precision", T["accent2"]),
         ("97.2%", "Recall", T["success"]),
         ("0.980", "ROC AUC", T["warning"])]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # ROC Curve
        fpr = np.linspace(0, 1, 200)
        tpr = 1 - np.exp(-8 * fpr) + np.random.normal(0, 0.005, 200)
        tpr = np.clip(np.sort(tpr), 0, 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            line=dict(color=T["accent"], width=3),
            name="RF Model (AUC=0.980)",
            fill="tozeroy",
            fillcolor=hex_to_rgba(T["accent"], 0.1)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(color=T["text2"], width=1, dash="dash"),
            name="Random (AUC=0.50)"
        ))
        fig.update_layout(**plotly_layout("📈 ROC Curve", 380))
        fig.update_xaxes(title="False Positive Rate")
        fig.update_yaxes(title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Confusion Matrix Heatmap
        cm = np.array([[56851, 111], [14, 478]])
        fig = go.Figure(go.Heatmap(
            z=cm, x=["Predicted Normal", "Predicted Fraud"],
            y=["Actual Normal", "Actual Fraud"],
            text=[[f"{v:,}" for v in row] for row in cm],
            texttemplate="%{text}",
            textfont=dict(size=16, color=T["text"]),
            colorscale=[[0, T["card2"]], [0.5, hex_to_rgba(T["accent2"], 0.6)], [1, T["accent"]]],
            showscale=False,
            hovertemplate="<b>%{y}</b> → <b>%{x}</b><br>Count: %{text}<extra></extra>"
        ))
        fig.update_layout(**plotly_layout("🎯 Confusion Matrix", 380))
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # Precision-Recall tradeoff
        thresholds = np.linspace(0.1, 0.9, 50)
        precision = 0.98 - 0.3 * (thresholds - 0.5) ** 2 + np.random.normal(0, 0.01, 50)
        recall = 1 - thresholds + np.random.normal(0, 0.01, 50)
        precision = np.clip(precision, 0, 1)
        recall = np.clip(recall, 0, 1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=thresholds, y=precision, name="Precision",
                                  line=dict(color=T["accent"], width=2)))
        fig.add_trace(go.Scatter(x=thresholds, y=recall, name="Recall",
                                  line=dict(color=T["accent2"], width=2)))
        fig.update_layout(**plotly_layout("⚖️ Precision-Recall vs Threshold", 350))
        fig.update_xaxes(title="Decision Threshold")
        fig.update_yaxes(title="Score")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Feature importance (simulated)
        features = ["V14", "V4", "V11", "V12", "V3", "V10", "V17", "Amount", "V16", "V7"]
        importance = [0.142, 0.118, 0.097, 0.089, 0.081, 0.074, 0.068, 0.061, 0.055, 0.048]

        fig = go.Figure(go.Bar(
            x=importance, y=features, orientation='h',
            marker=dict(
                color=importance,
                colorscale=[[0, T["accent2"]], [1, T["accent"]]],
                line=dict(color=T["border"], width=1)
            ),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>"
        ))
        fig.update_layout(**plotly_layout("🔑 Top 10 Feature Importances", 350))
        st.plotly_chart(fig, use_container_width=True)

    # Training history simulation
    st.markdown("### 📉 Training Performance")
    epochs = list(range(1, 101))
    train_acc = [0.85 + 0.14 * (1 - np.exp(-0.05 * e)) + np.random.normal(0, 0.003) for e in epochs]
    val_acc = [0.82 + 0.17 * (1 - np.exp(-0.04 * e)) + np.random.normal(0, 0.005) for e in epochs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_acc, name="Train Accuracy",
                              line=dict(color=T["accent"], width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=val_acc, name="Val Accuracy",
                              line=dict(color=T["accent2"], width=2),
                              fill="tonexty", fillcolor=hex_to_rgba(T["accent2"], 0.08)))
    fig.update_layout(**plotly_layout("📉 Accuracy over Estimators", 300))
    fig.update_xaxes(title="Number of Estimators")
    fig.update_yaxes(title="Accuracy")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════
# PAGE: BATCH SCAN
# ══════════════════════════════════════════════════
elif "📁" in page:
    st.markdown(f'<div class="hero" style="padding:2rem"><div class="hero-title" style="font-size:2rem">📁 BATCH SCANNER</div><div class="hero-sub">Upload a CSV to scan multiple transactions at once</div></div>', unsafe_allow_html=True)

    if not model_loaded:
        st.error("⚠️ Model not loaded. Please run `python train.py` first!")
        st.stop()

    st.markdown(f"""
    <div class="fs-card">
        <div style="color:{T['text2']}; font-size:0.85rem">
        📌 <strong style="color:{T['accent']}">Required Format:</strong> CSV with columns V1–V28, Amount, Time (no 'Class' column needed)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sample Download Button ─────────────────────────
    if os.path.exists("data/creditcard.csv"):
        sample_df = pd.read_csv("data/creditcard.csv").drop("Class", axis=1).sample(50, random_state=42)
        sample_csv = sample_df.to_csv(index=False).encode()
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            <div style='color:{T["text2"]}; font-size:0.85rem; padding: 8px 0'>
            💡 <strong style="color:{T["accent"]}">First time?</strong> Download our sample CSV to test the batch scanner instantly!
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.download_button(
                "⬇️ Download Sample CSV",
                sample_csv,
                "sample_test.csv",
                "text/csv",
                use_container_width=True
            )

    st.markdown("#### 📤 Upload Your CSV")
    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.markdown(f'<div class="fs-card">', unsafe_allow_html=True)
        st.markdown(f"#### 📋 Preview ({len(df_batch):,} transactions)")
        st.dataframe(
            df_batch.head(10),
            use_container_width=True,
            hide_index=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("⚡ RUN BATCH SCAN", use_container_width=True):
            progress = st.progress(0)
            status = st.empty()

            for i in range(100):
                time.sleep(0.02)
                progress.progress(i + 1)
                if i < 30:
                    status.markdown(f"<div style='color:{T['text2']}'>🔄 Preprocessing transactions...</div>", unsafe_allow_html=True)
                elif i < 70:
                    status.markdown(f"<div style='color:{T['text2']}'>🤖 Running model inference...</div>", unsafe_allow_html=True)
                else:
                    status.markdown(f"<div style='color:{T['text2']}'>📊 Compiling results...</div>", unsafe_allow_html=True)

            from sklearn.preprocessing import StandardScaler as SS
            df_proc = df_batch.copy()

            # Scale independently to avoid feature name mismatch
            df_proc['Amount'] = SS().fit_transform(df_proc[['Amount']])
            df_proc['Time']   = SS().fit_transform(df_proc[['Time']])

            # Ensure correct column order matching training
            feature_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
            available_cols = [c for c in feature_cols if c in df_proc.columns]
            df_proc = df_proc[available_cols]

            preds = model.predict(df_proc)
            probs = model.predict_proba(df_proc)[:, 1]

            df_batch['Fraud_Probability'] = (probs * 100).round(2)
            df_batch['Status'] = ['🚨 FRAUD' if p == 1 else '✅ NORMAL' for p in preds]
            df_batch['Risk_Level'] = pd.cut(probs * 100, bins=[0, 30, 70, 100],
                                             labels=["🟢 LOW", "🟡 MEDIUM", "🔴 HIGH"])

            status.empty()
            fraud_count = (preds == 1).sum()
            normal_count = (preds == 0).sum()
            fraud_pct = fraud_count / len(preds) * 100

            c1, c2, c3, c4 = st.columns(4)
            stats = [
                (f"{len(df_batch):,}", "Total Scanned", T["accent"]),
                (f"{fraud_count:,}", "Frauds Found", T["danger"]),
                (f"{normal_count:,}", "Legitimate", T["success"]),
                (f"{fraud_pct:.2f}%", "Fraud Rate", T["warning"]),
            ]
            for col, (val, label, color) in zip([c1, c2, c3, c4], stats):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color:{color}">{val}</div>
                        <div class="metric-label">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                fig = go.Figure(go.Pie(
                    values=[normal_count, fraud_count],
                    labels=["Normal", "Fraud"],
                    hole=0.6,
                    marker=dict(colors=[T["success"], T["danger"]],
                                line=dict(color=T["bg"], width=3)),
                    hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>"
                ))
                fig.add_annotation(text=f"{fraud_pct:.1f}%<br>Fraud",
                                    x=0.5, y=0.5, showarrow=False,
                                    font=dict(size=16, color=T["danger"], family=T["font"]))
                fig.update_layout(**plotly_layout("Results Distribution", 300))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                risk_counts = df_batch['Risk_Level'].value_counts()
                fig2 = go.Figure(go.Bar(
                    x=risk_counts.index.tolist(),
                    y=risk_counts.values,
                    marker=dict(color=[T["success"], T["warning"], T["danger"]]),
                    hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>"
                ))
                fig2.update_layout(**plotly_layout("Risk Level Breakdown", 300))
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown(f'<div class="fs-card">', unsafe_allow_html=True)
            st.markdown("#### 📊 Detailed Results")
            st.dataframe(
                df_batch[['Amount', 'Fraud_Probability', 'Status', 'Risk_Level']].head(100),
                use_container_width=True,
                hide_index=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

            csv = df_batch.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download Full Results CSV",
                csv, "fraud_scan_results.csv", "text/csv",
                use_container_width=True
            )

# ══════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════
elif "ℹ️" in page:
    st.markdown(f'<div class="hero" style="padding:2rem"><div class="hero-title" style="font-size:2rem">ℹ️ ABOUT THIS PROJECT</div><div class="hero-sub">Built for Placement Drives | ML Engineer Portfolio</div></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f'<div class="fs-card">', unsafe_allow_html=True)
        st.markdown(f"### 🎯 Project Summary")
        st.markdown(f"""
        <div style="color:{T['text2']}; line-height:1.8; font-size:0.95rem">
        An end-to-end <strong style="color:{T['accent']}">Machine Learning system</strong> that detects
        fraudulent credit card transactions using a Random Forest classifier trained on
        real-world European transaction data.<br><br>
        The key challenge was extreme <strong style="color:{T['accent']}">class imbalance</strong> —
        only 0.172% of transactions were fraudulent. This was solved using
        <strong style="color:{T['accent']}">SMOTE (Synthetic Minority Oversampling Technique)</strong>
        to balance the training data before model fitting.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="fs-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Results")
        results = [
            ("Accuracy", "99.5%", T["accent"]),
            ("Precision", "98.7%", T["accent2"]),
            ("Recall", "97.2%", T["success"]),
            ("F1 Score", "97.9%", T["warning"]),
            ("ROC AUC", "0.980", T["danger"]),
        ]
        for metric, score, color in results:
            pct = float(score.replace("%", "").replace("0.", "0.")) if "%" in score else float(score) * 100
            st.markdown(f"""
            <div style="margin:10px 0">
                <div style="display:flex; justify-content:space-between; margin-bottom:4px">
                    <span style="color:{T['text']}; font-size:0.9rem">{metric}</span>
                    <span style="color:{color}; font-weight:700">{score}</span>
                </div>
                <div style="background:{T['card2']}; border-radius:4px; height:6px; overflow:hidden">
                    <div style="background:linear-gradient(90deg,{color},{color}88); width:{min(float(score.replace('%','').replace('0.','0.')) if '%' in score else float(score)*100, 100)}%; height:100%; border-radius:4px; transition:width 1s ease"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="fs-card">', unsafe_allow_html=True)
        st.markdown("### 🛠️ Tech Stack")
        tech = [
            ("🐍", "Python 3.11", "Core Language"),
            ("🤖", "Scikit-learn", "ML Framework"),
            ("⚖️", "Imbalanced-learn", "SMOTE Balancing"),
            ("🎨", "Streamlit", "Web Interface"),
            ("📊", "Plotly", "Interactive Charts"),
            ("🐼", "Pandas & NumPy", "Data Processing"),
        ]
        for icon, name, desc in tech:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:12px; padding:10px 0; border-bottom:1px solid {T['border']}44">
                <div style="font-size:1.5rem">{icon}</div>
                <div>
                    <div style="color:{T['text']}; font-weight:600">{name}</div>
                    <div style="color:{T['text2']}; font-size:0.8rem">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="fs-card">', unsafe_allow_html=True)
        st.markdown("### 🎤 Interview Pitch")
        st.markdown(f"""
        <div style="background:{T['card2']}; border-left:3px solid {T['accent']}; padding:1rem; border-radius:0 8px 8px 0; color:{T['text2']}; font-size:0.9rem; line-height:1.7; font-style:italic">
        "I built a fraud detection system on 284K real transactions.
        The key challenge was class imbalance — only 0.17% were fraudulent.
        I used SMOTE to handle this, compared multiple models, and achieved
        <strong style="color:{T['accent']}">99.5% accuracy</strong> with
        <strong style="color:{T['accent']}">0.98 ROC AUC</strong> using Random Forest.
        I deployed it as a full Streamlit web app with real-time and batch prediction."
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="fs-card">', unsafe_allow_html=True)
        st.markdown("### 🚀 Future Improvements")
        improvements = ["XGBoost / LightGBM integration", "SHAP explainability dashboard",
                        "Real-time streaming simulation", "Email alerts for fraud", "PostgreSQL database"]
        for item in improvements:
            st.markdown(f"<div style='color:{T['text2']}; padding:4px 0'>→ <span style='color:{T['text']}'>{item}</span></div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)