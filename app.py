import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="Gen-Z Behavioral Archetype Engine",
    layout="wide"
)

# Custom CSS for modern glassmorphism & analytics dashboard aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background-color: #0f1117;
        color: #e6edf3;
    }
    
    .card-glass {
        background: rgba(22, 27, 34, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
    }
    
    .metric-badge {
        display: inline-block;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    
    .intervention-card {
        background: rgba(13, 17, 23, 0.6);
        border-left: 3px solid #30363d;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin-bottom: 8px;
        font-size: 13.5px;
        color: #c9d1d9;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = joblib.load('models/kmeans_k4.joblib')
    scaler = joblib.load('models/scaler.joblib')
    return model, scaler

try:
    kmeans, scaler = load_artifacts()
except Exception as e:
    st.error(f"Failed to load model artifacts from models/ directory. Details: {e}")
    st.stop()

PERSONA_REGISTRY = {
    0: {
        "name": "The Casual Browser",
        "tag": "PROFILE: BALANCED CONSUMPTION",
        "color": "#38bdf8",
        "bg_badge": "rgba(56, 189, 248, 0.15)",
        "desc": "Digital consumption remains within moderate thresholds and predominantly occurs during daylight hours. Demonstrates healthy session boundaries with minimal circadian interference.",
        "protocols": [
            "Monitor weekend screen time thresholds to prevent escalation into compulsive browsing sessions.",
            "Maintain consistent content consumption windows outside of designated recovery hours."
        ],
        "benchmark": {"daily_hours": 3.8, "session_min": 22.0, "sleep_screentime": 25.0}
    },
    1: {
        "name": "The Late-Night Doomscroller",
        "tag": "PROFILE: HIGH RISK / CIRCADIAN DISRUPTION",
        "color": "#f43f5e",
        "bg_badge": "rgba(244, 63, 94, 0.15)",
        "desc": "Screen engagement is heavily skewed toward late-night hours and immediate pre-sleep windows. Strongly correlated with reduced psychological well-being and circadian misalignment.",
        "protocols": [
            "Enforce physical device isolation from the immediate sleep environment at least 45 minutes prior to bedtime.",
            "Configure scheduled application downtime lockouts beginning at 11:00 PM.",
            "Substitute active algorithmic content consumption with passive, non-screen inputs."
        ],
        "benchmark": {"daily_hours": 7.5, "session_min": 45.0, "sleep_screentime": 90.0}
    },
    2: {
        "name": "The Mindful Minimalist",
        "tag": "PROFILE: OPTIMAL REGULATION",
        "color": "#10b981",
        "bg_badge": "rgba(16, 185, 129, 0.15)",
        "desc": "Purpose-driven digital engagement characterized by controlled platform dispersion and negligible pre-sleep screen exposure. Correlated with the highest well-being stability scores.",
        "protocols": [
            "Sustain established digital boundaries and intentional application entry triggers.",
            "Perform periodic notification audits across newly installed services to preserve cognitive focus."
        ],
        "benchmark": {"daily_hours": 1.5, "session_min": 15.0, "sleep_screentime": 10.0}
    },
    3: {
        "name": "The Hyper-Connected Micro-Checker",
        "tag": "PROFILE: HIGH ATTENTION FRAGMENTATION",
        "color": "#fbbf24",
        "bg_badge": "rgba(251, 191, 36, 0.15)",
        "desc": "Exceptionally elevated session frequency paired with fragmented, short-duration visits across multiple platforms, indicating reflexive checking compulsions and continuous task switching.",
        "protocols": [
            "Disable all non-essential visual push notifications and application badges.",
            "Consolidate messaging and social check-ins into fixed, batched time intervals.",
            "Introduce operational friction (e.g., nesting apps in multi-level folders) to break motor reflex loops."
        ],
        "benchmark": {"daily_hours": 5.2, "session_min": 10.0, "sleep_screentime": 35.0}
    }
}

st.subheader("Gen-Z Digital Behavioral Archetype Engine")
st.caption("Machine learning inference system for user behavior segmentation trained on 1,000,000 digital telemetry observations.")
st.write("")

col_params, col_output = st.columns([1, 1.3], gap="large")

with col_params:
    st.markdown("#### Behavioral Telemetry Inputs")
    
    daily_hours = st.slider("Daily Screen Time (Hours)", 0.5, 16.0, 4.5, 0.5)
    avg_session = st.slider("Average Session Duration (Minutes)", 5, 120, 25, 5)
    num_platforms = st.slider("Active Social Platforms", 1, 8, 3, 1)
    night_usage = st.selectbox("Late-Night Activity (12:00 AM - 5:00 AM)", ["Inactive", "Active"])
    screen_sleep = st.slider("Pre-Sleep Screen Time (Minutes)", 0, 180, 45, 5)
    
    night_val = 1 if night_usage == "Active" else 0
    session_freq = (daily_hours * 60) / max(avg_session, 1)
    sleep_ratio = (screen_sleep / max(daily_hours * 60, 1)) * 100
    
    st.write("")
    btn_run = st.button("Run Archetype Inference", type="primary", use_container_width=True)

with col_output:
    if btn_run:
        input_data = np.array([[
            daily_hours,
            avg_session,
            num_platforms,
            night_val,
            screen_sleep,
            session_freq,
            sleep_ratio
        ]])
        
        input_scaled = scaler.transform(input_data)
        cluster_id = int(kmeans.predict(input_scaled)[0])
        persona = PERSONA_REGISTRY[cluster_id]
        
        st.markdown("#### Diagnostic Results")
        
        st.markdown(f"""
        <div class="card-glass" style="border-left: 4px solid {persona['color']};">
            <span class="metric-badge" style="background-color: {persona['bg_badge']}; color: {persona['color']};">
                {persona['tag']}
            </span>
            <h2 style="margin: 0 0 10px 0; font-size: 22px; font-weight: 700; color: #f0f6fc;">
                {persona['name']}
            </h2>
            <p style="margin: 0; font-size: 14px; line-height: 1.6; color: #8b949e;">
                {persona['desc']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        radar_categories = ['Daily Hours', 'Session Duration', 'Pre-Sleep Usage', 'Check Frequency']
        user_metrics = [daily_hours, avg_session, screen_sleep, session_freq]
        bench_metrics = [
            persona['benchmark']['daily_hours'],
            persona['benchmark']['session_min'],
            persona['benchmark']['sleep_screentime'],
            (persona['benchmark']['daily_hours'] * 60) / max(persona['benchmark']['session_min'], 1)
        ]
        
        max_bounds = [16.0, 120.0, 180.0, 40.0]
        user_norm = [min(u / m * 100, 100) for u, m in zip(user_metrics, max_bounds)]
        bench_norm = [min(b / m * 100, 100) for b, m in zip(bench_metrics, max_bounds)]
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=user_norm + [user_norm[0]],
            theta=radar_categories + [radar_categories[0]],
            fill='toself',
            fillcolor=persona['bg_badge'],
            name='User Profile Metrics',
            line=dict(color=persona['color'], width=2)
        ))
        
        fig_radar.add_trace(go.Scatterpolar(
            r=bench_norm + [bench_norm[0]],
            theta=radar_categories + [radar_categories[0]],
            name='Archetype Benchmark Mean',
            line=dict(color='#484f58', width=1.5, dash='dot')
        ))
        
        fig_radar.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    showticklabels=False,
                    linecolor='rgba(255,255,255,0.05)',
                    gridcolor='rgba(255,255,255,0.05)'
                ),
                angularaxis=dict(
                    linecolor='rgba(255,255,255,0.08)',
                    gridcolor='rgba(255,255,255,0.08)',
                    color='#8b949e',
                    tickfont=dict(size=11)
                )
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=50, r=50, t=20, b=20),
            height=250,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25,
                xanchor="center",
                x=0.5,
                font=dict(size=11, color='#8b949e')
            )
        )
        
        st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("#### Digital Hygiene Protocols")
        for protocol in persona['protocols']:
            st.markdown(f'<div class="intervention-card">{protocol}</div>', unsafe_allow_html=True)
            
    else:
        st.markdown("""
        <div class="card-glass" style="border: 1px dashed rgba(255,255,255,0.15); text-align: center; padding: 48px 24px;">
            <p style="margin: 0; color: #8b949e; font-size: 14px;">
                Configure behavioral telemetry parameters in the left panel and click the inference button to map your digital archetype.
            </p>
        </div>
        """, unsafe_allow_html=True)