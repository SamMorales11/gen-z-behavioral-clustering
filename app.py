import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="Gen-Z Behavioral Archetype Engine",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom enterprise CSS styling (Glassmorphism, clean typography, zero emojis)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #0b0f17;
        color: #e2e8f0;
    }
    
    .header-container {
        padding: 8px 0 24px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
        margin-bottom: 24px;
    }
    
    .card-glass {
        background: rgba(18, 24, 38, 0.65);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 16px;
    }
    
    .kpi-tile {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 16px;
        text-align: left;
    }
    
    .kpi-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 6px;
        font-weight: 500;
    }
    
    .kpi-value {
        font-size: 22px;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
    }
    
    .kpi-sub {
        font-size: 11px;
        color: #64748b;
        margin-top: 4px;
    }
    
    .metric-badge {
        display: inline-block;
        font-size: 10.5px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 4px;
        margin-bottom: 14px;
    }
    
    .protocol-step {
        background: rgba(15, 23, 42, 0.5);
        border-left: 3px solid #334155;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 13px;
        line-height: 1.5;
        color: #cbd5e1;
    }
    
    .stSlider > label, .stSelectbox > label {
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #cbd5e1 !important;
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
    st.error(f"Inference initialization failed. Missing model artifacts in models/: {e}")
    st.stop()

PERSONA_REGISTRY = {
    0: {
        "name": "The Casual Browser",
        "tag": "CLUSTER ARTIFACT: BALANCED DIURNAL PATTERN",
        "color": "#38bdf8",
        "bg_badge": "rgba(56, 189, 248, 0.12)",
        "desc": "Telemetric profile indicates moderate cumulative exposure predominantly distributed across daytime windows. Demonstrates normative session lengths with minimal pre-sleep sleep phase latency disruption.",
        "protocols": [
            "Maintain daytime-only application access schedules to avoid drifting into late-night usage windows.",
            "Conduct routine checks on passive continuous scrolling triggers during unstructured leisure hours."
        ],
        "benchmark": {"daily_hours": 3.8, "session_min": 22.0, "sleep_screentime": 25.0, "frequency": 10.4}
    },
    1: {
        "name": "The Late-Night Doomscroller",
        "tag": "CLUSTER ARTIFACT: HIGH CIRCADIAN VULNERABILITY",
        "color": "#f43f5e",
        "bg_badge": "rgba(244, 63, 94, 0.12)",
        "desc": "Heavy engagement concentrated in the pre-sleep and midnight intervals. Empirically associated with sleep fragmentation, circadian misalignment, and the lowest well-being indices across the dataset.",
        "protocols": [
            "Establish a hardware-enforced isolation perimeter outside bedroom boundaries 45 minutes prior to target sleep onset.",
            "Enable strict OS-level scheduled blackouts for algorithmically driven feeds starting at 23:00.",
            "Replace high-luminance interactive media streams with non-screen acoustic stimuli."
        ],
        "benchmark": {"daily_hours": 7.5, "session_min": 45.0, "sleep_screentime": 90.0, "frequency": 10.0}
    },
    2: {
        "name": "The Mindful Minimalist",
        "tag": "CLUSTER ARTIFACT: OPTIMAL DIGITAL HYGIENE",
        "color": "#10b981",
        "bg_badge": "rgba(16, 185, 129, 0.12)",
        "desc": "Highly disciplined usage patterns characterized by controlled platform dispersion, negligible night activity, and short pre-sleep exposure. Correlates with the highest subjective well-being stability.",
        "protocols": [
            "Preserve established temporal gates and selective application usage parameters.",
            "Audit secondary permissions and notifications periodically to prevent attention scope expansion."
        ],
        "benchmark": {"daily_hours": 1.5, "session_min": 15.0, "sleep_screentime": 10.0, "frequency": 6.0}
    },
    3: {
        "name": "The Hyper-Connected Micro-Checker",
        "tag": "CLUSTER ARTIFACT: ACUTE ATTENTION FRAGMENTATION",
        "color": "#fbbf24",
        "bg_badge": "rgba(251, 191, 36, 0.12)",
        "desc": "Extremely frequent session entry points paired with compressed durations across multi-platform networks. Reflects persistent dopamine micro-loop initiation and fragmented continuous partial attention.",
        "protocols": [
            "Eliminate all real-time badge counts, banner alerts, and non-synchronous notification queues.",
            "Transition messaging and application engagement into batched review windows (e.g., three scheduled intervals per day).",
            "Increase motor interaction friction by removing home screen shortcuts and utilizing nested folder storage."
        ],
        "benchmark": {"daily_hours": 5.2, "session_min": 10.0, "sleep_screentime": 35.0, "frequency": 31.2}
    }
}

# Header Section
st.markdown("""
<div class="header-container">
    <h2 style="margin: 0; font-weight: 700; letter-spacing: -0.02em; font-size: 26px; color: #f8fafc;">
        Gen-Z Behavioral Archetype Diagnostic Engine
    </h2>
    <p style="margin: 6px 0 0 0; font-size: 13.5px; color: #94a3b8;">
        Unsupervised machine learning inference and behavioral telemetry profiling trained on 1,000,000 synthesized records.
    </p>
</div>
""", unsafe_allow_html=True)

col_inputs, col_outputs = st.columns([1, 1.4], gap="large")

with col_inputs:
    st.markdown("#### Input Telemetry Parameters")
    
    daily_hours = st.slider("Total Daily Usage Duration (Hours)", 0.5, 16.0, 4.5, 0.5)
    avg_session = st.slider("Mean Session Length (Minutes)", 5, 120, 25, 5)
    num_platforms = st.slider("Active Network Diversity (Platforms Count)", 1, 8, 3, 1)
    night_usage = st.selectbox("Midnight Interval Activity (00:00 - 05:00 Window)", ["Inactive", "Active"])
    screen_sleep = st.slider("Pre-Sleep Screen Exposure (Minutes)", 0, 180, 45, 5)
    
    night_val = 1 if night_usage == "Active" else 0
    session_freq = (daily_hours * 60) / max(avg_session, 1)
    sleep_ratio = (screen_sleep / max(daily_hours * 60, 1)) * 100
    
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    btn_predict = st.button("Execute Model Inference", type="primary", use_container_width=True)

with col_outputs:
    if btn_predict:
        input_features = np.array([[
            daily_hours,
            avg_session,
            num_platforms,
            night_val,
            screen_sleep,
            session_freq,
            sleep_ratio
        ]])
        
        # Scaling & Prediction
        input_scaled = scaler.transform(input_features)
        cluster_idx = int(kmeans.predict(input_scaled)[0])
        persona = PERSONA_REGISTRY[cluster_idx]
        
        # Calculate cluster proximity / confidence via inverse euclidean distance to centroids
        centroids = kmeans.cluster_centers_
        distances = np.linalg.norm(centroids - input_scaled, axis=1)
        inverse_distances = 1.0 / (distances + 1e-6)
        cluster_affinity = (inverse_distances / np.sum(inverse_distances)) * 100
        
        # Diagnostic Metric Row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="kpi-tile">
                <div class="kpi-label">Estimated Pickups</div>
                <div class="kpi-value">{int(round(session_freq))}</div>
                <div class="kpi-sub">Sessions / Day</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="kpi-tile">
                <div class="kpi-label">Pre-Sleep Saturation</div>
                <div class="kpi-value">{sleep_ratio:.1f}%</div>
                <div class="kpi-sub">Of Total Daily Exposure</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="kpi-tile">
                <div class="kpi-label">Model Confidence</div>
                <div class="kpi-value">{cluster_affinity[cluster_idx]:.1f}%</div>
                <div class="kpi-sub">Cluster Assignment Fit</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        
        # Primary Persona Diagnostic Card
        st.markdown(f"""
        <div class="card-glass" style="border-left: 4px solid {persona['color']};">
            <span class="metric-badge" style="background-color: {persona['bg_badge']}; color: {persona['color']};">
                {persona['tag']}
            </span>
            <h3 style="margin: 0 0 8px 0; font-size: 21px; font-weight: 700; color: #f8fafc;">
                {persona['name']}
            </h3>
            <p style="margin: 0; font-size: 13.5px; line-height: 1.6; color: #94a3b8;">
                {persona['desc']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tabs for Analytical Deep-Dive
        tab_radar, tab_affinity = st.tabs(["Telemetry Benchmark Comparison", "Cluster Distribution Vector"])
        
        with tab_radar:
            radar_labels = ['Total Hours', 'Session Length', 'Pre-Sleep Exposure', 'Session Frequency']
            user_values = [daily_hours, avg_session, screen_sleep, session_freq]
            bench_values = [
                persona['benchmark']['daily_hours'],
                persona['benchmark']['session_min'],
                persona['benchmark']['sleep_screentime'],
                persona['benchmark']['frequency']
            ]
            
            normalization_ceilings = [16.0, 120.0, 180.0, 40.0]
            norm_user = [min(u / m * 100, 100) for u, m in zip(user_values, normalization_ceilings)]
            norm_bench = [min(b / m * 100, 100) for b, m in zip(bench_values, normalization_ceilings)]
            
            fig_radar = go.Figure()
            
            fig_radar.add_trace(go.Scatterpolar(
                r=norm_user + [norm_user[0]],
                theta=radar_labels + [radar_labels[0]],
                fill='toself',
                fillcolor=persona['bg_badge'],
                name='User Observation',
                line=dict(color=persona['color'], width=2)
            ))
            
            fig_radar.add_trace(go.Scatterpolar(
                r=norm_bench + [norm_bench[0]],
                theta=radar_labels + [radar_labels[0]],
                name='Cluster Mean Benchmark',
                line=dict(color='#475569', width=1.5, dash='dot')
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        showticklabels=False,
                        linecolor='rgba(255,255,255,0.06)',
                        gridcolor='rgba(255,255,255,0.06)'
                    ),
                    angularaxis=dict(
                        linecolor='rgba(255,255,255,0.08)',
                        gridcolor='rgba(255,255,255,0.08)',
                        color='#94a3b8',
                        tickfont=dict(size=11)
                    )
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=45, r=45, t=25, b=25),
                height=260,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.25,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=11, color='#94a3b8')
                )
            )
            st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})
            
        with tab_affinity:
            affinity_data = pd.DataFrame({
                'Archetype': [PERSONA_REGISTRY[i]['name'] for i in range(4)],
                'Proximity Probability (%)': cluster_affinity
            }).sort_values(by='Proximity Probability (%)', ascending=True)
            
            fig_bar = go.Figure(go.Bar(
                x=affinity_data['Proximity Probability (%)'],
                y=affinity_data['Archetype'],
                orientation='h',
                marker=dict(
                    color=['#334155' if name != persona['name'] else persona['color'] for name in affinity_data['Archetype']],
                    line=dict(color='rgba(255,255,255,0.1)', width=1)
                )
            ))
            
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=20, t=15, b=15),
                height=230,
                xaxis=dict(
                    title=dict(text="Centroid Proximity Affinity (%)", font=dict(size=11, color='#94a3b8')),
                    color='#94a3b8',
                    gridcolor='rgba(255,255,255,0.05)',
                    range=[0, 100]
                ),
                yaxis=dict(
                    color='#cbd5e1',
                    tickfont=dict(size=11)
                )
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
            
        st.markdown("#### Structured Behavioral Protocols")
        for protocol in persona['protocols']:
            st.markdown(f'<div class="protocol-step">{protocol}</div>', unsafe_allow_html=True)
            
    else:
        st.markdown("""
        <div class="card-glass" style="border: 1px dashed rgba(255,255,255,0.14); text-align: center; padding: 56px 24px;">
            <p style="margin: 0; color: #94a3b8; font-size: 14px; line-height: 1.6;">
                Adjust behavioral telemetry inputs in the configuration panel and trigger execution to evaluate cluster assignment and comparative metrics.
            </p>
        </div>
        """, unsafe_allow_html=True)