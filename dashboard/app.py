"""
app.py
Real-Time Air Traffic Congestion Forecasting & Intelligent Airport Decision Support Console
Powered by Deep Learning (PyTorch LSTM & GRU) and Streamlit.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference import AirTrafficPredictor
from dashboard.simulation import TrafficSimulator
from src.decision_support import CONGESTION_LABELS, CONGESTION_COLORS

# Page Configuration
st.set_page_config(
    page_title="SkyFlow | Real-Time Air Traffic Congestion Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Flight Ops Theme
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .badge-high {
        background-color: #ef4444;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-medium {
        background-color: #eab308;
        color: black;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-low {
        background-color: #22c55e;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .advisory-box {
        background: #0f172a;
        border-left: 5px solid #38bdf8;
        padding: 14px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
    }
    .stProgress > div > div > div > div {
        background-color: #38bdf8;
    }
</style>
""", unsafe_allow_html=True)

# Airport Metadata
AIRPORT_INFO = {
    "HYD": {"name": "Hyderabad (Rajiv Gandhi Intl)", "city": "Hyderabad, IND", "capacity": 55},
    "DEL": {"name": "Delhi (Indira Gandhi Intl)", "city": "New Delhi, IND", "capacity": 90},
    "BOM": {"name": "Mumbai (Chhatrapati Shivaji Intl)", "city": "Mumbai, IND", "capacity": 65},
    "SFO": {"name": "San Francisco International", "city": "San Francisco, USA", "capacity": 70},
    "ATL": {"name": "Atlanta (Hartsfield-Jackson)", "city": "Atlanta, USA", "capacity": 120},
    "ORD": {"name": "Chicago O'Hare International", "city": "Chicago, USA", "capacity": 100},
    "LAX": {"name": "Los Angeles International", "city": "Los Angeles, USA", "capacity": 95}
}

# Session State Initialization
if "selected_airport" not in st.session_state:
    st.session_state.selected_airport = "HYD"
if "model_type" not in st.session_state:
    st.session_state.model_type = "LSTM"
if "simulator" not in st.session_state:
    st.session_state.simulator = TrafficSimulator(st.session_state.selected_airport)
if "auto_play" not in st.session_state:
    st.session_state.auto_play = False

# Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/airplane-mode-on.png", width=64)
    st.title("Radar Control Center")
    
    # Airport Selector
    airport_code = st.selectbox(
        "Select Airport Hub",
        options=list(AIRPORT_INFO.keys()),
        format_func=lambda code: f"{code} - {AIRPORT_INFO[code]['name']}",
        index=list(AIRPORT_INFO.keys()).index(st.session_state.selected_airport)
    )
    
    if airport_code != st.session_state.selected_airport:
        st.session_state.selected_airport = airport_code
        st.session_state.simulator.set_airport(airport_code)
        st.rerun()

    current_apt = AIRPORT_INFO[airport_code]
    st.caption(f"📍 {current_apt['city']} | Nominal Capacity: **{current_apt['capacity']} ops/hr**")
    
    st.divider()
    
    # Model Selection
    st.subheader("🤖 Prediction Engine")
    selected_model = st.radio(
        "Inference Architecture",
        options=["LSTM", "GRU", "Random Forest", "Linear Regression"],
        help="Compare real-time predictions from proposed Deep Learning vs. baseline models."
    )
    st.session_state.model_type = selected_model
    
    st.divider()
    
    # Real-Time Simulation Controls
    st.subheader("⏱️ Live Stream Simulation")
    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        if st.button("⏩ Step (+1 Hr)", use_container_width=True):
            st.session_state.simulator.step_forward(1)
            st.rerun()
    with col_sim2:
        if st.button("🔄 Reset Feed", use_container_width=True):
            st.session_state.simulator.reset()
            st.rerun()
            
    auto_refresh = st.checkbox("🔴 Live Auto-Play (3s)", value=st.session_state.auto_play)
    st.session_state.auto_play = auto_refresh

    st.divider()
    
    # Scenario Stress Injections
    st.subheader("🌩️ Stress Scenario Injection")
    st.caption("Simulate real-world operational disruptions to evaluate LSTM response:")
    
    weather_disruption = st.slider(
        "Severe Weather Impact (min delay)",
        min_value=0, max_value=60, value=0, step=5,
        help="Simulate dense fog, thunderstorm, or monsoon cell."
    )
    
    traffic_surge = st.slider(
        "Holiday / Special Event Surge (%)",
        min_value=0, max_value=50, value=0, step=10,
        help="Simulate unexpected passenger influx or diverted traffic."
    )
    
    runway_reduction = st.slider(
        "Runway Maintenance Closure (%)",
        min_value=0, max_value=50, value=0, step=25,
        help="Simulate partial runway closure reducing effective capacity."
    )

# Retrieve current operational window from simulator
simulator = st.session_state.simulator
recent_df = simulator.get_recent_history(
    lookback=6,
    weather_injection=weather_disruption,
    surge_injection=traffic_surge,
    runway_reduction=runway_reduction
)

# Run Inference
@st.cache_resource
def get_predictor(model_name: str):
    return AirTrafficPredictor(model_type=model_name)

predictor = get_predictor(st.session_state.model_type)

if len(recent_df) >= 6:
    pred_results = predictor.predict_sequence(
        recent_df, 
        airport=airport_code, 
        capacity=current_apt["capacity"]
    )
else:
    st.warning("⚠️ Insufficient historical sequence data. Please ensure data generation script has completed.")
    st.stop()

# Header Section
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("🛫 Air Traffic Congestion Forecasting & DSS")
    current_time_str = recent_df.iloc[-1].get("timestamp", "2026-10-27 12:00:00")
    st.markdown(f"**Airport:** `{airport_code}` — {current_apt['name']} | **Simulation Clock:** `{current_time_str}`")

with col_head2:
    st.markdown("<div style='text-align: right; padding-top: 15px;'>", unsafe_allow_html=True)
    if pred_results["congestion_label"] == "HIGH":
        st.markdown('<span class="badge-high">🔴 HIGH CONGESTION</span>', unsafe_allow_html=True)
    elif pred_results["congestion_label"] == "MEDIUM":
        st.markdown('<span class="badge-medium">🟡 MEDIUM CONGESTION</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-low">🟢 LOW CONGESTION</span>', unsafe_allow_html=True)
    st.markdown(f"<small>Confidence: <b>{pred_results['confidence']}%</b> | Horizon: <b>Next 1 Hr</b></small></div>", unsafe_allow_html=True)

st.markdown("---")

# Main KPI Row
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.metric(
        label="Current Flight Volume",
        value=f"{pred_results['current_flights']} flights",
        help="Total arriving and departing operations in current hour."
    )

with kpi2:
    delta_flights = pred_results['predicted_flights'] - pred_results['current_flights']
    st.metric(
        label="Predicted Next-Hour Flights",
        value=f"{pred_results['predicted_flights']} flights",
        delta=f"{delta_flights:+d} flights",
        delta_color="inverse" if delta_flights > 10 else "normal",
        help="Deep Learning forecast for the subsequent 60-minute window."
    )

with kpi3:
    st.metric(
        label="Avg Departure Delay",
        value=f"{pred_results['current_delay_min']:.1f} min",
        delta=f"{recent_df.iloc[-1]['weather_delay_factor']:.0f}m wx",
        delta_color="off"
    )

with kpi4:
    st.metric(
        label="Runway Utilization",
        value=f"{pred_results['current_runway_utilization']:.1f}%",
        delta=f"Forecast: {pred_results['predicted_runway_utilization']:.1f}%",
        delta_color="inverse" if pred_results['predicted_runway_utilization'] > 85 else "normal"
    )

with kpi5:
    st.metric(
        label="Airport Nominal Capacity",
        value=f"{current_apt['capacity']} ops/hr",
        delta=f"Active Model: {st.session_state.model_type}"
    )

st.markdown("<br>", unsafe_allow_html=True)

# Tabs Layout
tab_dashboard, tab_time_series, tab_dss, tab_benchmarks = st.tabs([
    "📊 Real-Time Operations", 
    "📈 Time-Series Deep Forecast", 
    "🛡️ Decision Support Directives (ATC)", 
    "🏆 Model Benchmark & Evaluation"
])

# ----------------- TAB 1: REAL-TIME OPERATIONS -----------------
with tab_dashboard:
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("🎛️ Runway Capacity Gauge")
        
        # Plotly Gauge Chart for Runway Utilization
        gauge_val = pred_results['predicted_runway_utilization']
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=gauge_val,
            title={'text': "Predicted Runway Load (%)", 'font': {'size': 18}},
            delta={'reference': pred_results['current_runway_utilization'], 'suffix': "%"},
            gauge={
                'axis': {'range': [None, 120], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#38bdf8"},
                'bgcolor': "#1e293b",
                'borderwidth': 2,
                'bordercolor': "#475569",
                'steps': [
                    {'range': [0, 65], 'color': 'rgba(34, 197, 94, 0.3)'},
                    {'range': [65, 85], 'color': 'rgba(234, 179, 8, 0.3)'},
                    {'range': [85, 120], 'color': 'rgba(239, 68, 68, 0.35)'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 85.0
                }
            }
        ))
        fig_gauge.update_layout(
            height=260,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown(f"""
        **Congestion Class Probabilities:**
        - 🟢 **Low:** {pred_results['class_probabilities']['Low']}%
        - 🟡 **Medium:** {pred_results['class_probabilities']['Medium']}%
        - 🔴 **High:** {pred_results['class_probabilities']['High']}%
        """)
        
    with col_right:
        st.subheader("⚡ Immediate Operations Summary")
        atc = pred_results["atc_advisories"]
        
        st.info(f"**Primary Runway Configuration Strategy:**\n{atc['runway_strategy']}")
        
        st.markdown("##### Active Tactical Directives:")
        for action in atc["flow_actions"]:
            st.markdown(f"- ✈️ {action}")

# ----------------- TAB 2: TIME-SERIES DEEP FORECAST -----------------
with tab_time_series:
    st.subheader("📈 Rolling Sequence (Lookback = 6 Hours) & Next-Hour Projection")
    st.caption("The LSTM processes this exact rolling multi-feature sequence to synthesize temporal trends:")
    
    # Prepare sequence DataFrame for Plotly
    plot_df = recent_df.copy()
    plot_df["Time_Step"] = [f"T-{5-i} hr" if i < 5 else "Current (T)" for i in range(len(plot_df))]
    
    # Add predicted next hour point
    next_step_row = {
        "Time_Step": "Forecast (T+1)",
        "total_flights": pred_results["predicted_flights"],
        "arriving_flights": int(pred_results["predicted_flights"] * 0.5),
        "departing_flights": int(pred_results["predicted_flights"] * 0.5),
        "avg_departure_delay": pred_results["current_delay_min"],
        "runway_utilization": pred_results["predicted_runway_utilization"]
    }
    extended_plot_df = pd.concat([plot_df, pd.DataFrame([next_step_row])], ignore_index=True)
    
    # Interactive Time-Series Line Chart
    fig_seq = go.Figure()
    
    # Observed Past 6 Hours
    fig_seq.add_trace(go.Scatter(
        x=plot_df["Time_Step"],
        y=plot_df["total_flights"],
        mode="lines+markers",
        name="Observed Flights",
        line=dict(color="#38bdf8", width=3),
        marker=dict(size=8)
    ))
    
    # Forecast Connection
    fig_seq.add_trace(go.Scatter(
        x=["Current (T)", "Forecast (T+1)"],
        y=[plot_df.iloc[-1]["total_flights"], pred_results["predicted_flights"]],
        mode="lines+markers",
        name=f"LSTM Forecast ({st.session_state.model_type})",
        line=dict(color="#f43f5e", width=3, dash="dash"),
        marker=dict(size=10, symbol="diamond")
    ))
    
    # Capacity Threshold Line
    fig_seq.add_hline(
        y=current_apt["capacity"],
        line_dash="dot",
        line_color="#ef4444",
        annotation_text="Nominal Airport Capacity",
        annotation_position="top left"
    )
    
    fig_seq.update_layout(
        title="Flight Traffic Progression & Deep Learning Projection",
        xaxis_title="Timeline",
        yaxis_title="Flight Operations / Hour",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1e293b",
        font=dict(color="white"),
        height=380
    )
    st.plotly_chart(fig_seq, use_container_width=True)
    
    # Historical Sequence Table
    st.markdown("##### Historical Sequence Tensor Vector (Input Features)")
    disp_cols = ["timestamp", "arriving_flights", "departing_flights", "total_flights", "avg_departure_delay", "weather_delay_factor", "runway_utilization"]
    st.dataframe(recent_df[disp_cols].tail(6), use_container_width=True)

# ----------------- TAB 3: DECISION SUPPORT DIRECTIVES -----------------
with tab_dss:
    st.subheader("🛡️ Air Traffic Control Decision Support Directives (ATCO / DSS)")
    atc = pred_results["atc_advisories"]
    
    st.markdown(f"**Alert Level:** `{atc['alert_level']}` | **Airport Capacity:** `{atc['capacity']} ops/hr`")
    
    col_dss1, col_dss2 = st.columns(2)
    
    with col_dss1:
        st.markdown("#### 📡 En-Route & Flow Management Directives")
        for act in atc["flow_actions"]:
            st.markdown(f"""
            <div class="advisory-box">
                <b>Flow Control Action:</b><br>{act}
            </div>
            """, unsafe_allow_html=True)
            
    with col_dss2:
        st.markdown("#### 🛫 Ground & Runway Management Strategy")
        st.markdown(f"""
        <div class="advisory-box" style="border-left-color: #22c55e;">
            <b>Active Strategy:</b><br>{atc['runway_strategy']}
        </div>
        """, unsafe_allow_html=True)
        
        for adv in atc["advisories"]:
            st.markdown(f"""
            <div class="advisory-box" style="border-left-color: #f59e0b;">
                <b>{adv['category']} [{adv['priority']}]:</b><br>{adv['action']}
            </div>
            """, unsafe_allow_html=True)

# ----------------- TAB 4: MODEL BENCHMARK & EVALUATION -----------------
with tab_benchmarks:
    st.subheader("🏆 Model Performance Benchmark & Comparative Analysis")
    st.caption("Comprehensive evaluation on held-out test sequence partition:")
    
    metrics_file = os.path.join(PROJECT_ROOT, "models", "metrics_comparison.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            benchmarks = json.load(f)
            
        # Display as clean styled DataFrame
        bench_df = pd.DataFrame(benchmarks).T
        bench_df = bench_df.reset_index().rename(columns={"index": "Model Architecture"})
        
        st.dataframe(bench_df, use_container_width=True)
        
        # Plot comparative metrics
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            fig_acc = px.bar(
                bench_df,
                x="Model Architecture",
                y="Accuracy",
                color="Model Architecture",
                title="Congestion Classification Accuracy (%)",
                text="Accuracy",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_acc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1e293b", font=dict(color="white"), showlegend=False)
            st.plotly_chart(fig_acc, use_container_width=True)
            
        with col_m2:
            fig_mae = px.bar(
                bench_df,
                x="Model Architecture",
                y="MAE",
                color="Model Architecture",
                title="Flight Volume Forecast MAE (Lower is Better)",
                text="MAE",
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
            fig_mae.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1e293b", font=dict(color="white"), showlegend=False)
            st.plotly_chart(fig_mae, use_container_width=True)
    else:
        st.info("ℹ️ Benchmark metrics file (`models/metrics_comparison.json`) will be generated after model training pipeline finishes.")

# Auto-Play Simulation loop
if st.session_state.auto_play:
    time.sleep(3)
    st.session_state.simulator.step_forward(1)
    st.rerun()
