"""
Battery Management System - Safety Interface
============================================
Real-time thermal runaway risk monitoring and voltage limit enforcement.

Streamlit web application for BMS integration.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from safety_optimizer import PhaseSafetyOptimizer

# Page configuration
st.set_page_config(
    page_title="BMS Safety Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-critical {
        background-color: #ff4b4b;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        text-align: center;
        font-size: 1.2rem;
    }
    .alert-safe {
        background-color: #00cc00;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        text-align: center;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize optimizer
@st.cache_resource
def load_optimizer():
    """Load safety optimizer (cached for performance)."""
    return PhaseSafetyOptimizer(model_dir='models')

try:
    optimizer = load_optimizer()
except FileNotFoundError as e:
    st.error(f"⚠️ {str(e)}")
    st.info("Please run `thermal_prediction/main.py` first to train and save the model.")
    st.stop()

# Header
st.markdown('<div class="main-header">Battery Management System - Safety Monitor</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar - BMS Inputs
st.sidebar.header("⚡ Live BMS Data Feed")
st.sidebar.markdown("---")

# Voltage Input - Slider + Number Input
st.sidebar.subheader("🔋 Live Voltage")
vol_col1, vol_col2 = st.sidebar.columns([2, 1])
with vol_col1:
    live_voltage_slider = st.slider(
        "Voltage (V)",
        min_value=2.5,
        max_value=4.5,
        value=4.2,
        step=0.01,
        key="voltage_slider",
        label_visibility="collapsed"
    )
with vol_col2:
    live_voltage_input = st.number_input(
        "V",
        min_value=2.5,
        max_value=4.5,
        value=live_voltage_slider,
        step=0.01,
        format="%.2f",
        key="voltage_input",
        label_visibility="collapsed"
    )

# Sync voltage values
live_voltage = live_voltage_input if live_voltage_input != live_voltage_slider else live_voltage_slider

st.sidebar.markdown("")  # Spacing

# Capacity Input - Slider + Number Input
st.sidebar.subheader("⚙️ SOH Capacity")
cap_col1, cap_col2 = st.sidebar.columns([2, 1])
with cap_col1:
    soh_capacity_slider = st.slider(
        "Capacity (Ah)",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.01,
        key="capacity_slider",
        label_visibility="collapsed"
    )
with cap_col2:
    soh_capacity_input = st.number_input(
        "Ah",
        min_value=1.0,
        max_value=5.0,
        value=soh_capacity_slider,
        step=0.01,
        format="%.2f",
        key="capacity_input",
        label_visibility="collapsed"
    )

# Sync capacity values
soh_capacity = soh_capacity_input if soh_capacity_input != soh_capacity_slider else soh_capacity_slider

st.sidebar.markdown("")  # Spacing

# Battery Phase Selection
st.sidebar.subheader("🏭 Battery Phase")
battery_phase = st.sidebar.selectbox(
    "Phase",
    options=["Phase 0 Prototype", "Phase 1 Production"],
    help="Battery development phase",
    label_visibility="collapsed"
)

st.sidebar.markdown("")  # Spacing

# Risk Threshold - Slider + Number Input
st.sidebar.subheader("⚠️ Risk Threshold")
risk_col1, risk_col2 = st.sidebar.columns([2, 1])
with risk_col1:
    max_risk_slider = st.slider(
        "Max Risk (kJ)",
        min_value=20.0,
        max_value=60.0,
        value=40.0,
        step=1.0,
        key="risk_slider",
        label_visibility="collapsed"
    )
with risk_col2:
    max_risk_input = st.number_input(
        "kJ",
        min_value=20.0,
        max_value=60.0,
        value=max_risk_slider,
        step=1.0,
        format="%.1f",
        key="risk_input",
        label_visibility="collapsed"
    )

# Sync risk threshold values
max_risk_threshold = max_risk_input if max_risk_input != max_risk_slider else max_risk_slider

st.sidebar.markdown("---")

# Phase Information
st.sidebar.subheader("📋 Phase Specifications")
phase_info = optimizer.get_phase_info(battery_phase)
for key, value in phase_info.items():
    st.sidebar.markdown(f"**{key}:** `{value}`")

# Calculate current state
stored_energy_wh = live_voltage * soh_capacity
predicted_risk_kj = optimizer.predict_risk(live_voltage, soh_capacity)
max_safe_voltage = optimizer.get_max_safe_voltage(soh_capacity, max_risk_threshold)
risk_level, risk_color = optimizer.get_risk_level(predicted_risk_kj)

# Voltage margin
voltage_margin = max_safe_voltage - live_voltage
is_safe = live_voltage <= max_safe_voltage

# Main Layout - 2 columns
col1, col2 = st.columns([1, 1])

# Column 1: Current State & Risk Gauge
with col1:
    st.subheader("📊 Current Battery State")
    
    # Metrics with larger display
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric(
            label="⚡ Voltage",
            value=f"{live_voltage:.2f} V",
            delta=f"{voltage_margin:+.2f} V margin" if abs(voltage_margin) > 0.01 else None,
            delta_color="normal" if is_safe else "inverse"
        )
    
    with metric_col2:
        st.metric(
            label="🔋 Capacity",
            value=f"{soh_capacity:.2f} Ah"
        )
    
    with metric_col3:
        st.metric(
            label="⚙️ Stored Energy",
            value=f"{stored_energy_wh:.2f} Wh"
        )
    
    # Risk Gauge
    st.markdown("#### Risk Gauge")
    
    # Create gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=predicted_risk_kj,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Thermal Runaway Risk (kJ)", 'font': {'size': 18}},
        delta={'reference': max_risk_threshold, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': risk_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#90EE90'},      # Light green
                {'range': [30, 50], 'color': '#FFFF99'},     # Light yellow
                {'range': [50, 70], 'color': '#FFB366'},     # Light orange
                {'range': [70, 100], 'color': '#FF6666'}     # Light red
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_risk_threshold
            }
        }
    ))
    
    fig_gauge.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=80, b=20),
        font={'size': 14}
    )
    
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Risk Level Badge
    st.markdown(f"**Risk Level:** `{risk_level}`")
    
    # Safety Alert
    st.markdown("#### Safety Status")
    
    if is_safe:
        st.markdown(f'<div class="alert-safe">✓ SAFE OPERATION<br>Voltage: {live_voltage:.2f}V ≤ Limit: {max_safe_voltage:.2f}V</div>', 
                   unsafe_allow_html=True)
    else:
        recommended_voltage = max_safe_voltage
        reduction_needed = live_voltage - max_safe_voltage
        
        st.markdown(f'<div class="alert-critical">⚠️ CRITICAL: DERATE CHARGE<br>Reduce Voltage by {reduction_needed:.2f}V</div>', 
                   unsafe_allow_html=True)
        
        st.warning(f"""
        **Immediate Action Required:**
        - Current: {live_voltage:.2f}V
        - Maximum Safe: {max_safe_voltage:.2f}V
        - **Recommended:** Reduce to {recommended_voltage:.2f}V or lower
        """)

# Column 2: Safety Boundary Map
with col2:
    st.subheader("Safety Operating Area")
    
    # Generate safety boundary
    capacities, max_voltages = optimizer.generate_safety_boundary(
        capacity_range=(1.0, 5.0),
        max_risk_kj=max_risk_threshold,
        num_points=100
    )
    
    # Create boundary plot
    fig_boundary = go.Figure()
    
    # Safe zone (below boundary)
    fig_boundary.add_trace(go.Scatter(
        x=capacities,
        y=max_voltages,
        fill='tozeroy',
        fillcolor='rgba(0, 255, 0, 0.2)',
        line=dict(color='green', width=3),
        name=f'Safe Zone (<{max_risk_threshold:.0f} kJ)',
        hovertemplate='Capacity: %{x:.2f} Ah<br>Max Safe V: %{y:.2f} V<extra></extra>'
    ))
    
    # Unsafe zone (above boundary)
    fig_boundary.add_trace(go.Scatter(
        x=capacities,
        y=[4.5] * len(capacities),
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(width=0),
        name=f'Unsafe Zone (>{max_risk_threshold:.0f} kJ)',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Current battery state
    fig_boundary.add_trace(go.Scatter(
        x=[soh_capacity],
        y=[live_voltage],
        mode='markers+text',
        marker=dict(
            size=20,
            color='red' if not is_safe else 'blue',
            symbol='diamond',
            line=dict(color='white', width=2)
        ),
        text=['Current State'],
        textposition='top center',
        textfont=dict(size=12, color='black', family='Arial Black'),
        name='Current Battery',
        hovertemplate='Capacity: %{x:.2f} Ah<br>Voltage: %{y:.2f} V<br>Risk: ' + f'{predicted_risk_kj:.1f} kJ<extra></extra>'
    ))
    
    fig_boundary.update_layout(
        xaxis_title="Capacity (Ah)",
        yaxis_title="Voltage (V)",
        hovermode='closest',
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        xaxis=dict(range=[0.8, 5.2]),
        yaxis=dict(range=[2.3, 4.6])
    )
    
    st.plotly_chart(fig_boundary, use_container_width=True)
    
    # Safety metrics table
    st.markdown("#### Safety Limits")
    
    safety_df = pd.DataFrame({
        'Parameter': ['Max Safe Voltage', 'Current Voltage', 'Margin', 'Risk Threshold', 'Predicted Risk'],
        'Value': [
            f'{max_safe_voltage:.2f} V',
            f'{live_voltage:.2f} V',
            f'{voltage_margin:+.2f} V',
            f'{max_risk_threshold:.1f} kJ',
            f'{predicted_risk_kj:.1f} kJ'
        ],
        'Status': [
            '✓ Limit',
            '⚠️ Critical' if not is_safe else '✓ Safe',
            '⚠️ Negative' if voltage_margin < 0 else '✓ Positive',
            '- Threshold',
            '⚠️ Exceeded' if predicted_risk_kj > max_risk_threshold else '✓ Below'
        ]
    })
    
    st.dataframe(safety_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
**BMS Safety Monitor** | Phase 0/1 Battery Thermal Runaway Risk Assessment  
*Hardcoded Parameters:* 18650 Format, 30µm Casing, Nail Trigger (Worst-case)  
*Model Performance:* R² = {:.1f}%, RMSE = {:.2f} kJ
""".format(optimizer.metadata['test_r2'] * 100, optimizer.metadata['test_rmse']))
