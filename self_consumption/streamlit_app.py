import streamlit as st
import numpy as np
import pandas as pd
import self_consumption_analysis as sca
import plotly.graph_objs as go

st.set_page_config(
    page_title="PV-Battery Self-Consumption Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("PV-Battery Self-Consumption Analysis")

# Sidebar - parameter selection
st.sidebar.header("Simulation Parameters")
pv_system_size = st.sidebar.slider("PV System Size (kWp)", min_value=0.5, max_value=10.0, value=1.8, step=0.1)
battery_capacity = st.sidebar.slider("Battery Capacity (kWh)", min_value=0.5, max_value=20.0, value=2.5, step=0.1)
model_error_low = st.sidebar.number_input("Model Error Low (kWh)", value=-0.5, step=0.01)
model_error_high = st.sidebar.number_input("Model Error High (kWh)", value=0.1, step=0.01)
period = st.sidebar.selectbox("Simulation Period", ["Single Day", "One Month", "One Year"])

# Load H0 profile data
h0_profile = pd.read_csv("model_evaluation_aws/h0_profile_15min_daily_avg.csv", header=None).values.flatten()
intervals_per_day = 96
h0_profile_kW = h0_profile / 1000
h0_profile_kWh = h0_profile_kW * 0.25
h0_daily_kWh = np.sum(h0_profile_kWh)
target_annual_load_kWh = 3500
scaling_factor = target_annual_load_kWh / (h0_daily_kWh * 365)
h0_profile_kWh_scaled = h0_profile_kWh * scaling_factor

def run_simulation(period, pv_system_size, battery_capacity, error_low, error_high):
    """
    Runs both reference and model simulations with correct error handling.
    Reference: always no error (model_error=0, error_low=0, error_high=0)
    Model: error applied if model_error_low/high are set (model_error=max(|low|,|high|,1e-6)).
    This avoids confusion and ensures reference/model are not identical unless intended.
    """
    if period == "Single Day":
        h0 = h0_profile_kWh_scaled
        pv_day = sca.typical_pv_day_profile(kWp=pv_system_size)
        # Reference: no error
        ref_result = sca.simulate_day(h0, pv_day, battery_capacity, error_low=0, error_high=0)
        # Model: user-specified error
        model_result = sca.simulate_day(h0, pv_day, battery_capacity, error_low=error_low, error_high=error_high)
        ref_summary = sca.summarize_result(ref_result, pv_day, h0)
        model_summary = sca.summarize_result(model_result, pv_day, h0)
        return h0, pv_day, ref_result, model_result, ref_summary, model_summary
    elif period == "One Month":
        h0 = np.tile(h0_profile_kWh_scaled, 31)
        pv_month = np.tile(sca.typical_pv_day_profile(kWp=pv_system_size), 31)
        ref_result = sca.simulate_day(h0, pv_month, battery_capacity, error_low=0, error_high=0)
        model_result = sca.simulate_day(h0, pv_month, battery_capacity, error_low=error_low, error_high=error_high)
        ref_summary = sca.summarize_result(ref_result, pv_month, h0)
        model_summary = sca.summarize_result(model_result, pv_month, h0)
        return h0, pv_month, ref_result, model_result, ref_summary, model_summary
    elif period == "One Year":
        h0 = np.tile(h0_profile_kWh_scaled, 365)
        def seasonal_pv_scale(day):
            return 0.5 + 0.5 * np.sin(2 * np.pi * (day - 80) / 365)
        pv_year = []
        for day in range(365):
            pv_scale = seasonal_pv_scale(day)
            pv_day = sca.typical_pv_day_profile(kWp=pv_system_size) * pv_scale
            pv_year.append(pv_day)
        pv_year = np.concatenate(pv_year)
        ref_result = sca.simulate_day(h0, pv_year, battery_capacity, error_low=0, error_high=0)
        model_result = sca.simulate_day(h0, pv_year, battery_capacity, error_low=error_low, error_high=error_high)
        ref_summary = sca.summarize_result(ref_result, pv_year, h0)
        model_summary = sca.summarize_result(model_result, pv_year, h0)
        return h0, pv_year, ref_result, model_result, ref_summary, model_summary

# Run simulation
h0, pv, ref_result, model_result, ref_summary, model_summary = run_simulation(period, pv_system_size, battery_capacity, model_error_low, model_error_high)

# Display results
st.subheader(f"Results for {period}")


# --- Metrics Display Refactor ---
def metric_with_emoji(label, value, emoji, unit=""):
    st.markdown(f"""
    <div style="
        padding: 0.5rem;
        border-radius: 0.5rem;
        background: #f8f9fa;
        margin-bottom: 0.5rem;
        border-left: 4px solid #e9ecef;
    ">
        <div style="
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.9rem;
            color: #495057;
        ">
            <span>{emoji}</span>
            <span>{label}</span>
        </div>
        <div style="
            font-size: 1.2rem;
            font-weight: 600;
            margin-top: 0.25rem;
            color: #212529;
        ">
            {value}{unit}
        </div>
    </div>
    """, unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("<h4>Reference Meter</h4>", unsafe_allow_html=True)
    metric_with_emoji("Total PV", f"{ref_summary['total_pv']:.2f}", "☀️", " kWh")
    metric_with_emoji("Total Load", f"{ref_summary['total_load']:.2f}", "🏠", " kWh")
    metric_with_emoji("Self-consumed", f"{ref_summary['self_consumed']:.2f}", "🍃", " kWh")
    metric_with_emoji("SSR", f"{ref_summary['ssr']:.1%}", "📊")
    metric_with_emoji("SCR", f"{ref_summary['scr']:.1%}", "📈")

with col2:
    st.markdown(f"<h4>Model<br><span style='font-size:0.7em;'>Error: [{model_error_low}, {model_error_high}]</span></h4>", unsafe_allow_html=True)
    metric_with_emoji("Total PV", f"{model_summary['total_pv']:.2f}", "☀️", " kWh")
    metric_with_emoji("Total Load", f"{model_summary['total_load']:.2f}", "🏠", " kWh")
    metric_with_emoji("Self-consumed", f"{model_summary['self_consumed']:.2f}", "🍃", " kWh")
    metric_with_emoji("SSR", f"{model_summary['ssr']:.1%}", "📊")
    metric_with_emoji("SCR", f"{model_summary['scr']:.1%}", "📈")


if period == "Single Day":
    # --- Power Flow Plot (without SOC) ---
    x = list(range(len(pv)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=pv, mode='lines', name='PV Generation', line=dict(color='gold')))
    fig.add_trace(go.Scatter(x=x, y=h0, mode='lines', name='Household Load', line=dict(color='blue')))
    fig.update_layout(title="Single Day Power Flow", xaxis_title="15-min Interval", yaxis_title="kWh", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # --- SOC Comparison Plot ---
    def plot_soc_comparison(ref_soc, model_soc, title="Battery State of Charge"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=ref_soc, mode='lines', name='SOC (Ref)', line=dict(color='green')))
        fig.add_trace(go.Scatter(y=model_soc, mode='lines', name='SOC (Model)', line=dict(color='red', dash='dash')))
        fig.update_layout(title=title, xaxis_title='Time', yaxis_title='State of Charge (kWh)', showlegend=True, margin=dict(l=20, r=20, t=40, b=20))
        return fig
    st.plotly_chart(
        plot_soc_comparison(ref_result["soc_hist"], model_result["soc_hist"], "Battery State of Charge Comparison"),
        use_container_width=True
    )

elif period == "One Month":
    days = list(range(1, 32))
    pv_daily = np.add.reduceat(pv, np.arange(0, len(pv), 96))
    ref_self = np.add.reduceat(np.array(ref_result["direct_consumption"]) + np.array(ref_result["battery_discharge"]), np.arange(0, len(pv), 96))
    model_self = np.add.reduceat(np.array(model_result["direct_consumption"]) + np.array(model_result["battery_discharge"]), np.arange(0, len(pv), 96))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=days, y=pv_daily, name='PV Generation', marker_color='gold'))
    fig.add_trace(go.Bar(x=days, y=ref_self, name='Self-consumed PV (Ref)', marker_color='green'))
    fig.add_trace(go.Bar(x=days, y=model_self, name='Self-consumed PV (Model)', marker_color='red'))
    fig.update_layout(barmode='group', title="One Month Daily Energy Flows", xaxis_title="Day", yaxis_title="kWh")
    st.plotly_chart(fig, use_container_width=True)

elif period == "One Year":
    days = list(range(1, 366))
    pv_daily = np.add.reduceat(pv, np.arange(0, len(pv), 96))
    ref_self = np.add.reduceat(np.array(ref_result["direct_consumption"]) + np.array(ref_result["battery_discharge"]), np.arange(0, len(pv), 96))
    model_self = np.add.reduceat(np.array(model_result["direct_consumption"]) + np.array(model_result["battery_discharge"]), np.arange(0, len(pv), 96))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=pv_daily, mode='lines', name='PV Generation', line=dict(color='gold')))
    fig.add_trace(go.Scatter(x=days, y=ref_self, mode='lines', name='Self-consumed PV (Ref)', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=days, y=model_self, mode='lines', name='Self-consumed PV (Model)', line=dict(color='red', dash='dash')))
    fig.update_layout(title="One Year Daily Energy Flows", xaxis_title="Day", yaxis_title="kWh")
    st.plotly_chart(fig, use_container_width=True)


st.info("Use the sidebar to adjust parameters and rerun the simulation interactively!")
