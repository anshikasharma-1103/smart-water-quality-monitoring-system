"""
Smart Water Quality Monitoring System
Tagline: "Monitoring Today, Protecting Tomorrow."

Main Streamlit Application Entrypoint
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import datetime
import json
import os

# Import modular project packages
from water_quality import (
    DEFAULT_REFERENCE_RANGES,
    LEGAL_DISCLAIMER,
    validate_reading,
    evaluate_parameter_status,
    calculate_water_quality_index
)
from sensor_reliability import (
    assess_all_sensors,
    check_sensor_health
)
from trend_detection import (
    analyze_all_trends,
    analyze_parameter_trend
)
from alert_engine import (
    generate_explainable_alerts
)
from data_service import (
    BASELINE_VALUES,
    generate_single_reading,
    generate_initial_history,
    inject_pollution_spike,
    inject_faulty_sensor,
    validate_csv_upload,
    fetch_thingspeak_data
)
from train_model import (
    load_or_train_model,
    predict_water_quality,
    train_and_save_model
)

# Page configuration
st.set_page_config(
    page_title="Smart Water Quality Monitoring System",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark navy blue, water blue, clean slate, subtle green)
st.markdown("""
<style>
    /* Dark Navy & Water Blue Theme */
    :root {
        --primary: #0284c7;
        --primary-dark: #0b192c;
        --secondary: #0ea5e9;
        --accent-emerald: #10b981;
        --bg-subtle: #f8fafc;
        --text-dark: #0f172a;
    }
    
    .main {
        background-color: #f8fafc;
    }
    
    .stMetric {
        background: white;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .app-header {
        background: linear-gradient(135deg, #0b192c 0%, #1e3a8a 50%, #0369a1 100%);
        color: white;
        padding: 24px 28px;
        border-radius: 14px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(11, 25, 44, 0.15);
    }
    
    .app-tagline {
        color: #93c5fd;
        font-size: 14px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        font-weight: 600;
        margin-top: 4px;
    }
    
    .disclaimer-banner {
        background-color: #eff6ff;
        border-left: 4px solid #0284c7;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 13px;
        color: #1e3a8a;
        margin-bottom: 20px;
    }

    .alert-card {
        padding: 14px 18px;
        border-radius: 10px;
        margin-bottom: 12px;
        border-left: 5px solid;
    }
    
    .alert-critical {
        background: #fef2f2;
        border-color: #ef4444;
        color: #991b1b;
    }
    
    .alert-warning {
        background: #fffbeb;
        border-color: #f59e0b;
        color: #92400e;
    }
    
    .alert-watch {
        background: #f0fdf4;
        border-color: #10b981;
        color: #065f46;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = generate_initial_history(20)

if "active_alerts" not in st.session_state:
    st.session_state.active_alerts = []

if "calibration_refs" not in st.session_state:
    st.session_state.calibration_refs = {
        "ph": 7.00,
        "turbidity": 0.5,
        "tds": 150.0,
        "conductivity": 250.0,
        "temperature": 20.0
    }

if "reference_ranges" not in st.session_state:
    st.session_state.reference_ranges = DEFAULT_REFERENCE_RANGES.copy()

if "stream_running" not in st.session_state:
    st.session_state.stream_running = False

if "last_seen_times" not in st.session_state:
    now_epoch = time.time()
    st.session_state.last_seen_times = {k: now_epoch for k in DEFAULT_REFERENCE_RANGES}


# Top Navigation Sidebar
st.sidebar.markdown("""
<div style="display:flex; align-items:center; gap:10px; padding: 10px 0;">
    <span style="font-size: 28px;">💧</span>
    <div>
        <h3 style="margin:0; font-size: 17px; font-weight:700; color:#0b192c;">Smart Water AI</h3>
        <p style="margin:0; font-size: 11px; color:#64748b;">Monitoring Today, Protecting Tomorrow</p>
    </div>
</div>
""", unsafe_allow_html=True)

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "1. Overview",
        "2. Live Water Monitoring",
        "3. AI Prediction",
        "4. Sensor Reliability",
        "5. Early-Warning Intelligence",
        "6. Alert Center",
        "7. Historical Analytics",
        "8. System Settings and About"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎮 Demo Mode Controls")
st.sidebar.caption("Simulate real-time parameters without hardware.")

col_sim1, col_sim2 = st.sidebar.columns(2)
with col_sim1:
    if st.button("➕ Next Sample", use_container_width=True):
        latest = st.session_state.history[-1] if st.session_state.history else None
        new_row = generate_single_reading(latest)
        st.session_state.history.append(new_row)
        now_ep = time.time()
        for k in DEFAULT_REFERENCE_RANGES:
            st.session_state.last_seen_times[k] = now_ep
        st.rerun()

with col_sim2:
    if st.button("🔄 Reset Data", use_container_width=True):
        st.session_state.history = generate_initial_history(20)
        st.session_state.active_alerts = []
        st.rerun()

if st.sidebar.button("⚠️ Simulate Pollution Spike", use_container_width=True):
    latest = st.session_state.history[-1] if st.session_state.history else None
    spike_row = inject_pollution_spike(latest)
    st.session_state.history.append(spike_row)
    now_ep = time.time()
    for k in DEFAULT_REFERENCE_RANGES:
        st.session_state.last_seen_times[k] = now_ep
    st.sidebar.success("Injected sudden industrial effluent spike!")
    st.rerun()

fault_type = st.sidebar.selectbox("Fault Type:", ["freeze", "jump", "noise"])
if st.sidebar.button("🛠️ Simulate Faulty Sensor", use_container_width=True):
    latest = st.session_state.history[-1] if st.session_state.history else None
    fault_row = inject_faulty_sensor(latest, fault_type=fault_type)
    st.session_state.history.append(fault_row)
    now_ep = time.time()
    for k in DEFAULT_REFERENCE_RANGES:
        st.session_state.last_seen_times[k] = now_ep
    st.sidebar.warning(f"Simulated sensor hardware fault: {fault_type.upper()}")
    st.rerun()

st.sidebar.markdown("---")
current_source = st.session_state.history[-1].get("source", "Demo (Simulated)") if st.session_state.history else "None"
st.sidebar.info(f"**Data Stream Source:**\n{current_source}")


# Compute Latest State
latest_reading = st.session_state.history[-1] if st.session_state.history else BASELINE_VALUES
sensor_health_map = assess_all_sensors(
    st.session_state.history,
    calibration_refs=st.session_state.calibration_refs,
    last_seen_times=st.session_state.last_seen_times
)
trend_map = analyze_all_trends(st.session_state.history, ref_ranges=st.session_state.reference_ranges)
wqi_res = calculate_water_quality_index(latest_reading, ranges=st.session_state.reference_ranges)

# Update alerts
new_alerts = generate_explainable_alerts(
    latest_reading,
    sensor_health_map,
    trend_map,
    ref_ranges=st.session_state.reference_ranges,
    existing_alerts=st.session_state.active_alerts
)
for a in new_alerts:
    # avoid duplicates
    if not any(x["alert_id"] == a["alert_id"] for x in st.session_state.active_alerts):
        st.session_state.active_alerts.insert(0, a)


# ==============================================================================
# SECTION 1: OVERVIEW
# ==============================================================================
if nav_choice == "1. Overview":
    st.markdown("""
    <div class="app-header">
        <h1 style="margin:0; font-size:28px; font-weight:800;">Smart Water Quality Monitoring System</h1>
        <div class="app-tagline">Monitoring Today, Protecting Tomorrow · AI & IoT Environmental Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="disclaimer-banner">⚖️ <strong>Regulatory Note:</strong> {LEGAL_DISCLAIMER}</div>', unsafe_allow_html=True)

    # Top KPI Row
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.metric(
            label="Water Quality Index (WQI)",
            value=f"{wqi_res['wqi']} / 100",
            delta=wqi_res["overall_status"]
        )
    with col_kpi2:
        st.metric(
            label="Abnormal Parameters",
            value=f"{wqi_res['abnormal_count']} of 5",
            delta="Normal" if wqi_res["abnormal_count"] == 0 else "Caution / Breach",
            delta_color="inverse"
        )
    with col_kpi3:
        unack = sum(1 for a in st.session_state.active_alerts if not a.get("acknowledged", False))
        st.metric(
            label="Active Unacknowledged Alerts",
            value=str(unack),
            delta="Clear" if unack == 0 else "Requires Inspection",
            delta_color="inverse"
        )
    with col_kpi4:
        st.metric(
            label="Latest Sample Timestamp",
            value=latest_reading.get("timestamp", "N/A").split(" ")[-1],
            delta=latest_reading.get("source", "Demo")
        )

    st.markdown("### 💧 Real-Time Parameter Telemetry")
    pcols = st.columns(5)
    params_keys = ["ph", "turbidity", "tds", "conductivity", "temperature"]
    for i, p in enumerate(params_keys):
        cfg = st.session_state.reference_ranges[p]
        val = latest_reading.get(p, 0.0)
        status_info = evaluate_parameter_status(p, val, st.session_state.reference_ranges)
        with pcols[i]:
            st.metric(
                label=f"{cfg['name']} ({cfg['unit']})",
                value=f"{val:.2f}",
                delta=status_info["status"],
                delta_color="normal" if status_info["status"] == "Safe" else "inverse"
            )
            st.caption(f"Safe: {cfg['min_safe']} - {cfg['max_safe']} {cfg['unit']}")

    st.markdown("---")
    col_chart, col_side = st.columns([2, 1])

    with col_chart:
        st.markdown("#### 📈 Multi-Parameter Trend Evolution")
        df_hist = pd.DataFrame(st.session_state.history)
        if not df_hist.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_hist["timestamp"], y=df_hist["ph"], mode="lines+markers", name="pH", line=dict(color="#0284c7")))
            fig.add_trace(go.Scatter(x=df_hist["timestamp"], y=df_hist["turbidity"], mode="lines+markers", name="Turbidity (NTU)", line=dict(color="#eab308")))
            fig.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                height=320,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis_title="Timestamp",
                yaxis_title="Reading"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_side:
        st.markdown("#### 🛡️ Hardware & Early Warning Pulse")
        unhealthy_sensors = [k for k, v in sensor_health_map.items() if v["status"] != "Healthy"]
        if not unhealthy_sensors:
            st.success("✅ All 5 sensor probes report **Healthy** electro-chemical diagnostics.")
        else:
            st.error(f"⚠️ {len(unhealthy_sensors)} sensor(s) require maintenance: {', '.join(unhealthy_sensors).upper()}")

        st.markdown("##### Recent Alert Dispatches")
        if not st.session_state.active_alerts:
            st.info("No active alerts logged. Water stream within standard nominal baseline.")
        else:
            for alert in st.session_state.active_alerts[:3]:
                sev_cls = "alert-critical" if alert["severity"] == "Critical" else ("alert-warning" if alert["severity"] == "Warning" else "alert-watch")
                st.markdown(f"""
                <div class="alert-card {sev_cls}">
                    <strong>[{alert['severity']}] {alert['param_name']}</strong> · {alert['timestamp']}<br>
                    <small>{alert['reason']}</small>
                </div>
                """, unsafe_allow_html=True)


# ==============================================================================
# SECTION 2: LIVE WATER MONITORING
# ==============================================================================
elif nav_choice == "2. Live Water Monitoring":
    st.markdown("## 🌊 Live Water Monitoring Console")
    st.caption("Detailed physical diagnostics, configurable bounds, and chemical context for monitored parameters.")
    st.markdown(f'<div class="disclaimer-banner">{LEGAL_DISCLAIMER}</div>', unsafe_allow_html=True)

    for p in ["ph", "turbidity", "tds", "conductivity", "temperature"]:
        cfg = st.session_state.reference_ranges[p]
        val = latest_reading.get(p, 0.0)
        status_info = evaluate_parameter_status(p, val, st.session_state.reference_ranges)
        trend_info = trend_map.get(p, {})
        sensor_info = sensor_health_map.get(p, {})

        with st.expander(f"💧 {cfg['name']} — {val:.2f} {cfg['unit']} [{status_info['status']}]", expanded=True):
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                st.write("**Observed Value:**")
                st.markdown(f"### {val:.2f} {cfg['unit']}")
                st.write(f"Safe Range: `{cfg['min_safe']} - {cfg['max_safe']} {cfg['unit']}`")
                st.write(f"Caution Range: `{cfg['min_caution']} - {cfg['max_caution']} {cfg['unit']}`")

            with c2:
                st.write("**Diagnostics & Trend:**")
                st.write(f"Trend Direction: **{trend_info.get('direction', 'Stable')}**")
                st.write(f"Z-Score: `{trend_info.get('z_score', 0.0):+.2f}`")
                st.write(f"Sensor Status: **{sensor_info.get('status', 'Healthy')}**")

            with c3:
                st.write("**Scientific Significance:**")
                st.write(cfg["description"])
                st.info(f"**Health & Ecological Impact:** {cfg['significance']}")


# ==============================================================================
# SECTION 3: AI PREDICTION
# ==============================================================================
elif nav_choice == "3. AI Prediction":
    st.markdown("## 🧠 AI-Based Water Quality Classification")
    st.caption("Random Forest Classifier trained on multi-parameter environmental profiles with explainable factor decomposition.")
    st.markdown(f'<div class="disclaimer-banner">🔬 {LEGAL_DISCLAIMER}</div>', unsafe_allow_html=True)

    tab_pred, tab_eval, tab_manual = st.tabs(["⚡ Live Stream Inference", "📊 Model Evaluation & Metrics", "🧪 Manual Sample Testing"])

    with tab_pred:
        pred_res = predict_water_quality(latest_reading)
        
        c_left, c_right = st.columns([1, 1])
        with c_left:
            st.markdown("### Model Classification Result")
            pred_class = pred_res.get("prediction", "Unknown")
            if pred_class == "Safe-range indication":
                st.success(f"### ✅ {pred_class}\nConfidence: **{pred_res['confidence_pct']}%**")
            elif pred_class == "Caution":
                st.warning(f"### ⚠️ {pred_class}\nConfidence: **{pred_res['confidence_pct']}%**")
            else:
                st.error(f"### 🚨 {pred_class}\nConfidence: **{pred_res['confidence_pct']}%**")

            st.write("**Class Probability Distribution:**")
            probs = pred_res.get("class_probabilities", {})
            for cls_name, prob_val in probs.items():
                st.progress(prob_val / 100.0, text=f"{cls_name}: {prob_val}%")

        with c_right:
            st.markdown("### Explainable Causal Attribution")
            factors = pred_res.get("contributing_factors", [])
            if not factors:
                st.info("All parameters are within nominal bounds. No single feature is driving an anomaly alert.")
            else:
                for f in factors:
                    st.error(f"**{f['feature']}**: {f['value']} {f['unit']} — {f['status']} (Ref: {f['reference_range']})")

            st.markdown("#### Recommended Next Actions")
            for rec in pred_res.get("recommendations", []):
                st.write(f"- {rec}")

    with tab_eval:
        st.markdown("### 📈 Model Performance on Demonstration Holdout Data")
        model_obj = load_or_train_model()
        metrics = model_obj["metrics"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
        m2.metric("Precision", f"{metrics['precision']*100:.1f}%")
        m3.metric("Recall", f"{metrics['recall']*100:.1f}%")
        m4.metric("F1-Score", f"{metrics['f1_score']*100:.1f}%")

        st.caption("Evaluation performed with 75/25 stratified holdout split on 1,600 reproducible synthetic samples (random_state=42).")

        c_imp, c_cm = st.columns([1, 1])
        with c_imp:
            st.markdown("#### Feature Importance Weights")
            imp_df = pd.DataFrame(list(metrics["feature_importances"].items()), columns=["Feature", "Importance"])
            imp_df = imp_df.sort_values(by="Importance", ascending=True)
            fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h", title="Random Forest Gini Importance")
            fig_imp.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_imp, use_container_width=True)

        with c_cm:
            st.markdown("#### Test Confusion Matrix")
            cm_arr = np.array(metrics["confusion_matrix"])
            fig_cm = px.imshow(
                cm_arr,
                x=["Safe", "Caution", "Unsafe"],
                y=["Safe", "Caution", "Unsafe"],
                text_auto=True,
                color_continuous_scale="Blues",
                title="Confusion Matrix (Ground Truth vs Predicted)"
            )
            fig_cm.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)

        if st.button("🔄 Retrain Model Artifact", use_container_width=True):
            train_and_save_model()
            st.success("Model retrained and saved to water_model.joblib!")
            st.rerun()

    with tab_manual:
        st.markdown("### 🧪 Interactive Sample Simulator")
        st.caption("Input custom physical readings to test classifier boundaries in real time.")
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            s_ph = st.slider("pH Level", 0.0, 14.0, 7.2, 0.1)
            s_turb = st.slider("Turbidity (NTU)", 0.0, 100.0, 1.5, 0.5)
            s_tds = st.slider("Total Dissolved Solids (ppm)", 0.0, 2500.0, 220.0, 10.0)
        with m_col2:
            s_ec = st.slider("Conductivity (µS/cm)", 0.0, 5000.0, 360.0, 20.0)
            s_temp = st.slider("Water Temperature (°C)", 0.0, 50.0, 22.0, 0.5)

        test_sample = {"ph": s_ph, "turbidity": s_turb, "tds": s_tds, "conductivity": s_ec, "temperature": s_temp}
        manual_res = predict_water_quality(test_sample)
        st.markdown("#### Test Prediction:")
        st.write(f"Class: **{manual_res['prediction']}** (Confidence: {manual_res['confidence_pct']}%)")
        if manual_res.get("contributing_factors"):
            for f in manual_res["contributing_factors"]:
                st.caption(f"• {f['feature']}: {f['status']}")


# ==============================================================================
# SECTION 4: SENSOR RELIABILITY INTELLIGENCE
# ==============================================================================
elif nav_choice == "4. Sensor Reliability":
    st.markdown("## 🛡️ Unique Feature A: Sensor Reliability Intelligence")
    st.caption("Detects hardware failure modes: frozen ADC output, missing values, noise variance, sudden jumps, and calibration drift.")
    st.markdown("""
    <div class="disclaimer-banner">
    ⚠️ <strong>Metrological Notice:</strong> Automated telemetry checks identify erratic electrical and statistical behaviors,
    but cannot conclusively certify physical sensor precision without periodic multi-point laboratory calibration standards.
    </div>
    """, unsafe_allow_html=True)

    s_cols = st.columns(5)
    params_keys = ["ph", "turbidity", "tds", "conductivity", "temperature"]
    for i, p in enumerate(params_keys):
        info = sensor_health_map.get(p, {})
        cfg = st.session_state.reference_ranges[p]
        with s_cols[i]:
            status_emoji = "🟢" if info["status"] == "Healthy" else ("🟡" if info["status"] == "Warning" else "🔴")
            st.metric(
                label=f"{cfg['name']}",
                value=f"{info['score']} / 100",
                delta=f"{status_emoji} {info['status']}"
            )

    st.markdown("### 📋 Sensor Diagnostic Breakdown")
    for p in params_keys:
        info = sensor_health_map.get(p, {})
        cfg = st.session_state.reference_ranges[p]
        with st.expander(f"{cfg['name']} Health Diagnostics — [{info['status']}] Score: {info['score']}/100", expanded=info["status"] != "Healthy"):
            c_d1, c_d2 = st.columns([1, 1])
            with c_d1:
                st.write("**Operational Issues Detected:**")
                if not info["issues"]:
                    st.success("No anomalies detected. Signal variance, continuity, and response velocity are nominal.")
                else:
                    for iss in info["issues"]:
                        st.error(f"• {iss}")
            with c_d2:
                st.write("**Prescribed Corrective Action:**")
                st.info(info["corrective_action"])

    st.markdown("---")
    st.markdown("### 🎯 User Calibration Reference Alignment")
    st.caption("Enter known certified buffer readings to check real-time sensor offset and electro-chemical drift.")
    
    cal_cols = st.columns(5)
    for i, p in enumerate(params_keys):
        cfg = st.session_state.reference_ranges[p]
        with cal_cols[i]:
            curr_ref = st.session_state.calibration_refs.get(p, cfg["ideal"])
            new_val = st.number_input(f"Ref {cfg['name']} ({cfg['unit']})", value=float(curr_ref), step=0.1, key=f"cal_{p}")
            st.session_state.calibration_refs[p] = new_val

    st.success("Calibration standards updated in session memory.")


# ==============================================================================
# SECTION 5: EARLY-WARNING INTELLIGENCE
# ==============================================================================
elif nav_choice == "5. Early-Warning Intelligence":
    st.markdown("## ⚡ Unique Feature B: Trend-Based Early Warning")
    st.caption("Detects emerging environmental shifts via rolling Z-scores, multi-point persistence filtering, and rate-of-worsening velocity.")
    st.markdown("""
    <div class="disclaimer-banner">
    📊 <strong>Analytical Clarification:</strong> A statistically unusual reading (high rolling Z-score) indicates divergence from the moving baseline;
    it does NOT automatically prove chemical or biological contamination.
    </div>
    """, unsafe_allow_html=True)

    t_cols = st.columns(5)
    params_keys = ["ph", "turbidity", "tds", "conductivity", "temperature"]
    for i, p in enumerate(params_keys):
        t_info = trend_map.get(p, {})
        cfg = st.session_state.reference_ranges[p]
        with t_cols[i]:
            state = t_info.get("state", "Normal")
            emoji = "🟢" if state == "Normal" else ("🟡" if state == "Watch" else ("🟠" if state == "Warning" else "🔴"))
            st.metric(
                label=f"{cfg['name']}",
                value=f"{t_info.get('latest_value', 0.0):.2f}",
                delta=f"{emoji} {state} ({t_info.get('direction', 'Stable')})"
            )

    st.markdown("### 🔬 Multi-Point Rolling Trend Analytics")
    for p in params_keys:
        t_info = trend_map.get(p, {})
        cfg = st.session_state.reference_ranges[p]
        with st.expander(f"{cfg['name']} — State: [{t_info.get('state', 'Normal')}] Rate: {t_info.get('rate_of_change_pct', 0.0):+.1f}%", expanded=t_info.get("state") != "Normal"):
            c1, c2 = st.columns([1, 1])
            with c1:
                st.write(f"**Baseline Mean:** `{t_info.get('baseline_mean', 0.0):.2f} {cfg['unit']}`")
                st.write(f"**Rolling Z-Score:** `{t_info.get('z_score', 0.0):+.2f}`")
                st.write(f"**Persistent Deviations:** `{t_info.get('persistent_count', 0)} consecutive cycles`")
                st.write(f"**Condition Initiated:** `{t_info.get('condition_start_time', 'N/A')}`")
            with c2:
                st.write(f"**Trend Explanation:** {t_info.get('explanation', '')}")
                st.info(f"**Prescribed Verification:** {t_info.get('recommendation', '')}")


# ==============================================================================
# SECTION 6: ALERT CENTER
# ==============================================================================
elif nav_choice == "6. Alert Center":
    st.markdown("## 🚨 Unique Feature C: Explainable Alert Engine")
    st.caption("Actionable alert feed with hardware reliability cross-checking and operator acknowledgment workflows.")
    
    sev_filter = st.selectbox("Filter by Severity:", ["All", "Critical", "Warning", "Watch"])
    
    filtered_alerts = [
        a for a in st.session_state.active_alerts
        if sev_filter == "All" or a["severity"] == sev_filter
    ]

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("✅ Acknowledge All", use_container_width=True):
            for a in st.session_state.active_alerts:
                a["acknowledged"] = True
            st.success("All alerts acknowledged.")
            st.rerun()

    if not filtered_alerts:
        st.info("No active alerts matching the selected filter.")
    else:
        for idx, alert in enumerate(filtered_alerts):
            sev_class = "alert-critical" if alert["severity"] == "Critical" else ("alert-warning" if alert["severity"] == "Warning" else "alert-watch")
            ack_tag = " [ACKNOWLEDGED]" if alert.get("acknowledged", False) else ""
            with st.container():
                st.markdown(f"""
                <div class="alert-card {sev_class}">
                    <div style="display:flex; justify-content:space-between;">
                        <strong>[{alert['severity']}] {alert['param_name']} breach{ack_tag}</strong>
                        <span>{alert['timestamp']}</span>
                    </div>
                    <p style="margin: 6px 0;"><strong>Observed Reading:</strong> {alert['observed_value']} · <em>{alert['threshold_info']}</em></p>
                    <p style="margin: 6px 0;"><strong>Reason:</strong> {alert['reason']}</p>
                    <p style="margin: 6px 0; color: {'#991b1b' if alert['is_sensor_unreliable'] else '#15803d'};">
                        <strong>Hardware Status:</strong> {alert['sensor_status_note']}
                    </p>
                    <p style="margin: 6px 0;"><strong>Action Required:</strong> {alert['recommended_action']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if not alert.get("acknowledged", False):
                    if st.button(f"Mark as Acknowledged #{alert['alert_id']}", key=f"ack_{alert['alert_id']}"):
                        alert["acknowledged"] = True
                        st.rerun()


# ==============================================================================
# SECTION 7: HISTORICAL ANALYTICS
# ==============================================================================
elif nav_choice == "7. Historical Analytics":
    st.markdown("## 📊 Historical Data Analytics & CSV Exchange")
    st.caption("Inspect recorded telemetry time series, export CSV logs, or upload field measurements.")

    df_hist = pd.DataFrame(st.session_state.history)

    # Export CSV
    csv_data = df_hist.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download History CSV",
        data=csv_data,
        file_name="water_quality_history.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.markdown("### 📤 Upload Field CSV Data")
    uploaded_file = st.file_uploader("Upload CSV file (Requires columns: ph, turbidity, tds, conductivity, temperature)", type=["csv"])
    if uploaded_file is not None:
        try:
            up_df = pd.read_csv(uploaded_file)
            is_valid, msg, clean_df = validate_csv_upload(up_df)
            if is_valid:
                st.success(f"CSV Validation Passed! {msg}")
                if st.button("Load Uploaded Dataset into Current Session"):
                    st.session_state.history = clean_df.to_dict(orient="records")
                    st.rerun()
            else:
                st.error(f"CSV Validation Error: {msg}")
        except Exception as e:
            st.error(f"Error parsing uploaded file: {str(e)}")

    st.markdown("---")
    st.markdown("### 📋 Session Telemetry Records")
    st.dataframe(df_hist, use_container_width=True)


# ==============================================================================
# SECTION 8: SYSTEM SETTINGS AND ABOUT
# ==============================================================================
elif nav_choice == "8. System Settings and About":
    st.markdown("## ⚙️ System Settings & Project Innovation")
    st.caption("Configurable regulatory limits, cloud credentials, IoT architecture, and ideathon novelties.")

    tab_arch, tab_iot, tab_limits, tab_about = st.tabs([
        "🏗️ System Architecture",
        "📡 IoT & Cloud Integration",
        "🎚️ Threshold Settings",
        "💡 Competition Novelty"
    ])

    with tab_arch:
        st.markdown("""
        ### End-to-End System Workflow
        
        ```text
        +-----------------------------------------------------------------------+
        |                            WATER SOURCE                               |
        |             (River, Borewell, Storage Tank, RO Output)                |
        +-----------------------------------------------------------------------+
                                           |
                                           v
        +-----------------------------------------------------------------------+
        |                         PHYSICAL SENSORS                              |
        |  1. Analog pH Probe (0-14 pH)                                         |
        |  2. Turbidity Sensor (Optical Transmittance, 0-1000 NTU)              |
        |  3. TDS Sensor (Total Dissolved Solids, ppm)                          |
        |  4. Electrical Conductivity Probe (µS/cm)                             |
        |  5. DS18B20 Waterproof Temperature Probe (°C)                         |
        +-----------------------------------------------------------------------+
                                           |
                                           v
        +-----------------------------------------------------------------------+
        |                 MICROCONTROLLER / GATEWAY (ESP32)                     |
        |  • Reads ADC & 1-Wire channels with multi-sample averaging            |
        |  • Packages JSON payload & sends via HTTP POST or MQTT over Wi-Fi     |
        +-----------------------------------------------------------------------+
                                           |
                                           v
        +-----------------------------------------------------------------------+
        |                   DATA SANITIZATION & VALIDATION                      |
        |  • Strips null/NaN, parses numeric values                             |
        |  • Checks absolute physical feasibility limits                        |
        +-----------------------------------------------------------------------+
                                           |
                                           v
        +-----------------------------------------------------------------------+
        |              UNIQUE FEATURE A: SENSOR RELIABILITY ENGINE              |
        |  • Detects frozen ADC flatlines, rate-of-change jumps, high noise     |
        |  • Calculates calibration drift against standard reference buffers    |
        +-----------------------------------------------------------------------+
                                           |
                                           v
        +-----------------------------------------------------------------------+
        |           UNIQUE FEATURE B: TREND ANALYSIS & AI CLASSIFICATION        |
        |  • Rolling Z-Scores + Multi-point persistence confirmation            |
        |  • Random Forest ML model predicts Safe / Caution / Unsafe classes    |
        +-----------------------------------------------------------------------+
                                           |
                                           v
        +-----------------------------------------------------------------------+
        |              UNIQUE FEATURE C: EXPLAINABLE ALERT ENGINE               |
        |  • Cross-checks threshold breaches against sensor hardware status     |
        |  • Generates actionable operator recommendations and acknowledgment   |
        +-----------------------------------------------------------------------+
                                           |
                                           v
        +-----------------------------------------------------------------------+
        |              RESPONSIVE DASHBOARD & OPERATIONAL CONTROL               |
        |  • Real-time Streamlit Web Interface / Mobile-friendly responsive     |
        +-----------------------------------------------------------------------+
        ```
        """)

    with tab_iot:
        st.markdown("### 📡 Cloud & IoT Hardware Integration")
        st.markdown("""
        The system supports direct cloud ingestion from **ThingSpeak IoT** and local/cloud **MQTT brokers** (e.g., HiveMQ, Mosquitto).
        """)

        c_ts1, c_ts2 = st.columns(2)
        with c_ts1:
            st.markdown("#### ThingSpeak Cloud REST Integration")
            ts_chan = st.text_input("ThingSpeak Channel ID", value=os.getenv("THINGSPEAK_CHANNEL_ID", ""))
            ts_key = st.text_input("ThingSpeak Read API Key", value=os.getenv("THINGSPEAK_READ_API_KEY", ""), type="password")
            
            if st.button("Test ThingSpeak Polling"):
                if not ts_chan or not ts_key:
                    st.warning("⚠️ ThingSpeak credentials are not configured. Running in Demo Simulation Mode.")
                else:
                    success, msg, data = fetch_thingspeak_data(ts_chan, ts_key)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)

        with c_ts2:
            st.markdown("#### Example ESP32 JSON Payload")
            st.json({
                "device_id": "esp32_water_node_01",
                "timestamp": "2026-10-10 12:45:00",
                "ph": 7.28,
                "turbidity": 1.65,
                "tds": 235.0,
                "conductivity": 376.0,
                "temperature": 21.8,
                "battery_voltage": 3.92
            })
            st.caption("ESP32 transmits this JSON payload to MQTT topic: `water_quality/sensors/node_1`")

    with tab_limits:
        st.markdown("### 🎚️ Configurable Reference Ranges")
        st.caption("Adjust safe operating limits based on regional standards (WHO / BIS / EPA).")
        
        for p in ["ph", "turbidity", "tds", "conductivity", "temperature"]:
            cfg = st.session_state.reference_ranges[p]
            c_l1, c_l2 = st.columns(2)
            with c_l1:
                cfg["min_safe"] = st.number_input(f"{cfg['name']} Min Safe ({cfg['unit']})", value=float(cfg["min_safe"]))
            with c_l2:
                cfg["max_safe"] = st.number_input(f"{cfg['name']} Max Safe ({cfg['unit']})", value=float(cfg["max_safe"]))

        st.success("Custom limits saved to session!")

    with tab_about:
        st.markdown("### 💡 Proposed Project Novelties")
        st.markdown("""
        1. **Sensor Reliability Intelligence:**  
           Traditional monitoring systems mistake dirty water for broken probes (or vice-versa). Our engine evaluates ADC flatlines, noise variance, rate-of-change jumps, and calibration drift to tag whether an anomaly is physical water contamination or a faulty probe.
        
        2. **Trend-Based Early Warning:**  
           Rather than relying on brittle static threshold alarms that trigger false positives on single spikes, our multi-point persistence filter and rolling Z-score detect subtle deteriorating trajectories *before* hazardous levels are reached.
        
        3. **Explainable Actionable Alerts:**  
           Alerts detail the exact parameter deviation, cross-check sensor health status, and recommend prescriptive next steps (e.g. clean sensor vs isolate intake pipe).
        
        4. **Multi-Modal AI + Physical Physics Alignment:**  
           Combines an explainable Random Forest classifier with strict physical boundary validation for transparent screening.
        """)
        st.markdown(f'<div class="disclaimer-banner">{LEGAL_DISCLAIMER}</div>', unsafe_allow_html=True)

