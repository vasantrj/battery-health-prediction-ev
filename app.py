# app.py — Smart EV-aware cell-level battery health dashboard

import streamlit as st
import numpy as np
import pandas as pd
import time
import joblib
import os
import plotly.express as px
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

# ---------------------------------------------------
# Paths & Config
# ---------------------------------------------------
MODEL_PATH = "models/best_model.pkl"
HISTORY_FILE = "data/history.csv"

os.makedirs("data", exist_ok=True)

if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(
        columns=["timestamp", "Voltage", "Current", "Temp", "SOC", "SOH", "RUL", "anomaly"]
    ).to_csv(HISTORY_FILE, index=False)

# Load model and history
# Note: if MODEL_PATH does not exist at runtime this will raise — keep as in your original code.
best_model = joblib.load(MODEL_PATH)
history_log = pd.read_csv(HISTORY_FILE)

# ---------------------------------------------------
# EV Specs & Chemistry
# ---------------------------------------------------
EV_DATABASE = {
    "Tata Nexon EV": {"pack_voltage": 350, "capacity_kwh": 30.2, "chemistry": "NMC"},
    "MG ZS EV": {"pack_voltage": 400, "capacity_kwh": 44.5, "chemistry": "NMC"},
    "Hyundai Kona": {"pack_voltage": 356, "capacity_kwh": 39.2, "chemistry": "NMC"},
    "Tesla Model 3": {"pack_voltage": 400, "capacity_kwh": 60.0, "chemistry": "NCA"},
    "Nissan Leaf": {"pack_voltage": 360, "capacity_kwh": 40.0, "chemistry": "NMC"},
    "BYD Atto 3": {"pack_voltage": 403, "capacity_kwh": 49.9, "chemistry": "LFP"},
}

CHEMISTRY_VOLTAGE = {
    "LFP": 3.2,
    "NMC": 3.7,
    "NCA": 3.6,
    "LTO": 2.3,
}

# ---------------------------------------------------
# Battery metric functions
# ---------------------------------------------------
def calculate_soc(temp: float) -> float:
    return max(0.0, 100.0 - temp)

def calculate_soh(temp: float) -> float:
    # Simple degradation model: hotter → faster SOH loss
    return max(0.0, 92.0 - 0.12 * (temp - 25.0))

def calculate_rul(temp: float) -> float:
    max_cycles = 1500.0
    loss = max(0.0, (temp - 25.0) * 4.8)
    return max(10.0, max_cycles - loss)

# ---------------------------------------------------
# Core prediction (cell level)
# ---------------------------------------------------
def compute_metrics(v: float, c: float):
    power = v * c
    df = pd.DataFrame([[v, c, power]],
                      columns=["Voltage_measured", "Current_measured", "Power"])
    temp = float(best_model.predict(df)[0])
    soc = calculate_soc(temp)
    soh = calculate_soh(temp)
    rul = calculate_rul(temp)
    return temp, soc, soh, rul

# ---------------------------------------------------
# Enhanced Anomaly Detection (Upgraded)
# ---------------------------------------------------
def detect_anomaly(temp: float, soc: float, voltage: float, current: float):
    """
    Returns list of alert strings. This upgraded detector uses:
      - temperature thresholds (critical / high)
      - SOC threshold (low)
      - voltage thresholds (over/under)
      - current threshold (overcurrent; absolute value used to catch charge/discharge)
      - power threshold (abnormal power consumption)
    """
    alerts = []

    # --- Temperature anomalies ---
    if temp > 65:
        alerts.append("🔥 Critical temperature detected (>65°C)")
    elif temp > 55:
        alerts.append("⚠️ High temperature detected (>55°C)")

    # --- SOC anomaly ---
    if soc < 20:
        alerts.append("⚠️ Low State of Charge (<20%)")

    # --- Voltage anomalies ---
    # Typical li-ion cell max ~4.2-4.3V; min ~2.5-3.0V (we keep conservative thresholds)
    if voltage > 4.3:
        alerts.append("⚠️ Overvoltage risk (>4.3V)")
    if voltage < 3.0:
        alerts.append("⚠️ Undervoltage detected (<3.0V)")

    # --- Current anomalies ---
    # Use absolute current because large negative (charging) or positive (discharging) may be problematic
    if abs(current) > 1.8:
        alerts.append("⚠️ Overcurrent detected (>1.8A)")

    # --- Power anomalies ---
    power = voltage * current
    # High power for small cell (dataset scale) may indicate abnormal stress
    if abs(power) > 7.5:
        alerts.append("⚠️ Abnormal power consumption (>7.5W)")

    return alerts

# ---------------------------------------------------
# History logging
# ---------------------------------------------------
def save_history(v, c, t, s, h, r, anomaly=False):
    global history_log
    row = {
        "timestamp": time.time(),
        "Voltage": v,
        "Current": c,
        "Temp": t,
        "SOC": s,
        "SOH": h,
        "RUL": r,
        "anomaly": bool(anomaly),
    }
    history_log = pd.concat([history_log, pd.DataFrame([row])], ignore_index=True)
    history_log.to_csv(HISTORY_FILE, index=False)

# ---------------------------------------------------
# Plotly → PNG (for PDF export)
# ---------------------------------------------------
def fig_to_png(fig):
    try:
        png_bytes = fig.to_image(format="png", engine="kaleido")
        return BytesIO(png_bytes)
    except Exception:
        return None

# ---------------------------------------------------
# PDF report generator
# ---------------------------------------------------
def generate_pdf(summary: dict, figs: dict):
    out_path = "data/battery_report.pdf"
    doc = SimpleDocTemplate(out_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = [Paragraph("🔋 Battery Health Report", styles["Title"]), Spacer(1, 12)]

    for k, v in summary.items():
        elements.append(Paragraph(f"<b>{k}:</b> {v}", styles["Normal"]))
        elements.append(Spacer(1, 6))

    for title, buf in figs.items():
        if buf is None:
            continue
        buf.seek(0)
        elements.append(Paragraph(title, styles["Heading3"]))
        elements.append(RLImage(buf, width=450, height=250))
        elements.append(Spacer(1, 10))

    doc.build(elements)
    return out_path

# ---------------------------------------------------
# Streamlit UI config & theme tweaks
# ---------------------------------------------------
st.set_page_config(
    page_title="EV Battery Health Predictor",
    page_icon="🔋",
    layout="wide"
)

st.markdown("""
<style>
    
    .stTabs [role="tab"] {
        font-size: 16px;
        font-weight: 600;
    }
    .stMetric {
        
        border-radius: 10px;
        padding: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔋 EV Battery Health Monitoring & Prediction System")

tabs = st.tabs([
    "🧮 Manual Prediction",
    "🚗 Smart EV Model Selector",
    "📁 CSV Batch Prediction",
    "📊 Historical Analytics",
    "📄 Export Report",
])

# ---------------------------------------------------
# TAB 1 — Manual Cell-Level Prediction
# ---------------------------------------------------
with tabs[0]:
    st.header("🧮 Manual Cell-Level Prediction")

    col1, col2 = st.columns(2)
    v = col1.number_input("Cell Voltage (V)", min_value=3.0, max_value=5.0, value=3.72, step=0.01)
    c = col2.number_input("Cell Current (A)", min_value=-0.5, max_value=2.0, value=1.20, step=0.01)

    if st.button("Predict Cell Health "):
        temp, soc, soh, rul = compute_metrics(v, c)
        alerts = detect_anomaly(temp, soc, v, c)
        save_history(v, c, temp, soc, soh, rul, bool(alerts))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🌡 Temperature", f"{temp:.2f} °C")
        m2.metric("⚡ SOC", f"{soc:.1f} %")
        m3.metric("💪 SOH", f"{soh:.1f} %")
        m4.metric("⏳ RUL", f"{rul:.0f} cycles")

        if alerts:
            for a in alerts:
                st.error(a)
        else:
            st.success("Battery operating in normal range.")

# -------------------------------------------------------
# TAB 2 — Smart EV Model Selector (EV-aware, cell-safe)
# -------------------------------------------------------
with tabs[1]:
    st.header("🚗 EV-Aware Battery Health Estimation")

    ev_names = list(EV_DATABASE.keys()) + ["Custom EV"]
    selected_ev = st.selectbox("Select EV Model", ev_names)

    if selected_ev != "Custom EV":
        ev = EV_DATABASE[selected_ev]
        pack_v = ev["pack_voltage"]
        cap = ev["capacity_kwh"]
        chem = ev["chemistry"]
        nominal_cell_v = CHEMISTRY_VOLTAGE.get(chem, 3.7)

        series_cells = max(1, round(pack_v / nominal_cell_v))
        est_cell_v = pack_v / series_cells

        st.info(
            f"**{selected_ev}**\n\n"
            f"- Chemistry: **{chem}**\n"
            f"- Pack Voltage: **{pack_v} V**\n"
            f"- Capacity: **{cap} kWh**\n"
            f"- Estimated Series Cells: **{series_cells}**\n"
            f"- Suggested Cell Voltage ≈ **{est_cell_v:.2f} V**"
        )

        col1, col2 = st.columns(2)
        cell_v = col1.number_input(
            "Measured / Approx Cell Voltage (V)",
            min_value=3.0, max_value=5.0,
            value=float(np.round(est_cell_v, 2)),
            step=0.01,
            help="Use suggested value or measured lab/BMS value."
        )
        cell_c = col2.number_input(
            "Measured Cell Current (A)",
            min_value=-0.5, max_value=2.0,
            value=1.0,
            step=0.05,
            help="Use realistic cell-level current (dataset-like scale)."
        )

        if st.button("Predict EV Cell Health "):
            temp, soc, soh, rul = compute_metrics(cell_v, cell_c)
            alerts = detect_anomaly(temp, soc, cell_v, cell_c)
            save_history(cell_v, cell_c, temp, soc, soh, rul, bool(alerts))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🌡 Temperature", f"{temp:.2f} °C")
            m2.metric("⚡ SOC", f"{soc:.1f} %")
            m3.metric("💪 SOH", f"{soh:.1f} %")
            m4.metric("⏳ RUL", f"{rul:.0f} cycles")

            if alerts:
                for a in alerts:
                    st.error(a)
            else:
                st.success("EV cell health in safe operating zone.")

    else:
        st.subheader("Custom EV Configuration")
        pack_v = st.number_input("Battery Pack Voltage (V)", min_value=100.0, max_value=1000.0, value=400.0, step=5.0)
        cap = st.number_input("Pack Capacity (kWh)", min_value=10.0, max_value=150.0, value=40.0, step=1.0)
        chem = st.selectbox("Battery Chemistry", list(CHEMISTRY_VOLTAGE.keys()))

        nominal_cell_v = CHEMISTRY_VOLTAGE.get(chem, 3.7)
        series_cells = max(1, round(pack_v / nominal_cell_v))
        est_cell_v = pack_v / series_cells

        st.info(
            f"- Chemistry: **{chem}**\n"
            f"- Estimated Series Cells: **{series_cells}**\n"
            f"- Suggested Cell Voltage ≈ **{est_cell_v:.2f} V**"
        )

        col1, col2 = st.columns(2)
        cell_v = col1.number_input(
            "Measured / Approx Cell Voltage (V)",
            min_value=3.0, max_value=5.0,
            value=float(np.round(est_cell_v, 2)),
            step=0.01
        )
        cell_c = col2.number_input(
            "Measured Cell Current (A)",
            min_value=-0.5, max_value=2.0,
            value=1.0,
            step=0.05
        )

        if st.button("Predict Custom EV Cell Health "):
            temp, soc, soh, rul = compute_metrics(cell_v, cell_c)
            alerts = detect_anomaly(temp, soc, cell_v, cell_c)
            save_history(cell_v, cell_c, temp, soc, soh, rul, bool(alerts))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🌡 Temperature", f"{temp:.2f} °C")
            m2.metric("⚡ SOC", f"{soc:.1f} %")
            m3.metric("💪 SOH", f"{soh:.1f} %")
            m4.metric("⏳ RUL", f"{rul:.0f} cycles")

            if alerts:
                for a in alerts:
                    st.error(a)
            else:
                st.success("✅ Custom EV cell health in safe range.")

# ---------------------------------------------------
# TAB 3 — CSV Batch Prediction
# ---------------------------------------------------
with tabs[2]:
    st.header("📁 CSV Batch Prediction")

    uploaded = st.file_uploader("Upload CSV with 'Voltage' and 'Current' columns", type="csv")

    if uploaded:
        batch_df = pd.read_csv(uploaded)

        if {"Voltage", "Current"}.issubset(batch_df.columns):

            def apply_row(row):
                t, s, h, r_ = compute_metrics(row["Voltage"], row["Current"])
                anomaly = bool(detect_anomaly(t, s, row["Voltage"], row["Current"]))
                return pd.Series([t, s, h, r_, anomaly])

            batch_df[["Temp", "SOC", "SOH", "RUL", "anomaly"]] = batch_df.apply(apply_row, axis=1)

            st.success("Batch prediction completed.")
            st.dataframe(batch_df)

            st.download_button(
                "Download Predictions CSV",
                batch_df.to_csv(index=False),
                file_name="batch_predictions.csv",
                mime="text/csv"
            )
        else:
            st.error("Uploaded CSV must contain 'Voltage' and 'Current' columns.")

# ---------------------------------------------------
# TAB 4 — Historical Analytics
# ---------------------------------------------------
with tabs[3]:
    st.header("📊 Historical Battery Analytics")

    if len(history_log) > 2:
        df_hist = history_log.copy()
        df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"], unit="s")

        st.subheader("SOC Trend Over Time")
        st.plotly_chart(
            px.line(df_hist, x="timestamp", y="SOC", title="State of Charge (%)"),
            use_container_width=True
        )

        st.subheader("Temperature Trend Over Time")
        st.plotly_chart(
            px.line(df_hist, x="timestamp", y="Temp", title="Temperature (°C)"),
            use_container_width=True
        )

        st.subheader("SOH Degradation Curve")
        st.plotly_chart(
            px.line(df_hist, x="timestamp", y="SOH", title="State of Health (%)"),
            use_container_width=True
        )

# ---------------------------------------------------
# TAB 5 — Export Report
# ---------------------------------------------------
with tabs[4]:
    st.header("📄 Export Battery Health Report")

    if st.button("Generate PDF Report "):
        if len(history_log) == 0:
            st.warning("No history available to generate report.")
        else:
            avg_temp = history_log["Temp"].astype(float).mean()
            avg_soc = history_log["SOC"].astype(float).mean()

            summary = {
                "Total Records": len(history_log),
                "Average Temperature": f"{avg_temp:.2f} °C",
                "Average SOC": f"{avg_soc:.2f} %",
            }

            figs = {}
            figs["Temperature Trend"] = fig_to_png(px.line(history_log, y="Temp", title="Temperature Trend"))
            figs["SOC Trend"] = fig_to_png(px.line(history_log, y="SOC", title="SOC Trend"))

            pdf_path = generate_pdf(summary, figs)

            with open(pdf_path, "rb") as f:
                st.success("Report generated successfully.")
                st.download_button(
                    "⬇️ Download PDF Report",
                    f,
                    file_name="battery_report.pdf",
                    mime="application/pdf"
                )

# End of file
# streamlit run app.py
               
               
               
               
                