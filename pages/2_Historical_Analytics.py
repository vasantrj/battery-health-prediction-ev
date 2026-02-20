import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 Historical Battery Analytics Dashboard")

# Load historical logged data
DATA_PATH = "data/history.csv"  # or parquet file
try:
    df = pd.read_csv(DATA_PATH)
    st.success("Historical data loaded successfully!")
except:
    st.warning("No historical data found. Start real-time monitoring to generate history.")
    st.stop()

st.subheader("🔋 Raw Data Preview")
st.dataframe(df.head())

# ---- Plot 1: Temperature trend ----
st.subheader("🌡 Temperature Trend Over Time")
fig1, ax1 = plt.subplots()
ax1.plot(df["timestamp"], df["Temp"], linewidth=2)
ax1.set_xlabel("Time")
ax1.set_ylabel("Temperature (°C)")
st.pyplot(fig1)

# ---- Plot 2: SOC trend ----
st.subheader("⚡ SOC Trend Over Time")
fig2, ax2 = plt.subplots()
ax2.plot(df["timestamp"], df["SOC"], linewidth=2)
ax2.set_xlabel("Time")
ax2.set_ylabel("SOC (%)")
st.pyplot(fig2)

# ---- Comparison by cycles ----
if "cycle" in df.columns:
    st.subheader("🔁 Cycle-wise Temperature Comparison")
    fig3, ax3 = plt.subplots()
    for cycle in df["cycle"].unique():
        subset = df[df["cycle"] == cycle]
        ax3.plot(subset["timestamp"], subset["Temp"], label=f"Cycle {cycle}")
    ax3.legend()
    st.pyplot(fig3)

# ---- Degradation curve ----
st.subheader("📉 Battery Degradation Curve")
fig4, ax4 = plt.subplots()
ax4.plot(df["timestamp"], df["SOH"])
ax4.set_xlabel("Time")
ax4.set_ylabel("SOH (%)")
st.pyplot(fig4)

# ---- RUL prediction trend ----
st.subheader("⏳ Remaining Useful Life Prediction Trend")
fig5, ax5 = plt.subplots()
ax5.plot(df["timestamp"], df["RUL"])
ax5.set_xlabel("Time")
ax5.set_ylabel("RUL (Cycles)")
st.pyplot(fig5)
