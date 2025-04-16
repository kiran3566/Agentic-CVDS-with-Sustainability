
from data import data as df
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate complex data
np.random.seed(42)
days = np.arange(1, 61)
cattle_ids = [f"{i}" for i in range(1, 10)]

data=df
multi_data = []
for _, row in data.iterrows():
    multi_data.append({
        "Day": row["Step"],
        "Cow": row["AgentID"],
        "Weight": row["Shrunken_Body_Weight"],
        "Feed": row["DMI"],
        "EBF": row["EBF"],
        "Emissions": row["Emissions"]
    })

full_df = pd.DataFrame(multi_data)

# Layout config
st.set_page_config(layout="wide")
st.markdown("""
    <style>
        .main {
            background: linear-gradient(to bottom right, #ecfdf5, #d1fae5);
        }
        h1, h2, h3 {
            color: #064e3b;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🐄 Agent-Based CVDS Simulation Control Center")
st.markdown("A multi-panel dashboard for exploring cattle performance, emissions, and feed dynamics.")

# Sidebar controls
st.sidebar.title("🔧 Simulation Controls")
diet_energy = st.sidebar.slider("Diet Energy (Mcal/kg)", 1.5, 3.0, 2.2, 0.1)
cow_count = st.sidebar.slider("Number of Cattle", 1, 150, 100, step=1)
weather = st.sidebar.selectbox("Weather Condition", ["Normal", "Hot", "Cold", "Humid"])
include_emissions = st.sidebar.checkbox("Include Emissions Mitigation", value=True)

st.sidebar.markdown("---")
st.sidebar.button("🚀 Run Full Simulation")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Summary", "📈 Time Series", "📋 Comparison"])

with tab1:
    st.header("📊 Summary Statistics")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Average Body Weight", f"{full_df['Weight'].mean():.2f} lbs")
        st.metric("Average Feed Intake", f"{full_df['Feed'].mean():.2f} kg/day")
    with col_b:
        st.metric("Methane Emission", f"{full_df['Emissions'].mean():.2f} kg/day")
        st.metric("Number of Cattle", f"{cow_count}")

    st.subheader("📌 Final weights of each cow - Total Feed - Total Methane emitted")
    pivot = full_df.groupby("Cow").agg({
    "Weight": "last",
    "Feed": "sum",
    "Emissions": "sum"}).round(2)

    st.dataframe(pivot)

with tab2:
    st.header("📈 Daily Progression by Cow")
    fig, ax = plt.subplots(3, 1, figsize=(14, 12))
    for cow in cattle_ids:
        cow_df = full_df[full_df['Cow'] == int(cow)]
        ax[0].plot(cow_df['Day'], cow_df['Weight'], label=f"cow {cow}")
        ax[1].plot(cow_df['Day'], cow_df['Emissions'], label=f"cow {cow}")
        ax[2].plot(cow_df['Day'], cow_df['Feed'], label=f"cow {cow}")

    ax[0].set_title("Weight over Time")
    ax[1].set_title("Methane Emission over Time")
    ax[2].set_title("Feed Intake over Time")
    for a in ax:
        a.legend(fontsize='small')
        a.grid(True)
    st.pyplot(fig)

with tab3:
    st.header("📋 Scenario Comparisons")
    selected_cows = st.multiselect("Select Cattle to Compare", cattle_ids, default=cattle_ids[:3])
    compare_metric = st.radio("Metric to Compare", ["Weight", "Emissions", "Feed"])

    fig, ax = plt.subplots(figsize=(12, 6))
    for cow in selected_cows:
        data = full_df[full_df['Cow'] == int(cow)]
        ax.plot(data['Day'], data[compare_metric], label=cow)
    ax.set_title(f"Comparison of {compare_metric} Over Time")
    ax.set_xlabel("Day")
    ax.set_ylabel(compare_metric)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

st.markdown("---")
st.caption("© 2025 Agent-Based CVDS | Complex Streamlit Edition | Built with 🧠 + ❤️")

