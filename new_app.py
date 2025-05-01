
from data import data as df
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
from model import *
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
        "Emissions": row["Emissions(g)"]
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
cow_count = st.sidebar.slider("Number of Cattle", 1, 123, 10, step=1)
weather = st.sidebar.selectbox("Weather Condition", ["Normal", "Hot", "Cold", "Humid"])
include_emissions = st.sidebar.checkbox("Include Emissions Mitigation", value=True)

st.sidebar.markdown("---")
st.sidebar.button("🚀 Run Full Simulation")

cattle_ids = [f"{i}" for i in range(1, cow_count+1)]

# Tabs
tab1, tab2, tab3,tab4 = st.tabs(["📊 Summary", "📈 Time Series", "📋 Comparison","🐄 New Scenario"])

with tab1:
    st.header("📊 Summary Statistics")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Average Body Weight", f"{full_df['Weight'].iloc[:cow_count].mean():.2f} lbs")
        st.metric("Average Feed Intake", f"{full_df['Feed'].iloc[:cow_count].mean():.2f} kg/day")
    with col_b:
        st.metric("Methane Emission", f"{full_df['Emissions'].iloc[:cow_count].mean():.2f} kg/day")
        st.metric("Number of Cattle", f"{cow_count}")

    st.subheader("📌 Final weights of each cow - Total Feed - Total Methane emitted")
    pivot = full_df.groupby("Cow").agg({
    "Weight": "last",
    "Feed": "sum",
    "Emissions": "sum"}).iloc[:cow_count].round(2)

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

with tab4:
    st.header("🐄 Input Cattle Information")

    # Input: number of cattle and pens
    num_cattle = st.number_input("Enter number of cattle", min_value=1, max_value=100, step=1, key="num_cattle")
    num_pens = st.number_input("Enter number of pens", min_value=1, max_value=20, step=1, key="num_pens")

    pen_options = [str(i + 1) for i in range(num_pens)]
    today = datetime.today().date()

    # ---------- Cattle Table ----------
    st.subheader("📋 Cattle Table")

    default_cattle_data = [
        {
            "Cow": f"Cow {i + 1}",
            "PenID": pen_options[i % num_pens],
            "iDate": today,
            "HipHeight": 45.0,
            "AgeHipHeight": 8.0,
            "Beef": "TRUE",
            "Sex": "S",
            "iBW": 600.0,
            "IsiBWShrunk": "TRUE",
            "Implants": "FALSE",
            "Holstein": "FALSE",
            "BCS": 5
        }
        for i in range(num_cattle)
    ]

    cattle_df = st.data_editor(
        default_cattle_data,
        use_container_width=True,
        num_rows="dynamic",
        disabled=["Cow"],
        column_config={
            "Cow": st.column_config.TextColumn("🐄 Cow ID"),
            "PenID": st.column_config.SelectboxColumn("Pen ID", options=pen_options),
            "iDate": st.column_config.DateColumn("🗓️ Intake Date"),
            "HipHeight": st.column_config.NumberColumn("Hip Height (in)"),
            "AgeHipHeight": st.column_config.NumberColumn("Age from Hip Height (months)"),
            "Beef": st.column_config.SelectboxColumn("Beef", options=["TRUE", "FALSE"]),
            "Sex": st.column_config.SelectboxColumn("Sex", options=["B", "H", "S"]),
            "iBW": st.column_config.NumberColumn("Initial BW (lbs)"),
            "IsiBWShrunk": st.column_config.SelectboxColumn("Is BW Shrunk?", options=["TRUE", "FALSE"]),
            "Implants": st.column_config.SelectboxColumn("Implants", options=["TRUE", "FALSE"]),
            "Holstein": st.column_config.SelectboxColumn("Holstein", options=["TRUE", "FALSE"]),
            "BCS": st.column_config.NumberColumn("Body Condition Score (1–9)", min_value=1, max_value=9, step=1)
        }
    )

    cattle_inputs = cattle_df if isinstance(cattle_df, list) else cattle_df.to_dict("records")

    # ---------- Feed Plan Table ----------
    st.subheader("🌾 Feed Plan by Pen")

    default_feed_data = []
    for i in range(num_pens):
        pen_id = str(i + 1)
        default_feed_data.append({
            "PenRecord": pen_id,
            "iDate": today.strftime("%m/%d/%Y"),
            "NEm": 1.2,
            "NEg": 0.8
        })
        # Add a future entry to prevent IndexError in model
        default_feed_data.append({
            "PenRecord": pen_id,
            "iDate": (today + timedelta(days=999)).strftime("%m/%d/%Y"),
            "NEm": 1.2,
            "NEg": 0.8
        })

    feed_df = st.data_editor(
        default_feed_data,
        use_container_width=True,
        disabled=["PenRecord"],
        column_config={
            "PenRecord": st.column_config.TextColumn("Pen ID"),
            "iDate": st.column_config.TextColumn("Feed Start Date (MM/DD/YYYY)"),
            "NEm": st.column_config.NumberColumn("Net Energy for Maintenance (Mcal/lb)"),
            "NEg": st.column_config.NumberColumn("Net Energy for Gain (Mcal/lb)")
        }
    )

    feed_plan = feed_df if isinstance(feed_df, list) else feed_df.to_dict("records")

    st.success("Cattle and feed plan inputs collected.")

    # ---------- Simulation Runner ----------
    st.subheader("🚀 Run Simulation")

    n_days = st.number_input("Enter number of simulation days", min_value=1, max_value=365, value=30)

    if st.button("Run CVDS Simulation"):
        for cow in cattle_inputs:
            if isinstance(cow["iDate"], (datetime, date)):
                cow["iDate"] = cow["iDate"].strftime("%m/%d/%Y")

        for feed in feed_plan:
            if isinstance(feed["iDate"], (datetime, date)):
                feed["iDate"] = feed["iDate"].strftime("%m/%d/%Y")

        model = CVDSModel(init_cattles=cattle_inputs, feed_plan=feed_plan, n_pens=num_pens)

        for _ in range(n_days):
            model.step()

        df = model.datacollector.get_agent_vars_dataframe().reset_index()
        df = df.rename(columns={"AgentID": "Cow", "Step": "Day"})
        df["Cow"] = df["Cow"].astype(str)

        if "Emissions(g)" not in df.columns:
            df["Emissions(g)"] = 0.0

        st.success("Simulation completed.")

        # ---------- Plots ----------
        st.subheader("📉 Shrunken Body Weight over Time")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        for cow_id in df["Cow"].unique():
            cow_df = df[df["Cow"] == cow_id]
            ax1.plot(cow_df["Day"], cow_df["Shrunken_Body_Weight"], label=f"Cow {cow_id}")
        ax1.set_xlabel("Day")
        ax1.set_ylabel("Shrunken Body Weight (lbs)")
        ax1.set_title("Shrunken Body Weight vs Day")
        ax1.legend(fontsize="x-small")
        ax1.grid(True)
        st.pyplot(fig1)

        st.subheader("🔥 Methane Emissions over Time")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        for cow_id in df["Cow"].unique():
            cow_df = df[df["Cow"] == cow_id]
            ax2.plot(cow_df["Day"], cow_df["Emissions(g)"], label=f"Cow {cow_id}")
        ax2.set_xlabel("Day")
        ax2.set_ylabel("Emissions (g CH₄)")
        ax2.set_title("Emissions vs Day")
        ax2.legend(fontsize="x-small")
        ax2.grid(True)
        st.pyplot(fig2)

        st.subheader("💪 Empty Body Fat (EBF) over Time")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        for cow_id in df["Cow"].unique():
            cow_df = df[df["Cow"] == cow_id]
            ax3.plot(cow_df["Day"], cow_df["EBF"], label=f"Cow {cow_id}")
        ax3.set_xlabel("Day")
        ax3.set_ylabel("EBF (%)")
        ax3.set_title("Empty Body Fat vs Day")
        ax3.legend(fontsize="x-small")
        ax3.grid(True)
        st.pyplot(fig3)

        # Store in session for use in other tabs
        st.session_state["cvds_output_df"] = df


st.markdown("---")
st.caption("© 2025 Agent-Based CVDS | Complex Streamlit Edition | Built with 🧠 + ❤️")

