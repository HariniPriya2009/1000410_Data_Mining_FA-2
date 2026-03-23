import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="ATM Intelligence Dashboard",
    layout="wide"
)

# =========================
# CSS THEME
# =========================

st.markdown("""
<style>

.stApp {
    background-color: #0f172a;
}

.main-header {
    font-size: 2.6rem;
    color: #38bdf8;
    text-align: center;
    font-weight: bold;
}

.sub-header {
    font-size: 1.5rem;
    color: #a5b4fc;
    border-bottom: 2px solid #6366f1;
    margin-top: 1rem;
}

.card {
    background-color: #1e293b;
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
}

.insight {
    background-color: #1e293b;
    padding: 1rem;
    border-left: 5px solid #38bdf8;
    border-radius: 8px;
    color: white;
}

.warning {
    background-color: #3f1d1d;
    padding: 1rem;
    border-left: 5px solid yellow;
    border-radius: 8px;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =========================
# GRAPH THEME
# =========================

plt.style.use("dark_background")

sns.set_style("dark")

plt.rcParams["figure.facecolor"] = "#0f172a"
plt.rcParams["axes.facecolor"] = "#1e293b"
plt.rcParams["axes.labelcolor"] = "white"
plt.rcParams["xtick.color"] = "white"
plt.rcParams["ytick.color"] = "white"
plt.rcParams["axes.titlecolor"] = "#38bdf8"

# =========================
# HEADER
# =========================

st.markdown(
    "<div class='main-header'>ATM Cash Intelligence Dashboard</div>",
    unsafe_allow_html=True
)

# =========================
# FILE UPLOAD
# =========================

file = st.file_uploader("Upload Dataset", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    df["Date"] = pd.to_datetime(df["Date"])

    df["Day_Name"] = df["Date"].dt.day_name()

    # =========================
    # SIDEBAR FILTER
    # =========================

    st.sidebar.header("Filters")

    atm = st.sidebar.selectbox(
        "Select ATM",
        df["ATM_ID"].unique()
    )

    df = df[df["ATM_ID"] == atm]

    # =========================
    # KPI CARDS
    # =========================

    st.markdown(
        "<div class='sub-header'>Key Metrics</div>",
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    c1.markdown(
        f"<div class='card'>Total Withdrawals<br>{int(df['Total_Withdrawals'].sum())}</div>",
        unsafe_allow_html=True
    )

    c2.markdown(
        f"<div class='card'>Total Deposits<br>{int(df['Total_Deposits'].sum())}</div>",
        unsafe_allow_html=True
    )

    c3.markdown(
        f"<div class='card'>Avg Demand<br>{int(df['Cash_Demand_Next_Day'].mean())}</div>",
        unsafe_allow_html=True
    )

    # =========================
    # HISTOGRAM
    # =========================

    st.markdown(
        "<div class='sub-header'>Withdrawals Distribution</div>",
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots()

    sns.histplot(
        df["Total_Withdrawals"],
        kde=True,
        ax=ax
    )

    st.pyplot(fig)

    # =========================
    # LINE
    # =========================

    st.markdown(
        "<div class='sub-header'>Withdrawals Over Time</div>",
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots()

    ax.plot(
        df["Date"],
        df["Total_Withdrawals"]
    )

    st.pyplot(fig)

    # =========================
    # BAR
    # =========================

    st.markdown(
        "<div class='sub-header'>Day vs Withdrawals</div>",
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots()

    sns.barplot(
        x=df["Day_of_Week"],
        y=df["Total_Withdrawals"],
        ax=ax
    )

    st.pyplot(fig)

    # =========================
    # SCATTER
    # =========================

    st.markdown(
        "<div class='sub-header'>Cash vs Demand</div>",
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots()

    sns.scatterplot(
        x=df["Previous_Day_Cash_Level"],
        y=df["Cash_Demand_Next_Day"],
        ax=ax
    )

    st.pyplot(fig)

    # =========================
    # HEATMAP
    # =========================

    st.markdown(
        "<div class='sub-header'>Correlation</div>",
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots()

    sns.heatmap(
        df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

    # =========================
    # CLUSTERING
    # =========================

    st.markdown(
        "<div class='sub-header'>KMeans Clustering</div>",
        unsafe_allow_html=True
    )

    X = df[[
        "Total_Withdrawals",
        "Total_Deposits",
        "Cash_Demand_Next_Day"
    ]]

    scaler = MinMaxScaler()

    X = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3)

    df["Cluster"] = kmeans.fit_predict(X)

    fig, ax = plt.subplots()

    sns.scatterplot(
        x=df["Total_Withdrawals"],
        y=df["Cash_Demand_Next_Day"],
        hue=df["Cluster"],
        ax=ax
    )

    st.pyplot(fig)

    # =========================
    # ANOMALY
    # =========================

    st.markdown(
        "<div class='sub-header'>Anomaly Detection</div>",
        unsafe_allow_html=True
    )

    df["Anomaly"] = df["Total_Withdrawals"] > df["Previous_Day_Cash_Level"]

    anomalies = df[df["Anomaly"] == True]

    st.dataframe(anomalies)

    # =========================
    # INSIGHTS
    # =========================

    st.markdown(
        "<div class='sub-header'>Insights</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='insight'>Weekend withdrawals are higher</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='insight'>Urban ATMs show more demand</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='warning'>Some ATMs show abnormal spikes</div>",
        unsafe_allow_html=True
    )
