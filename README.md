# 1000410_Data_Mining_FA-2

ATM Intelligence Demand Forecasting with Data Mining

🎯 Purpose of the Project

This project focuses on transforming a cleaned ATM dataset into actionable insights using data mining techniques. It includes:

Exploratory Data Analysis (EDA)

Clustering of ATMs based on demand

Anomaly detection during holidays/events

An interactive dashboard for decision-making

The goal is to help bank managers optimize cash management and make data-driven decisions.

Key Features of the Application

Streamlit Link: https://1000410-harini-priya-karthikeyan-datamining-fa.streamlit.app/

1. Dashboard Overview

Total transactions, withdrawals, and ATMs

Withdrawal trends over time

Distribution by location

Key insights (peak days, events, holidays)

2. Exploratory Data Analysis (EDA)

Histogram (Withdrawals & Deposits)

Box plots (Outlier detection)

Time-based trends (daily, weekly, monthly)

Holiday & special event impact

Weather & competitor ATM analysis

Correlation heatmap & scatter plots


Helps identify patterns, trends, and relationships

3. Clustering Analysis

Uses K-Means Clustering

Feature selection option

Elbow Method & Silhouette Score

Cluster visualization (2D & 3D)

Cluster Interpretation:

High-Demand ATMs

Medium-Demand ATMs

Steady-Demand ATMs

Low-Demand ATMs

Helps group ATMs for efficient cash allocation

4. Anomaly Detection

Methods used:

Z-Score

IQR Method

Isolation Forest

Local Outlier Factor (LOF)

Detects unusual withdrawal spikes

Compares anomalies on holidays & events

Visual highlights of anomalies

Helps prevent cash shortages or overstocking

5. Interactive ATM Planner

Filter by:

Day of Week

Time of Day

Location Type

Displays:

Summary metrics

Charts

Data preview

Export options:

CSV file

Text report

Enables real-time decision support

Technologies Used

Python

Streamlit (for dashboard)

Pandas & NumPy (data processing)

Plotly & Matplotlib (visualization)

Seaborn (EDA)

Scikit-learn (ML models)
