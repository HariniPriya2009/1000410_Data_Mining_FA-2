# =========================================
# FA-1 ATM Intelligence Demand Forecasting
# Data Preprocessing + Exploration
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries loaded")


# =========================================
# Load dataset
# =========================================

file_path = "atm_cash_management_dataset.csv"

df = pd.read_csv(file_path)

print("Dataset loaded")
print(df.head())


# =========================================
# Dataset info
# =========================================

print(df.shape)
print(df.columns)
print(df.info())


# =========================================
# Missing values
# =========================================

print("Missing values:")
print(df.isnull().sum())


# Fill missing values

df["Holiday_Flag"].fillna(0, inplace=True)
df["Special_Event_Flag"].fillna(0, inplace=True)

df.dropna(inplace=True)

print("After cleaning")
print(df.isnull().sum())


# =========================================
# Date formatting
# =========================================

df["Date"] = pd.to_datetime(df["Date"])

df["Month"] = df["Date"].dt.month
df["Week_Number"] = df["Date"].dt.isocalendar().week
df["Day_Name"] = df["Date"].dt.day_name()

print(df.head())


# =========================================
# Encoding categories
# =========================================

day_map = {
    "Monday":1,
    "Tuesday":2,
    "Wednesday":3,
    "Thursday":4,
    "Friday":5,
    "Saturday":6,
    "Sunday":7
}

df["Day_of_Week"] = df["Day_Name"].map(day_map)


time_map = {
    "Morning":1,
    "Afternoon":2,
    "Evening":3,
    "Night":4
}

df["Time_of_Day"] = df["Time_of_Day"].map(time_map)


loc_map = {
    "Urban":1,
    "Semi-Urban":2,
    "Rural":3
}

df["Location_Type"] = df["Location_Type"].map(loc_map)


weather_map = {
    "Sunny":1,
    "Rainy":2,
    "Cloudy":3,
    "Storm":4
}

df["Weather_Condition"] = df["Weather_Condition"].map(weather_map)

print("Encoding done")


# =========================================
# Normalization
# =========================================

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

num_cols = [
    "Total_Withdrawals",
    "Total_Deposits",
    "Previous_Day_Cash_Level",
    "Cash_Demand_Next_Day"
]

df[num_cols] = scaler.fit_transform(df[num_cols])

print("Normalization done")


# =========================================
# Logical check
# =========================================

df["Error_Flag"] = df["Total_Withdrawals"] > df["Previous_Day_Cash_Level"]

print("Errors found:", df["Error_Flag"].sum())


# =========================================
# VISUALIZATION (for storyboard)
# =========================================

plt.figure()
sns.histplot(df["Total_Withdrawals"])
plt.title("Withdrawals Distribution")
plt.show()


plt.figure()
sns.boxplot(x=df["Total_Withdrawals"])
plt.title("Outliers Check")
plt.show()


plt.figure()
plt.plot(df["Date"], df["Total_Withdrawals"])
plt.title("Withdrawals Over Time")
plt.show()


plt.figure()
sns.barplot(x=df["Holiday_Flag"], y=df["Total_Withdrawals"])
plt.title("Holiday Impact")
plt.show()


plt.figure()
sns.barplot(x=df["Day_of_Week"], y=df["Total_Withdrawals"])
plt.title("Withdrawals by Day")
plt.show()


plt.figure()
sns.scatterplot(
    x=df["Previous_Day_Cash_Level"],
    y=df["Cash_Demand_Next_Day"]
)
plt.title("Cash Level vs Next Day Demand")
plt.show()


# =========================================
# Save cleaned dataset
# =========================================

df.to_csv("cleaned_atm_data.csv", index=False)

print("Saved cleaned data")
