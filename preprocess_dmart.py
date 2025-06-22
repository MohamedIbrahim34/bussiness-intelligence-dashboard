# preprocess_dmart.py

import pandas as pd

def load_and_clean_dmart_data(path="C:/Users/hafee/Desktop/D-mart AI dashboard/data/DmartSalesData.csv.csv"):
    df = pd.read_csv(path)

    # ============================
    # 🔹 Step 1: Basic Cleaning
    # ============================
    df.drop_duplicates(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df.dropna(subset=['Date', 'Time'], inplace=True)

    df = df[df['Total'] > 0]

    # ============================
    # 🔹 Step 2: Remove Outliers (on Total Sales)
    # ============================
    lower = df['Total'].quantile(0.01)
    upper = df['Total'].quantile(0.99)
    df = df[(df['Total'] >= lower) & (df['Total'] <= upper)]

    # ============================
    # 🔹 Step 3: Feature Engineering
    # ============================
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Weekday'] = df['Date'].dt.weekday
    df['Is Weekend'] = df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)
    df['First Half'] = df['Date'].dt.day.apply(lambda x: 1 if x <= 15 else 0)
    df['Month Start'] = df['Date'].dt.is_month_start.astype(int)
    df['Month End'] = df['Date'].dt.is_month_end.astype(int)

    df['Hour'] = df['Time'].dt.hour

    # Revenue related
    df['Profit Margin'] = df['gross income'] / df['Total']
    df['Income per Item'] = df['gross income'] / df['Quantity']

    # Handle inf/nan
    df.replace([float('inf'), -float('inf')], 0, inplace=True)
    df.fillna(0, inplace=True)

    return df
