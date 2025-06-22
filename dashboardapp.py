# dashboard_app.py (Final Updated Version)

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ================================
# 🔹 Load Model and Data
# ================================
model = joblib.load("models/sales_predictor.pkl")
features = joblib.load("models/feature_names.pkl")
test_df = pd.read_csv("models/test_data.csv")

# ================================
# 🔹 Ensure All Cities Are Represented
# ================================
expected_cities = ["Yangon", "Naypyitaw", "Mandalay"]
city_cols = [col for col in test_df.columns if col.startswith("City_")]

# Recover city names based on one-hot encoded columns
city_map = {col: col.replace("City_", "") for col in city_cols}
test_df["City"] = test_df[city_cols].idxmax(axis=1).map(city_map)

# Add dummy entries for missing cities with proper structure
for city in expected_cities:
    if city not in test_df["City"].unique():
        dummy_row = {col: 0 for col in test_df.columns}
        dummy_row.update({f"City_{city}": 1, "Actual Sales": 0, "City": city})
        test_df = pd.concat([test_df, pd.DataFrame([dummy_row])], ignore_index=True)

# ================================
# 🔹 Page Setup
# ================================
st.set_page_config(page_title="D-Mart BI Dashboard", layout="wide")
st.title("🧠📊 AI-Powered Business Intelligence Dashboard – D-Mart")

# ================================
# 🔹 KPI Section
# ================================
kpi1 = test_df["Actual Sales"].sum()
kpi2 = test_df["Rating"].mean()
kpi3 = test_df["Actual Sales"].mean()

col1, col2, col3 = st.columns(3)
col1.metric("🛒 Total Sales", f"₹{kpi1:,.0f}")
col2.metric("⭐ Average Rating", f"{kpi2:.2f}")
col3.metric("💰 Avg Transaction", f"₹{kpi3:,.0f}")
st.subheader("🏆 Top-Selling Product Lines")
product_cols = [col for col in test_df.columns if col.startswith("Product line_")]
product_df = test_df[product_cols].sum().sort_values(ascending=False).reset_index()
product_df.columns = ["Product", "Sales"]
product_df["Product"] = product_df["Product"].str.replace("Product line_", "")
fig_top = px.bar(product_df, x="Product", y="Sales")
st.plotly_chart(fig_top, use_container_width=True)

# ================================
# 🔹 Dropdown to Select City
# ================================
st.subheader("🏙️ Select City to View Analytics")
selected_city = st.selectbox("Choose a City", expected_cities)
filtered_df = test_df[test_df["City"] == selected_city]

# ================================
# 🔹 City-wise Sales Bar
# ================================
st.subheader("📍 City-wise Sales")
city_sales = test_df.groupby("City")["Actual Sales"].sum().reindex(expected_cities, fill_value=0).reset_index()
fig1 = px.bar(city_sales, x="City", y="Actual Sales", color="Actual Sales")
st.plotly_chart(fig1, use_container_width=True)

# ================================
# 🔹 Sales by Hour
# ================================
st.subheader(f"⏰ Sales by Hour – {selected_city}")
city_hour = filtered_df.groupby("Hour")["Actual Sales"].sum().reset_index()
fig2 = px.line(city_hour, x="Hour", y="Actual Sales", markers=True)
st.plotly_chart(fig2, use_container_width=True)

# ================================
# 🔹 Sales by Day of Week
# ================================
st.subheader(f"📅 Sales by Day of Week – {selected_city}")
city_day = filtered_df.groupby("Weekday")["Actual Sales"].sum().reset_index()
fig3 = px.bar(city_day, x="Weekday", y="Actual Sales")
st.plotly_chart(fig3, use_container_width=True)

# ================================
# 🔮 Predict Sales from User Input
# ================================
st.subheader("🔮 Predict Sales from User Input")

numerics = ["Quantity", "Unit price", "gross income", "Rating", "Hour", "Weekday", "Month"]
checkboxes = [
    "First Half", "Month Start", "Month End", "City_Naypyitaw", "City_Yangon", "City_Mandalay",
    "Customer type_Normal", "Customer type_Member", "Gender_Male", "Gender_Female",
    "Product line_Fashion accessories", "Product line_Food and beverages",
    "Product line_Health and beauty", "Product line_Home and lifestyle",
    "Product line_Sports and travel", "Payment_Credit card", "Payment_Ewallet", "Payment_Cash"
]

input_dict = {f: 0 for f in features}
input_dict["Quantity"] = st.slider("Quantity", 1, 10, 3)
input_dict["Unit price"] = st.slider("Unit Price", 10.0, 100.0, 50.0)
input_dict["gross income"] = st.slider("Gross Income", 5.0, 100.0, 20.0)
input_dict["Rating"] = st.slider("Customer Rating", 1.0, 10.0, 7.0)
input_dict["Hour"] = st.slider("Hour of Purchase", 0, 23, 14)
input_dict["Weekday"] = st.slider("Day of Week (0=Mon)", 0, 6, 3)
input_dict["Month"] = st.slider("Month", 1, 12, 6)
input_dict["Is Weekend"] = 1 if input_dict["Weekday"] >= 5 else 0

for label in checkboxes:
    if label in features:
        input_dict[label] = 1 if st.checkbox(label, label in ["First Half"]) else 0

input_df = pd.DataFrame([input_dict])[features]
predicted_log = model.predict(input_df)[0]
predicted_sales = np.expm1(predicted_log)
st.success(f"💰 Predicted Sales: ₹{predicted_sales:,.2f}")

# ================================
# 🔮 7-Day Sales Forecast
# ================================
st.subheader("📉 Sales Forecasting (7-day ahead prediction demo)")
future = test_df.sample(7).copy()
future_X = future[features]
future_pred = np.expm1(model.predict(future_X))
future["Forecasted Sales"] = future_pred
future["Date"] = pd.date_range(datetime.today(), periods=7)
fig7 = px.line(future, x="Date", y="Forecasted Sales", markers=True, title="7-Day Sales Forecast")
st.plotly_chart(fig7, use_container_width=True)

# ================================
# 📈 Actual vs Predicted Sales Visualization
# ================================
st.subheader("📊 Actual vs Predicted Sales")

y_pred_log = model.predict(test_df[features])
y_pred = np.expm1(y_pred_log)
test_df["Predicted Sales"] = y_pred

fig_actual_vs_pred = px.scatter(
    test_df,
    x="Actual Sales",
    y="Predicted Sales",
    trendline="ols",
    color_discrete_sequence=["#00CC96"],
    title="📈 Actual vs Predicted Sales",
    labels={"Actual Sales": "Actual Sales", "Predicted Sales": "Predicted Sales"},
)
st.plotly_chart(fig_actual_vs_pred, use_container_width=True)

# ================================
# 📊 Model Performance Metrics
# ================================
r2 = r2_score(test_df["Actual Sales"], test_df["Predicted Sales"])
mae = mean_absolute_error(test_df["Actual Sales"], test_df["Predicted Sales"])
rmse = np.sqrt(mean_squared_error(test_df["Actual Sales"], test_df["Predicted Sales"]))

st.markdown(f"""
**🔍 Model Performance Metrics:**
- 🧮 R² Score: `{r2:.4f}`
- 📉 MAE: `{mae:.2f}`
- 📉 RMSE: `{rmse:.2f}`
""")
 
 # Footer
st.markdown("---")
st.caption(" Built by Mohamed Ibrahim | Powered by Streamlit, XGBoost, and Plotly")
