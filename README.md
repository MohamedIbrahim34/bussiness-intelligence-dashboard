# 🧠📊 D-Mart AI-Powered Business Intelligence Dashboard

Welcome to the **D-Mart Sales Intelligence Dashboard**, a Streamlit-powered web app that leverages **machine learning** and **interactive visualizations** to analyze and predict retail sales across Myanmar cities: **Yangon**, **Naypyitaw**, and **Mandalay**.


---

## 🚀 Features

- 📊 **City-Wise Analytics**: Filter and compare sales across cities.
- ⏰ **Sales by Hour**: Discover peak business hours.
- 📅 **Sales by Weekday**: Understand day-of-week performance.
- 🔮 **Sales Prediction**: Predict transaction sales using XGBoost.
- 📈 **Actual vs Predicted Analysis**: Visualize how accurate the model is.
- 📉 **7-Day Forecast**: Simulated sales forecasts for the next week.
- 🧮 **Model Performance Metrics**: R², MAE, and RMSE insights.

---

## 🗂️ Project Structure

📁 D-mart AI dashboard/
│

├── data/

│ └── DmartSalesData.csv.csv # Raw sales dataset


├── models/

│ ├── sales_predictor.pkl # Trained XGBoost model

│ ├── feature_names.pkl # Feature columns

│ └── test_data.csv # Test set with actual sales


├── dashboardapp.py # Streamlit dashboard app

├── load_dmart_model.py # Model training & export

├── preprocess_dmart.py # Data cleaning & feature engineering


---

## 🧠 Model Details

- **Algorithm**: XGBoost Regressor
- **Target Variable**: Log-transformed Total Sales
- **Input Features**: Customer attributes, product info, datetime features
- **Metrics Used**:
  - 📉 Mean Absolute Error (MAE)
  - 📉 Root Mean Squared Error (RMSE)
  - 🧮 R² Score

---

## 🛠️ Setup Instructions in bash

1. Clone the Repository

In bash git clone https://github.com/MohamedIbrahim34/bussiness-intelligence-dashboard.git
cd bussiness-intelligence-dashboard

2. Create & Activate a Virtual Environment

python -m venv venv
venv\Scripts\activate   # On Windows

3. Install Dependencies

pip install -r requirements.txt
If you don't have requirements.txt, install manually:
pip install streamlit pandas joblib xgboost scikit-learn plotly

4. Run the Dashboard

streamlit run dashboardapp.py

Made  by Mohamed Ibrahim

📧 Email:hafeezibrahim000@gmail.com
📌 GitHub: @MohamedIbrahim34

![Dashboard Screenshot](https://github.com/user-attachments/assets/9f8e46eb-f934-4bda-9e70-6ea379dc0de9)

