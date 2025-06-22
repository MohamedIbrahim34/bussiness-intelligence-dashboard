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
![Screenshot 2025-06-22 144830](https://github.com/user-attachments/assets/b7962ef1-7d63-4163-b111-5e4cfb0ffea6)
![Screenshot 2025-06-22 144839](https://github.com/user-attachments/assets/949509e3-d21c-419e-8595-a46f6979c816)
![Screenshot 2025-06-22 145814](https://github.com/user-attachments/assets/1ed9969b-473e-4734-a639-c849fa8b899f)
![Screenshot 2025-06-22 192953](https://github.com/user-attachments/assets/6a8549ea-4722-44c6-91ae-8f62004de698)
![Screenshot 2025-06-22 193002](https://github.com/user-attachments/assets/a79cd4da-7652-4d82-9563-1c27d03520f7)
![Screenshot 2025-06-22 193019](https://github.com/user-attachments/assets/d6ecd698-537d-4918-97ec-bc02c1f6927e)
![Screenshot 2025-06-22 193027](https://github.com/user-attachments/assets/49c9f33f-f3e9-4ded-a4cd-6e328ed49f19)
![Screenshot 2025-06-22 193037](https://github.com/user-attachments/assets/b0332872-00a7-44e6-b911-fc7397c92fde)
![Screenshot 2025-06-22 193047](https://github.com/user-attachments/assets/3545d936-888e-4704-9f4c-7320aa4b06fb)
![Screenshot 2025-06-22 193055](https://github.com/user-attachments/assets/2a34414d-099b-48ca-ac52-59dc3aab940c)
![Screenshot 2025-06-22 200952](https://github.com/user-attachments/assets/8fe4363c-74b6-43ff-833b-6644f7e0dc58)






