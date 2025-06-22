# load_data_dmart.py

import os
import numpy as np
import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocess_dmart import load_and_clean_dmart_data

# ==============================
# 🔹 Load & Preprocess D-Mart Data
# ==============================
df = load_and_clean_dmart_data() 

# 🔹 Define Categorical Features
categorical = ['City', 'Customer type', 'Gender', 'Product line', 'Payment']

# 🔹 One-hot Encoding
df = pd.get_dummies(df, columns=categorical, drop_first=False)

# 🔹 Feature & Target Columns
X = df[['Quantity', 'Unit price', 'gross income', 'Rating', 'Hour', 'Weekday', 'Is Weekend',
        'Month', 'First Half', 'Month Start', 'Month End'] +
       [col for col in df.columns if col.startswith(tuple([f"{c}_" for c in categorical]))]]

y = np.log1p(df['Total'])  # Target: log-transformed Total Sales

# 🔹 Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================
# 🔹 Train XGBoost Model
# ==============================
model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.9, colsample_bytree=0.9, random_state=42)
model.fit(X_train, y_train)

# ==============================
# 🔹 Evaluate
# ==============================
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_actual = np.expm1(y_test)

mae = mean_absolute_error(y_actual, y_pred)
mse = mean_squared_error(y_actual, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_actual, y_pred)

print("✅ D-Mart Sales Model Trained")
print(f"📊 MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")

# ==============================
# 🔹 Save Model & Features
# ==============================
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/sales_predictor.pkl")
joblib.dump(X.columns.tolist(), "models/feature_names.pkl")
print("📦 Model & Feature Names Saved in 'models/'")

# ==============================
# 🔹 Save Test Data for Dashboard
# ==============================
test_df = X_test.copy()
test_df["Actual Sales"] = y_actual.values
test_df.to_csv("models/test_data.csv", index=False)
print("📊 Test data with actual sales saved to 'models/test_data.csv'")

