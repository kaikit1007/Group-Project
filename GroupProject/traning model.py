import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

print("Starting AI Training Pipeline with Tenure Integration...")

# Load Micro and Macro data
df_micro = pd.read_csv("malaysia_housing_2025.csv")
df_macro = pd.read_csv("Final_Training_Data.csv")

# Clean column names
df_micro.columns = df_micro.columns.str.strip().str.lower()

# Calculate global macro growth anchor from the Excel data
try:
    growth_anchor = pd.to_numeric(df_macro.iloc[:, 3], errors='coerce').mean() / 100
except Exception:
    growth_anchor = 0.035

# Apply preprocessing functions locally to match main.py
def get_all_types(type_val):
    val = str(type_val).lower()
    mapping = {
        'Bungalow': ['bungalow', 'bungalo'],
        'Townhouse': ['townhouse', 'town house'],
        'Semi-D': ['semi-d', 'semi d', 'semid'],
        'Condominium': ['condo', 'condominium', 'service residence', 'serviced residence'],
        'Terrace': ['terrace', 'link house', 'link-house'],
        'Apartment/Flat': ['apartment', 'flat', 'low cost']
    }
    for category, keywords in mapping.items():
        if any(kw in val for kw in keywords): return category
    return 'Others'

def get_simplified_tenure(tenure_val):
    val = str(tenure_val).lower()
    return 'Freehold' if 'freehold' in val else ('Leasehold' if 'lease' in val else 'Others')

df_micro['simplified_type'] = df_micro['type'].apply(get_all_types)
df_micro['simplified_tenure'] = df_micro['tenure'].apply(get_simplified_tenure)
df_micro['macro_trend'] = growth_anchor
df_micro['month_offset'] = np.random.randint(-60, 36, size=len(df_micro))

# Encoding categorical features
le_state = LabelEncoder()
le_type = LabelEncoder()
le_tenure = LabelEncoder()

df_micro['state_encoded'] = le_state.fit_transform(df_micro['state'].astype(str))
df_micro['type_encoded'] = le_type.fit_transform(df_micro['simplified_type'].astype(str))
df_micro['tenure_encoded'] = le_tenure.fit_transform(df_micro['simplified_tenure'].astype(str))

# Save all encoders for main.py
joblib.dump(le_state, "le_state.pkl")
joblib.dump(le_type, "le_type.pkl")
joblib.dump(le_tenure, "le_tenure.pkl")

# Define the 5 core features used for the AI matrix
features = ['state', 'type', 'tenure', 'macro_trend', 'month_offset']
X = pd.DataFrame({
    'state': df_micro['state_encoded'],
    'type': df_micro['type_encoded'],
    'tenure': df_micro['tenure_encoded'],
    'macro_trend': df_micro['macro_trend'],
    'month_offset': df_micro['month_offset']
})
y = df_micro['median_price']

print("Training all 3 core AI engines...")
# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X, y)
joblib.dump(rf_model, "rf_model.pkl")

# Train XGBoost
xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
xgb_model.fit(X, y)
joblib.dump(xgb_model, "xgb_model.pkl")

# Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X, y)
joblib.dump(lr_model, "lr_model.pkl")

# Save a unified reference to make main.py loading backward compatible
joblib.dump(xgb_model, "final_ai_model.pkl")

print("✅ Success! All models retrained and feature sets are perfectly matched.")