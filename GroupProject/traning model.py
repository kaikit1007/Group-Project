import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

print("Starting AI Training Pipeline with 80:20 Split & Size/Area Integration...")

# Load Micro and Macro data
df_micro = pd.read_csv("malaysia_housing_2025.csv")
df_macro = pd.read_csv("Final_Training_Data.csv")

# Clean column names
df_micro.columns = df_micro.columns.str.strip().str.lower()
df_micro = df_micro.dropna(subset=["state", "tenure", "type", "median_price"])

# Calculate global macro growth anchor from the Excel data
try:
    growth_anchor = pd.to_numeric(df_macro.iloc[:, 3], errors='coerce').mean() / 100
except Exception:
    growth_anchor = 0.035

# 1. Feature Engineering (Match main.py logic exactly)
if 'median_psf' not in df_micro.columns:
    df_micro['median_psf'] = df_micro['median_price'] / 1200

if 'township' in df_micro.columns:
    df_micro['urban_rural'] = np.where(
        df_micro['township'].astype(str).str.contains('kampung|felda|kg', case=False, na=False), 'Rural', 'Urban')
else:
    df_micro['urban_rural'] = 'Urban'

df_micro['built_up_sqft'] = np.where(df_micro["median_psf"] > 0, df_micro["median_price"] / df_micro["median_psf"], 1000.0)

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

# 2. Encoding Categorical Features
le_state = LabelEncoder()
le_type = LabelEncoder()
le_tenure = LabelEncoder()
le_urban = LabelEncoder()

df_micro['state_encoded'] = le_state.fit_transform(df_micro['state'].astype(str))
df_micro['type_encoded'] = le_type.fit_transform(df_micro['simplified_type'].astype(str))
df_micro['tenure_encoded'] = le_tenure.fit_transform(df_micro['simplified_tenure'].astype(str))
df_micro['urban_encoded'] = le_urban.fit_transform(df_micro['urban_rural'].astype(str))

# Save all encoders for main.py
joblib.dump(le_state, "le_state.pkl")
joblib.dump(le_type, "le_type.pkl")
joblib.dump(le_tenure, "le_tenure.pkl")
joblib.dump(le_urban, "le_urban.pkl")

# 3. Define the 7 core features
features = ['state', 'type', 'tenure', 'urban_rural', 'built_up_sqft', 'macro_trend', 'month_offset']
X = pd.DataFrame({
    'state': df_micro['state_encoded'],
    'type': df_micro['type_encoded'],
    'tenure': df_micro['tenure_encoded'],
    'urban_rural': df_micro['urban_encoded'],
    'built_up_sqft': df_micro['built_up_sqft'],
    'macro_trend': df_micro['macro_trend'],
    'month_offset': df_micro['month_offset']
})
y = df_micro['median_price']

# 4. Perform 80:20 Train-Test Split
print("\nPerforming 80:20 Train-Test Split...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training Data: {len(X_train)} rows | Testing Data: {len(X_test)} rows\n")

print("Training all 3 core AI engines and evaluating performance...")

# Train & Evaluate Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
print(f"Random Forest  -> Accuracy (R²): {r2_score(y_test, rf_preds) * 100:.2f}% | MAE: RM {mean_absolute_error(y_test, rf_preds):,.2f}")
joblib.dump(rf_model, "rf_model.pkl")

# Train & Evaluate XGBoost
xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
print(f"XGBoost        -> Accuracy (R²): {r2_score(y_test, xgb_preds) * 100:.2f}% | MAE: RM {mean_absolute_error(y_test, xgb_preds):,.2f}")
joblib.dump(xgb_model, "xgb_model.pkl")

# Train & Evaluate Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
print(f"Linear Reg     -> Accuracy (R²): {r2_score(y_test, lr_preds) * 100:.2f}% | MAE: RM {mean_absolute_error(y_test, lr_preds):,.2f}")
joblib.dump(lr_model, "lr_model.pkl")

print("\n✅ Success! All models trained and saved.")