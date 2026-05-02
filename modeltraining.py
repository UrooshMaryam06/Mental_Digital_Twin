import pandas as pd
import numpy as np
import pickle
import warnings
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, LabelEncoder, normalize
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

warnings.filterwarnings('ignore')

# STEP 1: LOAD THE DATA
df = pd.read_csv('energy_dataset.csv')
df['time'] = pd.to_datetime(df['time'], utc=True)
df = df.sort_values('time').set_index('time')

print("\n[STEP 5] Creating generation features...")
df['renewable'] = (df['generation solar'].fillna(0) + df['generation wind onshore'].fillna(0) + 
                  df['generation hydro run-of-river and poundage'].fillna(0) + 
                  df['generation hydro water reservoir'].fillna(0) + 
                  df['generation hydro pumped storage consumption'].fillna(0) + 
                  df['generation biomass'].fillna(0) + df['generation other renewable'].fillna(0))

df['fossil'] = (df['generation fossil gas'].fillna(0) + df['generation fossil hard coal'].fillna(0) + 
               df['generation fossil brown coal/lignite'].fillna(0) + df['generation fossil oil'].fillna(0))

df['nuclear'] = df['generation nuclear'].fillna(0)
df['other'] = df['generation other'].fillna(0) + df['generation waste'].fillna(0)
df['total_gen'] = df['renewable'] + df['fossil'] + df['nuclear'] + df['other']

# STEP 2: DROP USELESS COLUMNS
columns_to_drop = [
    'generation hydro pumped storage aggregated', 'generation geothermal', 'generation marine',
    'forecast wind offshore day ahead', 'generation other', 'generation other renewable',
    'generation fossil oil', 'generation fossil oil shale', 'generation fossil peat',
    'generation wind offshore', 'generation fossil coal-derived gas', 'forecast wind offshore eday ahead'
]
df = df.drop(columns=columns_to_drop, errors='ignore')

# STEP 3: CREATE TIME FEATURES
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# STEP 4: CREATE LAG FEATURES
df['demand_lag_1h'] = df['total load actual'].shift(1)
df['demand_lag_24h'] = df['total load actual'].shift(24)
df['demand_lag_168h'] = df['total load actual'].shift(168)
df['price_lag_1h'] = df['price actual'].shift(1)
df['price_lag_24h'] = df['price actual'].shift(24)

# STEP 6 & 7: RATIOS AND ROLLING
df['renewable_pct'] = np.where(df['total_gen'] > 0, (df['renewable'] / df['total_gen']) * 100, 0)
df['demand_avg_24h'] = df['total load actual'].rolling(window=24, min_periods=1).mean().shift(1)
df['price_avg_24h'] = df['price actual'].rolling(window=24, min_periods=1).mean().shift(1)

# CLEANUP
df = df.dropna(subset=['demand_lag_168h', 'price_lag_24h']).copy()
df = df.drop(df.columns[(df == 0).all()], axis=1)

# STEP 10: FEATURE SELECTION
# !!! IMPORTANT: These must match exactly in training and prediction !!!
demand_features = [
    'hour', 'day_of_week', 'month', 'is_weekend', 'hour_sin', 'hour_cos',
    'demand_lag_1h', 'demand_lag_24h', 'demand_lag_168h',
    'price_lag_1h', 'price_lag_24h', 'renewable', 'fossil', 'nuclear',
    'renewable_pct', 'demand_avg_24h', 'price_avg_24h',
    'forecast wind onshore day ahead', 'forecast solar day ahead', 'total load forecast'
]

price_features = [
    'hour', 'day_of_week', 'month', 'is_weekend', 'hour_sin', 'hour_cos',
    'price_lag_1h', 'price_lag_24h', 'demand_lag_1h', 'demand_lag_24h',
    'forecast wind onshore day ahead', 'forecast solar day ahead',
    'renewable', 'fossil', 'nuclear', 'renewable_pct',
    'price_avg_24h', 'demand_avg_24h', 'total load forecast', 'price day ahead'
]

# FINAL PURGE
all_required = list(set(demand_features + price_features + ['total load actual', 'price actual']))
df = df.dropna(subset=all_required).copy()
print(f"Purged data. Records: {len(df)}")

# TRAIN-TEST SPLIT
split = int(len(df) * 0.8)
X_train_d = df[demand_features].iloc[:split]
X_test_d  = df[demand_features].iloc[split:]
y_train_d = df['total load actual'].iloc[:split]
y_test_d  = df['total load actual'].iloc[split:]

X_train_p = df[price_features].iloc[:split]
X_test_p  = df[price_features].iloc[split:]
y_train_p = df['price actual'].iloc[:split]
y_test_p  = df['price actual'].iloc[split:]

# TRAINING
print("Training Linear Regression...")
lr_demand = LinearRegression().fit(X_train_d, y_train_d)
lr_price = LinearRegression().fit(X_train_p, y_train_p)

print("Training Random Forests...")
rf_demand = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1).fit(X_train_d, y_train_d)
rf_price = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1).fit(X_train_p, y_train_p)

# ... (Keep everything above in modeltraining.py the same) ...
# --- FINAL CORRECTED SAVE SECTION ---
os.makedirs('artifacts', exist_ok=True)

def save(obj, name):
    with open(f'artifacts/{name}', 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved: {name}")

# This was missing in the last snippet!
def make_labels(series, low, high):
    return ['Low' if v < low else 'High' if v > high else 'Medium' for v in series]

# Standard Regression Models
save(lr_demand, 'reg_demand.pkl')
save(lr_price, 'reg_price.pkl')
save(rf_demand, 'clf_demand.pkl') 
save(rf_price, 'clf_price.pkl')
save(demand_features, 'demand_features.pkl')
save(price_features, 'price_features.pkl')

# Thresholds and Labels
d33, d66 = np.percentile(y_train_d, 33), np.percentile(y_train_d, 66)
p33, p66 = np.percentile(y_train_p, 33), np.percentile(y_train_p, 66)
save({'d33': d33, 'd66': d66, 'p33': p33, 'p66': p66}, 'thresholds.pkl')

le_demand = LabelEncoder().fit(['Low', 'Medium', 'High'])
le_price = LabelEncoder().fit(['Low', 'Medium', 'High'])
save(le_demand, 'le_demand.pkl')
save(le_price, 'le_price.pkl')

# Clustering
cluster_data = df[['hour_sin', 'hour_cos', 'renewable_pct', 'demand_lag_1h', 'price_lag_1h']].dropna()
cluster_scaler = RobustScaler().fit(cluster_data)
kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10).fit(cluster_scaler.transform(cluster_data))
save(cluster_scaler, 'cluster_scaler.pkl')
save(kmeans_model, 'kmeans.pkl')

# Profile Data Generation
pred_demand_test = lr_demand.predict(X_test_d)
pred_price_test  = lr_price.predict(X_test_p)
test_clusters = kmeans_model.predict(cluster_scaler.transform(X_test_d[['hour_sin', 'hour_cos', 'renewable_pct', 'demand_lag_1h', 'price_lag_1h']]))

profile_df = pd.DataFrame({
    'pred_demand': pred_demand_test,
    'pred_price': pred_price_test,
    'demand_class_enc': le_demand.transform(make_labels(y_test_d, d33, d66)),
    'price_class_enc': le_price.transform(make_labels(y_test_p, p33, p66)),
    'cluster': test_clusters,
    'hour_sin': X_test_d['hour_sin'].values,
    'hour_cos': X_test_d['hour_cos'].values,
    'month_sin': np.sin(2 * np.pi * X_test_d['month'].values / 12),
    'month_cos': np.cos(2 * np.pi * X_test_d['month'].values / 12),
    'renewable_pct': df.loc[X_test_d.index, 'renewable_pct'].values,
    'is_weekend': X_test_d['is_weekend'].values,
}, index=X_test_d.index)

profile_vectors = profile_df.values.astype(float)
profiles_norm = normalize(profile_vectors, norm='l2')

save(profile_df, 'profile_df.pkl')
save(profiles_norm, 'profiles_norm.pkl')

rec_table = {
    ('Low', 'Low'): 'Low demand and low price: good time to charge EVs.',
    ('High', 'High'): 'Peak demand and high price: minimize usage.',
}
save(rec_table, 'rec_table.pkl')

print("All artifacts generated successfully!")

# ... (inside your existing save section at the bottom)

# This is the specific file your main.py is currently complaining about
save({'Low': 0, 'Medium': 1, 'High': 2}, 'label_map.pkl')

print("All artifacts generated successfully!")