"""
ELECTRICITY PREPROCESSING - SIMPLE VERSION
=========================================================
Load data → Add features → Split → Save
That's it! Easy as 1-2-3
"""

import pandas as pd
import numpy as np
import pickle

print("=" * 60)
print("ELECTRICITY DATA PREPROCESSING")
print("=" * 60)

# ============================================
# STEP 1: LOAD THE DATA
# ============================================

print("\n[STEP 1] Loading data...")

df = pd.read_csv('electricity_data.csv')  # Change filename here
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

print(f"✅ Loaded! {len(df)} rows of data")
print(f"   From: {df['time'].min()}")
print(f"   To: {df['time'].max()}")


# ============================================
# STEP 2: DROP USELESS COLUMNS
# ============================================

print("\n[STEP 2] Cleaning data...")

# These columns are empty or useless - remove them
columns_to_drop = [
    'generation_hydro_pumped_storage_aggregated',
    'generation_geothermal',
    'generation_marine',
    'forecast_wind_offshore_day_ahead'
]

df = df.drop(columns=columns_to_drop, errors='ignore')

print(f"✅ Dropped empty columns")


# ============================================
# STEP 3: CREATE TIME FEATURES
# ============================================

print("\n[STEP 3] Creating time features...")

df['hour'] = df['time'].dt.hour           # 0, 1, 2, ..., 23
df['day_of_week'] = df['time'].dt.dayofweek  # 0=Mon, 1=Tue, ..., 6=Sun
df['month'] = df['time'].dt.month         # 1, 2, ..., 12
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)  # 0=weekday, 1=weekend
df['is_holiday'] = 0  # You can mark holidays manually if needed

print(f"✅ Time features created:")
print(f"   - hour (0-23)")
print(f"   - day_of_week (0-6)")
print(f"   - month (1-12)")
print(f"   - is_weekend (0/1)")
print(f"   - is_holiday (0/1)")


# ============================================
# STEP 4: CREATE LAG FEATURES (Past Values)
# ============================================

print("\n[STEP 4] Creating lag features (past values)...")

# Past demand values
df['demand_lag_1h'] = df['total_load_actual'].shift(1)      # 1 hour ago
df['demand_lag_24h'] = df['total_load_actual'].shift(24)    # Yesterday same hour
df['demand_lag_168h'] = df['total_load_actual'].shift(168)  # Last week

# Past price values
df['price_lag_1h'] = df['price_actual'].shift(1)            # 1 hour ago
df['price_lag_24h'] = df['price_actual'].shift(24)          # Yesterday

print(f"✅ Lag features created:")
print(f"   - demand_lag_1h, demand_lag_24h, demand_lag_168h")
print(f"   - price_lag_1h, price_lag_24h")


# ============================================
# STEP 5: CREATE GENERATION FEATURES
# ============================================

print("\n[STEP 5] Creating generation features...")

# Add all renewable sources together
df['renewable'] = (
    df['generation_solar'].fillna(0) +
    df['generation_wind_onshore'].fillna(0) +
    df['generation_wind_offshore'].fillna(0) +
    df['generation_hydro_run_of_river_and_poundage'].fillna(0) +
    df['generation_hydro_water_reservoir'].fillna(0)
)

# Add all fossil sources together
df['fossil'] = (
    df['generation_fossil_gas'].fillna(0) +
    df['generation_fossil_hard_coal'].fillna(0) +
    df['generation_fossil_brown_coal_lignite'].fillna(0)
)

# Nuclear and biomass
df['nuclear'] = df['generation_nuclear'].fillna(0)
df['biomass'] = df['generation_biomass'].fillna(0)

# Total generation
df['total_gen'] = df['renewable'] + df['fossil'] + df['nuclear'] + df['biomass']

print(f"✅ Generation features created:")
print(f"   - renewable (solar + wind + hydro)")
print(f"   - fossil (gas + coal)")
print(f"   - nuclear")
print(f"   - biomass")
print(f"   - total_gen")


# ============================================
# STEP 6: CREATE SIMPLE RATIOS
# ============================================

print("\n[STEP 6] Creating ratio features...")

# What percentage is renewable?
df['renewable_pct'] = np.where(
    df['total_gen'] > 0,
    (df['renewable'] / df['total_gen']) * 100,
    0
)

print(f"✅ Ratio features created:")
print(f"   - renewable_pct (% of renewable energy)")


# ============================================
# STEP 7: CREATE ROLLING AVERAGES
# ============================================

print("\n[STEP 7] Creating rolling averages (last 24 hours)...")

# Average demand over last 24 hours
df['demand_avg_24h'] = df['total_load_actual'].rolling(window=24, min_periods=1).mean()

# Average price over last 24 hours
df['price_avg_24h'] = df['price_actual'].rolling(window=24, min_periods=1).mean()

print(f"✅ Rolling averages created:")
print(f"   - demand_avg_24h")
print(f"   - price_avg_24h")


# ============================================
# STEP 8: REMOVE EMPTY ROWS
# ============================================

print("\n[STEP 8] Cleaning empty rows...")

# From lag features, first few rows will be empty
rows_before = len(df)
df = df.dropna()
rows_removed = rows_before - len(df)

print(f"✅ Removed {rows_removed} empty rows")
print(f"   Remaining: {len(df)} rows")


# ============================================
# STEP 9: CLIP EXTREME VALUES
# ============================================

print("\n[STEP 9] Removing extreme outliers...")

# Remove top 1% and bottom 1% of extreme values
demand_lower = df['total_load_actual'].quantile(0.01)
demand_upper = df['total_load_actual'].quantile(0.99)
df['total_load_actual'] = df['total_load_actual'].clip(demand_lower, demand_upper)

price_lower = df['price_actual'].quantile(0.01)
price_upper = df['price_actual'].quantile(0.99)
df['price_actual'] = df['price_actual'].clip(price_lower, price_upper)

print(f"✅ Outliers clipped:")
print(f"   Demand: {demand_lower:.0f} - {demand_upper:.0f} MW")
print(f"   Price: €{price_lower:.2f} - €{price_upper:.2f}/MWh")


# ============================================
# STEP 10: SELECT FEATURES FOR MODELS
# ============================================

print("\n[STEP 10] Selecting features...")

# Features to use for predicting DEMAND
demand_features = [
    'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday',
    'demand_lag_1h', 'demand_lag_24h', 'demand_lag_168h',
    'price_lag_1h', 'price_lag_24h',
    'renewable', 'fossil', 'nuclear',
    'renewable_pct',
    'demand_avg_24h', 'price_avg_24h'
]

# Features to use for predicting PRICE
price_features = [
    'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday',
    'price_lag_1h', 'price_lag_24h',
    'demand_lag_1h', 'demand_lag_24h',
    'renewable', 'fossil', 'nuclear',
    'renewable_pct',
    'price_avg_24h', 'demand_avg_24h'
]

print(f"✅ Features selected:")
print(f"   Demand model: {len(demand_features)} features")
print(f"   Price model: {len(price_features)} features")


# ============================================
# STEP 11: SPLIT INTO TRAIN & TEST
# ============================================

print("\n[STEP 11] Splitting into train & test...")

# Use first 80% for training, last 20% for testing
split_point = int(len(df) * 0.8)

# DEMAND DATA
X_demand_train = df[:split_point][demand_features]
y_demand_train = df[:split_point]['total_load_actual']

X_demand_test = df[split_point:][demand_features]
y_demand_test = df[split_point:]['total_load_actual']

# PRICE DATA
X_price_train = df[:split_point][price_features]
y_price_train = df[:split_point]['price_actual']

X_price_test = df[split_point:][price_features]
y_price_test = df[split_point:]['price_actual']

print(f"✅ Train-Test Split:")
print(f"   Training: {len(X_demand_train)} rows ({split_point} to {df['time'].iloc[split_point-1]})")
print(f"   Testing:  {len(X_demand_test)} rows ({df['time'].iloc[split_point]} to {df['time'].iloc[-1]})")


# ============================================
# STEP 12: VERIFY DATA
# ============================================

print("\n[STEP 12] Verifying data quality...")

print(f"✅ Demand:")
print(f"   Train: {y_demand_train.min():.0f} - {y_demand_train.max():.0f} MW (avg: {y_demand_train.mean():.0f})")
print(f"   Test:  {y_demand_test.min():.0f} - {y_demand_test.max():.0f} MW (avg: {y_demand_test.mean():.0f})")

print(f"✅ Price:")
print(f"   Train: €{y_price_train.min():.2f} - €{y_price_train.max():.2f}/MWh (avg: €{y_price_train.mean():.2f})")
print(f"   Test:  €{y_price_test.min():.2f} - €{y_price_test.max():.2f}/MWh (avg: €{y_price_test.mean():.2f})")

print(f"✅ No missing values:")
print(f"   X_demand_train: {X_demand_train.isnull().sum().sum()}")
print(f"   X_demand_test: {X_demand_test.isnull().sum().sum()}")


# ============================================
# STEP 13: SAVE EVERYTHING
# ============================================

print("\n[STEP 13] Saving files...")

# Save train-test data
data = {
    'X_demand_train': X_demand_train,
    'y_demand_train': y_demand_train,
    'X_demand_test': X_demand_test,
    'y_demand_test': y_demand_test,
    'X_price_train': X_price_train,
    'y_price_train': y_price_train,
    'X_price_test': X_price_test,
    'y_price_test': y_price_test,
}

pickle.dump(data, open('preprocessed_data.pkl', 'wb'))

# Save feature names
features = {
    'demand_features': demand_features,
    'price_features': price_features,
}

pickle.dump(features, open('features.pkl', 'wb'))

print(f"✅ Saved:")
print(f"   - preprocessed_data.pkl (train/test data)")
print(f"   - features.pkl (feature names)")


# ============================================
# DONE!
# ============================================

print("\n" + "=" * 60)
print("✅ PREPROCESSING COMPLETE!")
print("=" * 60)
print(f"\nYour data is ready for training!")
print(f"\nTo use in your model:")
print(f"  import pickle")
print(f"  data = pickle.load(open('preprocessed_data.pkl', 'rb'))")
print(f"  X_train = data['X_demand_train']")
print(f"  y_train = data['y_demand_train']")
print(f"\nThen train XGBoost!")
print("=" * 60)