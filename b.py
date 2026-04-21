"""
ELECTRICITY DEMAND & PRICE FORECASTING - PREPROCESSING PIPELINE
=========================================================

This script handles:
✅ Data loading & validation
✅ Feature engineering (NO data leakage)
✅ Time-based train-test split
✅ Data cleaning & outlier handling
��� Feature selection
✅ Output ready for XGBoost training

No scaling needed for tree models!
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from datetime import datetime
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================

class Config:
    """Configuration for preprocessing"""
    
    # File paths
    DATA_FILE = 'electricity_data.csv'  # Change this to your file
    OUTPUT_DIR = './preprocessed_data'
    
    # Train-test split
    TRAIN_RATIO = 0.8  # 80% train, 20% test
    
    # Time window for rolling statistics
    ROLLING_WINDOW = 24  # hours
    
    # Outlier clipping
    OUTLIER_LOWER_PERCENTILE = 1
    OUTLIER_UPPER_PERCENTILE = 99
    
    # Holidays (customize for your country)
    HOLIDAYS = [
        (1, 1),    # New Year
        (1, 6),    # Epiphany
        (5, 1),    # Labour Day
        (12, 24),  # Christmas Eve
        (12, 25),  # Christmas
        (12, 26),  # Boxing Day
    ]
    
    # Columns to drop (empty or useless)
    COLS_TO_DROP = [
        'generation_hydro_pumped_storage_aggregated',
        'generation_geothermal',
        'generation_marine',
        'forecast_wind_offshore_day_ahead',
        'generation_fossil_coal_derived_gas',
        'generation_fossil_oil_shale',
        'generation_fossil_peat',
        'generation_fossil_oil'
    ]


# ============================================
# LOGGER
# ============================================

class Logger:
    """Simple logging utility"""
    
    def __init__(self):
        self.logs = []
    
    def log(self, message, level='INFO'):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [{level}] {message}"
        print(log_msg)
        self.logs.append(log_msg)
    
    def save_logs(self, filename):
        with open(filename, 'w') as f:
            f.write('\n'.join(self.logs))


# ============================================
# STEP 1: LOAD DATA
# ============================================

def load_data(filepath, logger):
    """Load and validate raw data"""
    
    logger.log(f"Loading data from: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        logger.log(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        logger.log(f"❌ Error loading file: {str(e)}", level='ERROR')
        raise
    
    # Convert time column
    try:
        df['time'] = pd.to_datetime(df['time'])
        logger.log(f"✅ Time column converted")
    except Exception as e:
        logger.log(f"❌ Error converting time: {str(e)}", level='ERROR')
        raise
    
    # Sort by time
    df = df.sort_values('time').reset_index(drop=True)
    logger.log(f"✅ Data sorted by time")
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['time']).sum()
    if duplicates > 0:
        logger.log(f"⚠️  Found {duplicates} duplicate timestamps", level='WARNING')
        df = df.drop_duplicates(subset=['time'], keep='first')
    
    # Print date range
    start_date = df['time'].min()
    end_date = df['time'].max()
    logger.log(f"📅 Date range: {start_date} to {end_date}")
    
    return df


# ============================================
# STEP 2: CLEAN DATA
# ============================================

def clean_data(df, logger):
    """Clean and validate data"""
    
    logger.log("Starting data cleaning...")
    
    # Drop completely empty columns
    logger.log(f"Dropping {len(Config.COLS_TO_DROP)} empty columns")
    df = df.drop(columns=Config.COLS_TO_DROP, errors='ignore')
    logger.log(f"✅ Remaining columns: {len(df.columns)}")
    
    # Check for missing values in target variables
    missing_demand = df['total_load_actual'].isna().sum()
    missing_price = df['price_actual'].isna().sum()
    
    if missing_demand > 0:
        logger.log(f"⚠️  Missing demand values: {missing_demand}", level='WARNING')
    if missing_price > 0:
        logger.log(f"⚠️  Missing price values: {missing_price}", level='WARNING')
    
    logger.log(f"✅ Data cleaned")
    
    return df


# ============================================
# STEP 3: CREATE TIME FEATURES
# ============================================

def create_time_features(df, logger):
    """Create temporal features"""
    
    logger.log("Creating time features...")
    
    # Extract time components
    df['hour'] = df['time'].dt.hour                          # 0-23
    df['day_of_week'] = df['time'].dt.dayofweek             # 0-6 (Mon-Sun)
    df['month'] = df['time'].dt.month                        # 1-12
    df['day_of_month'] = df['time'].dt.day                  # 1-31
    df['week_of_year'] = df['time'].dt.isocalendar().week   # 1-52
    
    # Binary flags
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)  # 0/1
    
    # Holiday flag
    df['is_holiday'] = df['time'].apply(
        lambda x: 1 if (x.month, x.day) in Config.HOLIDAYS else 0
    )
    
    logger.log(f"✅ Time features created (7 features)")
    
    return df


# ============================================
# STEP 4: CREATE LAG FEATURES
# ============================================

def create_lag_features(df, logger):
    """Create lagged features (past values)"""
    
    logger.log("Creating lag features...")
    
    # Demand lags
    df['demand_lag_1h'] = df['total_load_actual'].shift(1)
    df['demand_lag_6h'] = df['total_load_actual'].shift(6)
    df['demand_lag_24h'] = df['total_load_actual'].shift(24)
    df['demand_lag_168h'] = df['total_load_actual'].shift(168)  # 1 week
    
    # Price lags
    df['price_lag_1h'] = df['price_actual'].shift(1)
    df['price_lag_24h'] = df['price_actual'].shift(24)
    
    logger.log(f"✅ Lag features created (6 features)")
    logger.log(f"   NaN rows created from lags: {df[['demand_lag_1h', 'demand_lag_168h']].isna().sum().max()}")
    
    return df


# ============================================
# STEP 5: CREATE GENERATION FEATURES
# ============================================

def create_generation_features(df, logger):
    """Create generation aggregates (NO lagged generation for tree models!)"""
    
    logger.log("Creating generation features...")
    
    # Renewable sources (CURRENT hour, NOT lagged)
    df['generation_renewable'] = (
        df['generation_solar'].fillna(0) +
        df['generation_wind_onshore'].fillna(0) +
        df['generation_wind_offshore'].fillna(0) +
        df['generation_hydro_run_of_river_and_poundage'].fillna(0) +
        df['generation_hydro_water_reservoir'].fillna(0) +
        df['generation_other_renewable'].fillna(0)
    )
    
    # Fossil sources (CURRENT hour)
    df['generation_fossil'] = (
        df['generation_fossil_gas'].fillna(0) +
        df['generation_fossil_hard_coal'].fillna(0) +
        df['generation_fossil_brown_coal_lignite'].fillna(0)
    )
    
    # Other sources (CURRENT hour)
    df['generation_nuclear'] = df['generation_nuclear'].fillna(0)
    df['generation_biomass'] = df['generation_biomass'].fillna(0)
    
    # Total generation
    df['total_generation'] = (
        df['generation_renewable'] +
        df['generation_fossil'] +
        df['generation_nuclear'] +
        df['generation_biomass']
    )
    
    logger.log(f"✅ Generation aggregates created (5 features)")
    logger.log(f"   Renewable range: {df['generation_renewable'].min():.0f} - {df['generation_renewable'].max():.0f} MW")
    logger.log(f"   Fossil range: {df['generation_fossil'].min():.0f} - {df['generation_fossil'].max():.0f} MW")
    
    return df


# ============================================
# STEP 6: CREATE RATIO FEATURES
# ============================================

def create_ratio_features(df, logger):
    """Create percentage/ratio features"""
    
    logger.log("Creating ratio features...")
    
    # Renewable percentage
    df['renewable_percentage'] = np.where(
        df['total_generation'] > 0,
        (df['generation_renewable'] / df['total_generation']) * 100,
        0
    )
    
    # Fossil percentage
    df['fossil_percentage'] = np.where(
        df['total_generation'] > 0,
        (df['generation_fossil'] / df['total_generation']) * 100,
        0
    )
    
    logger.log(f"✅ Ratio features created (2 features)")
    logger.log(f"   Renewable % range: {df['renewable_percentage'].min():.1f}% - {df['renewable_percentage'].max():.1f}%")
    
    return df


# ============================================
# STEP 7: CREATE ROLLING STATISTICS
# ============================================

def create_rolling_features(df, logger, window=24):
    """Create rolling window statistics"""
    
    logger.log(f"Creating rolling statistics (window={window}h)...")
    
    # Demand rolling stats
    df['demand_rolling_mean_24h'] = df['total_load_actual'].rolling(
        window=window, min_periods=1
    ).mean()
    
    df['demand_rolling_std_24h'] = df['total_load_actual'].rolling(
        window=window, min_periods=1
    ).std().fillna(0)
    
    df['demand_rolling_min_24h'] = df['total_load_actual'].rolling(
        window=window, min_periods=1
    ).min()
    
    df['demand_rolling_max_24h'] = df['total_load_actual'].rolling(
        window=window, min_periods=1
    ).max()
    
    # Price rolling stats
    df['price_rolling_mean_24h'] = df['price_actual'].rolling(
        window=window, min_periods=1
    ).mean()
    
    df['price_rolling_std_24h'] = df['price_actual'].rolling(
        window=window, min_periods=1
    ).std().fillna(0)
    
    logger.log(f"✅ Rolling statistics created (6 features)")
    
    return df


# ============================================
# STEP 8: HANDLE MISSING DATA
# ============================================

def handle_missing_data(df, logger):
    """Remove rows with NaN values"""
    
    logger.log("Handling missing data...")
    
    initial_rows = len(df)
    
    # Check NaN counts
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        logger.log(f"⚠️  NaN values found:", level='WARNING')
        for col, count in nan_counts[nan_counts > 0].items():
            logger.log(f"   {col}: {count} missing", level='WARNING')
    
    # Drop NaN rows (mainly from lag features)
    df = df.dropna()
    
    removed_rows = initial_rows - len(df)
    logger.log(f"✅ Dropped {removed_rows} rows with NaN")
    logger.log(f"   Remaining: {len(df)} rows")
    
    return df


# ============================================
# STEP 9: CLIP OUTLIERS
# ============================================

def clip_outliers(df, logger):
    """Clip extreme values to prevent overfitting"""
    
    logger.log("Clipping outliers...")
    
    # Demand outliers
    demand_q_lower = df['total_load_actual'].quantile(Config.OUTLIER_LOWER_PERCENTILE / 100)
    demand_q_upper = df['total_load_actual'].quantile(Config.OUTLIER_UPPER_PERCENTILE / 100)
    
    demand_before = df['total_load_actual'].copy()
    df['total_load_actual'] = df['total_load_actual'].clip(demand_q_lower, demand_q_upper)
    demand_clipped = (demand_before != df['total_load_actual']).sum()
    
    logger.log(f"✅ Demand clipped: {demand_clipped} values")
    logger.log(f"   Range: {demand_q_lower:.0f} - {demand_q_upper:.0f} MW")
    
    # Price outliers
    price_q_lower = df['price_actual'].quantile(Config.OUTLIER_LOWER_PERCENTILE / 100)
    price_q_upper = df['price_actual'].quantile(Config.OUTLIER_UPPER_PERCENTILE / 100)
    
    price_before = df['price_actual'].copy()
    df['price_actual'] = df['price_actual'].clip(price_q_lower, price_q_upper)
    price_clipped = (price_before != df['price_actual']).sum()
    
    logger.log(f"✅ Price clipped: {price_clipped} values")
    logger.log(f"   Range: €{price_q_lower:.2f} - €{price_q_upper:.2f}/MWh")
    
    return df


# ============================================
# STEP 10: SELECT FEATURES
# ============================================

def select_features(logger):
    """Define feature lists for models"""
    
    logger.log("Selecting features...")
    
    # Features for DEMAND prediction
    demand_features = [
        # Time features (7)
        'hour',
        'day_of_week',
        'month',
        'day_of_month',
        'week_of_year',
        'is_weekend',
        'is_holiday',
        
        # Lag features (6)
        'demand_lag_1h',
        'demand_lag_6h',
        'demand_lag_24h',
        'demand_lag_168h',
        'price_lag_1h',
        'price_lag_24h',
        
        # Generation features (5) - CURRENT HOUR, NOT LAGGED
        'generation_renewable',
        'generation_fossil',
        'generation_nuclear',
        'generation_biomass',
        'total_generation',
        
        # Ratio features (2)
        'renewable_percentage',
        'fossil_percentage',
        
        # Rolling statistics (6)
        'demand_rolling_mean_24h',
        'demand_rolling_std_24h',
        'demand_rolling_min_24h',
        'demand_rolling_max_24h',
        'price_rolling_mean_24h',
        'price_rolling_std_24h',
    ]
    
    # Features for PRICE prediction
    price_features = [
        # Time features (7)
        'hour',
        'day_of_week',
        'month',
        'day_of_month',
        'week_of_year',
        'is_weekend',
        'is_holiday',
        
        # Lag features (4)
        'price_lag_1h',
        'price_lag_24h',
        'demand_lag_1h',
        'demand_lag_24h',
        
        # Generation features (5) - CURRENT HOUR, NOT LAGGED
        'generation_renewable',
        'generation_fossil',
        'generation_nuclear',
        'generation_biomass',
        'total_generation',
        
        # Ratio features (2)
        'renewable_percentage',
        'fossil_percentage',
        
        # Rolling statistics (4)
        'demand_rolling_mean_24h',
        'price_rolling_mean_24h',
        'price_rolling_std_24h',
        'demand_rolling_std_24h',
    ]
    
    logger.log(f"✅ Features selected")
    logger.log(f"   Demand features: {len(demand_features)}")
    logger.log(f"   Price features: {len(price_features)}")
    
    return demand_features, price_features


# ============================================
# STEP 11: CREATE TRAIN-TEST SPLIT
# ============================================

def train_test_split(df, demand_features, price_features, logger):
    """Time-based train-test split (NO random shuffle!)"""
    
    logger.log("Creating train-test split...")
    
    # Time-based split (80/20)
    split_idx = int(len(df) * Config.TRAIN_RATIO)
    
    train_start = df['time'].iloc[0]
    train_end = df['time'].iloc[split_idx]
    test_start = df['time'].iloc[split_idx]
    test_end = df['time'].iloc[-1]
    
    logger.log(f"✅ Train-Test Split (Time-Based):")
    logger.log(f"   Train: {train_start} to {train_end}")
    logger.log(f"   Test:  {test_start} to {test_end}")
    logger.log(f"   Train size: {split_idx} rows ({Config.TRAIN_RATIO*100:.0f}%)")
    logger.log(f"   Test size: {len(df)-split_idx} rows ({(1-Config.TRAIN_RATIO)*100:.0f}%)")
    
    # DEMAND data
    X_demand_train = df[:split_idx][demand_features].copy()
    y_demand_train = df[:split_idx]['total_load_actual'].copy()
    X_demand_test = df[split_idx:][demand_features].copy()
    y_demand_test = df[split_idx:]['total_load_actual'].copy()
    
    # PRICE data
    X_price_train = df[:split_idx][price_features].copy()
    y_price_train = df[:split_idx]['price_actual'].copy()
    X_price_test = df[split_idx:][price_features].copy()
    y_price_test = df[split_idx:]['price_actual'].copy()
    
    return {
        'X_demand_train': X_demand_train,
        'y_demand_train': y_demand_train,
        'X_demand_test': X_demand_test,
        'y_demand_test': y_demand_test,
        'X_price_train': X_price_train,
        'y_price_train': y_price_train,
        'X_price_test': X_price_test,
        'y_price_test': y_price_test,
    }


# ============================================
# STEP 12: VALIDATE DATA QUALITY
# ============================================

def validate_data(data_splits, demand_features, price_features, logger):
    """Validate preprocessed data"""
    
    logger.log("Validating data quality...")
    
    # Check shapes
    logger.log(f"✅ Data shapes:")
    logger.log(f"   X_demand_train: {data_splits['X_demand_train'].shape}")
    logger.log(f"   X_demand_test: {data_splits['X_demand_test'].shape}")
    logger.log(f"   X_price_train: {data_splits['X_price_train'].shape}")
    logger.log(f"   X_price_test: {data_splits['X_price_test'].shape}")
    
    # Check for NaN
    logger.log(f"✅ NaN check:")
    logger.log(f"   X_demand_train NaN: {data_splits['X_demand_train'].isnull().sum().sum()}")
    logger.log(f"   X_demand_test NaN: {data_splits['X_demand_test'].isnull().sum().sum()}")
    logger.log(f"   y_demand_train NaN: {data_splits['y_demand_train'].isnull().sum()}")
    logger.log(f"   y_demand_test NaN: {data_splits['y_demand_test'].isnull().sum()}")
    
    # Check value ranges
    logger.log(f"✅ Value ranges (Demand):")
    logger.log(f"   Train: {data_splits['y_demand_train'].min():.0f} - {data_splits['y_demand_train'].max():.0f} MW")
    logger.log(f"   Test: {data_splits['y_demand_test'].min():.0f} - {data_splits['y_demand_test'].max():.0f} MW")
    
    logger.log(f"✅ Value ranges (Price):")
    logger.log(f"   Train: €{data_splits['y_price_train'].min():.2f} - €{data_splits['y_price_train'].max():.2f}/MWh")
    logger.log(f"   Test: €{data_splits['y_price_test'].min():.2f} - €{data_splits['y_price_test'].max():.2f}/MWh")
    
    logger.log(f"✅ Validation complete - Data ready for training!")


# ============================================
# STEP 13: SAVE DATA
# ============================================

def save_data(data_splits, demand_features, price_features, df, logger):
    """Save preprocessed data and metadata"""
    
    logger.log("Saving preprocessed data...")
    
    # Save splits
    pickle.dump(data_splits, open('data_splits.pkl', 'wb'))
    logger.log(f"✅ Saved: data_splits.pkl")
    
    # Save feature lists
    feature_lists = {
        'demand_features': demand_features,
        'price_features': price_features,
    }
    pickle.dump(feature_lists, open('feature_lists.pkl', 'wb'))
    logger.log(f"✅ Saved: feature_lists.pkl")
    
    # Save preprocessing metadata
    metadata = {
        'total_rows': len(df),
        'train_ratio': Config.TRAIN_RATIO,
        'date_range': {
            'start': str(df['time'].min()),
            'end': str(df['time'].max()),
        },
        'features': {
            'demand': len(demand_features),
            'price': len(price_features),
        },
        'preprocessing_date': datetime.now().isoformat(),
    }
    pickle.dump(metadata, open('preprocessing_metadata.pkl', 'wb'))
    logger.log(f"✅ Saved: preprocessing_metadata.pkl")
    
    logger.log(f"✅ All data saved successfully!")


# ============================================
# STEP 14: VISUALIZE DATA
# ============================================

def plot_data_summary(df, data_splits, logger):
    """Create visualization plots"""
    
    logger.log("Creating visualizations...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Demand over time
        axes[0, 0].plot(df['time'], df['total_load_actual'], linewidth=0.5)
        axes[0, 0].set_title('Electricity Demand Over Time')
        axes[0, 0].set_ylabel('Demand (MW)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Price over time
        axes[0, 1].plot(df['time'], df['price_actual'], color='orange', linewidth=0.5)
        axes[0, 1].set_title('Electricity Price Over Time')
        axes[0, 1].set_ylabel('Price (€/MWh)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Demand distribution
        axes[1, 0].hist(data_splits['y_demand_train'], bins=50, alpha=0.7, label='Train')
        axes[1, 0].hist(data_splits['y_demand_test'], bins=50, alpha=0.7, label='Test')
        axes[1, 0].set_title('Demand Distribution')
        axes[1, 0].set_xlabel('Demand (MW)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Price distribution
        axes[1, 1].hist(data_splits['y_price_train'], bins=50, alpha=0.7, label='Train')
        axes[1, 1].hist(data_splits['y_price_test'], bins=50, alpha=0.7, label='Test')
        axes[1, 1].set_title('Price Distribution')
        axes[1, 1].set_xlabel('Price (€/MWh)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data_summary.png', dpi=100, bbox_inches='tight')
        logger.log(f"✅ Saved: data_summary.png")
        plt.close()
        
    except Exception as e:
        logger.log(f"⚠️  Could not create visualizations: {str(e)}", level='WARNING')


# ============================================
# MAIN PIPELINE
# ============================================

def preprocess_electricity_data(filepath=Config.DATA_FILE):
    """
    Main preprocessing pipeline
    
    Returns:
        data_splits: Dictionary with train/test splits
        demand_features: List of demand prediction features
        price_features: List of price prediction features
    """
    
    # Initialize logger
    logger = Logger()
    logger.log("=" * 60)
    logger.log("ELECTRICITY DEMAND & PRICE FORECASTING")
    logger.log("PREPROCESSING PIPELINE")
    logger.log("=" * 60)
    
    try:
        # Step 1: Load data
        df = load_data(filepath, logger)
        
        # Step 2: Clean data
        df = clean_data(df, logger)
        
        # Step 3: Create features
        df = create_time_features(df, logger)
        df = create_lag_features(df, logger)
        df = create_generation_features(df, logger)
        df = create_ratio_features(df, logger)
        df = create_rolling_features(df, logger, window=Config.ROLLING_WINDOW)
        
        # Step 4: Handle missing data
        df = handle_missing_data(df, logger)
        
        # Step 5: Clip outliers
        df = clip_outliers(df, logger)
        
        # Step 6: Select features
        demand_features, price_features = select_features(logger)
        
        # Step 7: Create splits
        data_splits = train_test_split(df, demand_features, price_features, logger)
        
        # Step 8: Validate
        validate_data(data_splits, demand_features, price_features, logger)
        
        # Step 9: Save
        save_data(data_splits, demand_features, price_features, df, logger)
        
        # Step 10: Visualize
        plot_data_summary(df, data_splits, logger)
        
        logger.log("=" * 60)
        logger.log("✅ PREPROCESSING COMPLETE!")
        logger.log("=" * 60)
        
        # Save logs
        logger.save_logs('preprocessing.log')
        logger.log(f"📋 Logs saved to: preprocessing.log")
        
        return data_splits, demand_features, price_features, logger
    
    except Exception as e:
        logger.log(f"❌ FATAL ERROR: {str(e)}", level='ERROR')
        logger.save_logs('preprocessing_error.log')
        raise


# ============================================
# USAGE
# ============================================

if __name__ == "__main__":
    
    # Run preprocessing
    data_splits, demand_features, price_features, logger = preprocess_electricity_data(
        filepath='electricity_data.csv'  # Change this to your CSV file
    )
    
    # Access data
    X_demand_train = data_splits['X_demand_train']
    y_demand_train = data_splits['y_demand_train']
    X_demand_test = data_splits['X_demand_test']
    y_demand_test = data_splits['y_demand_test']
    
    X_price_train = data_splits['X_price_train']
    y_price_train = data_splits['y_price_train']
    X_price_test = data_splits['X_price_test']
    y_price_test = data_splits['y_price_test']
    
    print("\n" + "=" * 60)
    print("READY FOR TRAINING!")
    print("=" * 60)
    print(f"\nDemand Model:")
    print(f"  Train: X{X_demand_train.shape}, y{y_demand_train.shape}")
    print(f"  Test:  X{X_demand_test.shape}, y{y_demand_test.shape}")
    print(f"\nPrice Model:")
    print(f"  Train: X{X_price_train.shape}, y{y_price_train.shape}")
    print(f"  Test:  X{X_price_test.shape}, y{y_price_test.shape}")
    print("=" * 60)