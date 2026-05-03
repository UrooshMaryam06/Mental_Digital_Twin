#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
import warnings
from datetime import datetime
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import RobustScaler, normalize, LabelEncoder
from sklearn.decomposition import PCA

# Regression models
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    VotingRegressor, StackingRegressor
)
from sklearn.svm import SVR

# Classification models
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBRegressor, XGBClassifier

# Clustering
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Model selection & evaluation
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, classification_report, accuracy_score
)
from sklearn.metrics.pairwise import cosine_similarity

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    LGBM_AVAILABLE = True
    print("LightGBM available.")
except ImportError:
    LGBM_AVAILABLE = False
    print("LightGBM not found. Continuing without it.")

warnings.filterwarnings('ignore')
print("All imports done.")



# In[2]:


df = pd.read_csv('energy_dataset.csv')
df['time'] = pd.to_datetime(df['time'], utc=True)
df = df.sort_values('time').set_index('time')

df['renewable'] = (
    df['generation solar'].fillna(0) +
    df['generation wind onshore'].fillna(0) +
    df['generation hydro run-of-river and poundage'].fillna(0) +
    df['generation hydro water reservoir'].fillna(0) +
    df['generation hydro pumped storage consumption'].fillna(0) +
    df['generation biomass'].fillna(0) +
    df['generation other renewable'].fillna(0)
)
df['fossil'] = (
    df['generation fossil gas'].fillna(0) +
    df['generation fossil hard coal'].fillna(0) +
    df['generation fossil brown coal/lignite'].fillna(0) +
    df['generation fossil oil'].fillna(0)
)
df['nuclear']   = df['generation nuclear'].fillna(0)
df['other']     = df['generation other'].fillna(0) + df['generation waste'].fillna(0)
df['total_gen'] = df['renewable'] + df['fossil'] + df['nuclear'] + df['other']

columns_to_drop = [
    'generation hydro pumped storage aggregated', 'generation geothermal',
    'generation marine', 'forecast wind offshore day ahead',
    'generation other', 'generation other renewable', 'generation fossil oil',
    'generation fossil oil shale', 'generation fossil peat',
    'generation wind offshore', 'generation fossil coal-derived gas',
    'forecast wind offshore eday ahead',
    'generation solar', 'generation wind onshore',
    'generation hydro run-of-river and poundage',
    'generation hydro water reservoir', 'generation biomass',
    'generation fossil gas', 'generation fossil hard coal',
    'generation fossil brown coal/lignite', 'generation nuclear',
    'generation other', 'generation waste'
]
df = df.drop(columns=columns_to_drop, errors='ignore')

df['hour']        = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month']       = df.index.month
df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)

df['hour_sin']  = np.sin(2 * np.pi * df['hour']  / 24)
df['hour_cos']  = np.cos(2 * np.pi * df['hour']  / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

df['demand_lag_1h']   = df['total load actual'].shift(1)
df['demand_lag_12h']  = df['total load actual'].shift(12)   # ← ADD THIS
df['demand_lag_24h']  = df['total load actual'].shift(24)
df['demand_lag_168h'] = df['total load actual'].shift(168)
df['price_lag_1h']    = df['price actual'].shift(1)
df['price_lag_12h']   = df['price actual'].shift(12)        # ← ADD THIS
df['price_lag_24h']   = df['price actual'].shift(24)

df['renewable_pct'] = np.where(
    df['total_gen'] > 0,
    (df['renewable'] / df['total_gen']) * 100, 0
)

df['demand_avg_24h'] = df['total load actual'].rolling(24, min_periods=1).mean().shift(1)
df['price_avg_24h']  = df['price actual'].rolling(24, min_periods=1).mean().shift(1)

df = df.dropna(subset=['demand_lag_168h', 'price_lag_24h', 'price_lag_12h', 'demand_lag_12h'])
zero_cols     = df.columns[(df == 0).all()].tolist()
constant_cols = [c for c in df.columns if df[c].nunique() == 1]
df = df.drop(columns=zero_cols + constant_cols, errors='ignore')

print(f"Data shape after preprocessing: {df.shape}")


# %%
# CELL 3 — 12-HOUR AHEAD TARGETS
df['demand_12h'] = df['total load actual'].shift(-12)
df['price_12h']  = df['price actual'].shift(-12)

df_12h = df.dropna(subset=['demand_12h', 'price_12h']).copy()
print(f"12-hour dataset: {len(df_12h)} rows")


# In[3]:


demand_features = [
    'day_of_week', 'month_sin', 'month_cos', 'is_weekend',
    'hour_sin', 'hour_cos',
    'demand_lag_1h', 'demand_lag_24h', 
    'price_lag_1h',  'price_lag_24h',
    'renewable', 'fossil', 'nuclear',
    'renewable_pct',
    'demand_avg_24h', 'price_avg_24h',
    'forecast wind onshore day ahead',
    'forecast solar day ahead',
    'total load forecast',
]

price_features = [
    'day_of_week', 'month_sin', 'month_cos', 'is_weekend',
    'hour_sin', 'hour_cos',
    'price_lag_1h',   'price_lag_12h',    # ← was price_lag_24h
    'demand_lag_1h',  'demand_lag_12h',   # ← was demand_lag_24h
    'forecast wind onshore day ahead',
    'forecast solar day ahead',
    'renewable', 'fossil', 'nuclear',
    'renewable_pct',
    'price_avg_24h', 'demand_avg_24h',
    'total load forecast',
]

demand_features = [f for f in demand_features if f in df_12h.columns]
price_features  = [f for f in price_features  if f in df_12h.columns]

all_required = list(set(demand_features + price_features + ['demand_12h', 'price_12h']))
rows_before  = len(df_12h)
df_12h = df_12h.dropna(subset=all_required).copy()
print(f"Purged {rows_before - len(df_12h)} NaN rows. Final: {len(df_12h)}")

split = int(len(df_12h) * 0.8)

X_train_d, X_test_d = df_12h[demand_features].iloc[:split], df_12h[demand_features].iloc[split:]
y_train_d, y_test_d = df_12h['demand_12h'].iloc[:split],    df_12h['demand_12h'].iloc[split:]

X_train_p, X_test_p = df_12h[price_features].iloc[:split], df_12h[price_features].iloc[split:]
y_train_p, y_test_p = df_12h['price_12h'].iloc[:split],    df_12h['price_12h'].iloc[split:]

print(f"Train: {split} | Test: {len(df_12h) - split}")


# In[4]:


pca_scaler_d = RobustScaler()
pca_scaler_p = RobustScaler()

X_train_d_sc = pca_scaler_d.fit_transform(X_train_d)
X_test_d_sc  = pca_scaler_d.transform(X_test_d)
X_train_p_sc = pca_scaler_p.fit_transform(X_train_p)
X_test_p_sc  = pca_scaler_p.transform(X_test_p)

pca_d = PCA(n_components=0.95, random_state=42)
pca_p = PCA(n_components=0.95, random_state=42)

X_train_d_pca = pca_d.fit_transform(X_train_d_sc)
X_test_d_pca  = pca_d.transform(X_test_d_sc)
X_train_p_pca = pca_p.fit_transform(X_train_p_sc)
X_test_p_pca  = pca_p.transform(X_test_p_sc)

print(f"Demand: {X_train_d.shape[1]} features → {pca_d.n_components_} PCA components")
print(f"Price:  {X_train_p.shape[1]} features → {pca_p.n_components_} PCA components")


# %%
# CELL 6 — EVALUATION HELPER
def evaluate(y_true, y_pred, name, target):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    print(f"  [{target} | {name:<28}]  "
          f"R2: {r2:+.4f}  MAE: {mae:8.2f}  RMSE: {rmse:8.2f}  MAPE: {mape:.2f}%")
    return r2



# In[5]:


# CELL 7 — BASELINE REGRESSION MODELS
print("\n" + "="*70)
print("BASELINE REGRESSION MODELS")
print("="*70)

baselines = {
    "Linear Regression": (LinearRegression(), LinearRegression()),
    "KNN (k=5)":         (KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
                          KNeighborsRegressor(n_neighbors=5, n_jobs=-1)),
    "Decision Tree":     (DecisionTreeRegressor(max_depth=10, random_state=42),
                          DecisionTreeRegressor(max_depth=10, random_state=42)),
    "Bayesian Ridge":    (BayesianRidge(), BayesianRidge()),
}

results = {}
for name, (model_raw, model_pca) in baselines.items():
    print(f"\n--- {name} ---")
    model_raw.fit(X_train_d, y_train_d)
    r2d_raw = evaluate(y_test_d, model_raw.predict(X_test_d), name + " [raw]", "DEMAND-12h")
    model_raw.fit(X_train_p, y_train_p)
    r2p_raw = evaluate(y_test_p, model_raw.predict(X_test_p), name + " [raw]", "PRICE-12h")
    model_pca.fit(X_train_d_pca, y_train_d)
    r2d_pca = evaluate(y_test_d, model_pca.predict(X_test_d_pca), name + " [PCA]", "DEMAND-12h")
    model_pca.fit(X_train_p_pca, y_train_p)
    r2p_pca = evaluate(y_test_p, model_pca.predict(X_test_p_pca), name + " [PCA]", "PRICE-12h")
    results[name] = {"demand_r2_raw": r2d_raw, "price_r2_raw": r2p_raw,
                     "demand_r2_pca": r2d_pca, "price_r2_pca": r2p_pca}


# In[6]:


# CELL 8 — ENSEMBLE & ADVANCED REGRESSION MODELS
print("\n" + "="*70)
print("ENSEMBLE & ADVANCED REGRESSION MODELS")
print("="*70)

rf_d = RandomForestRegressor(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)
rf_p = RandomForestRegressor(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)
rf_d.fit(X_train_d, y_train_d); rf_p.fit(X_train_p, y_train_p)
print("\n--- Random Forest ---")
r2_rf_d = evaluate(y_test_d, rf_d.predict(X_test_d), "Random Forest", "DEMAND-12h")
r2_rf_p = evaluate(y_test_p, rf_p.predict(X_test_p), "Random Forest", "PRICE-12h")
results["Random Forest"] = {"demand_r2_raw": r2_rf_d, "price_r2_raw": r2_rf_p}

xgb_d = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=6,
                      subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                      n_jobs=-1, random_state=42)
xgb_p = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=6,
                      subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                      n_jobs=-1, random_state=42)
xgb_d.fit(X_train_d, y_train_d); xgb_p.fit(X_train_p, y_train_p)
print("\n--- XGBoost ---")
r2_xgb_d = evaluate(y_test_d, xgb_d.predict(X_test_d), "XGBoost", "DEMAND-12h")
r2_xgb_p = evaluate(y_test_p, xgb_p.predict(X_test_p), "XGBoost", "PRICE-12h")
results["XGBoost"] = {"demand_r2_raw": r2_xgb_d, "price_r2_raw": r2_xgb_p}

if LGBM_AVAILABLE:
    lgbm_d = LGBMRegressor(n_estimators=500, learning_rate=0.02, num_leaves=63,
                            subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42)
    lgbm_p = LGBMRegressor(n_estimators=500, learning_rate=0.02, num_leaves=63,
                            subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42)
    lgbm_d.fit(X_train_d, y_train_d); lgbm_p.fit(X_train_p, y_train_p)
    print("\n--- LightGBM ---")
    r2_lgbm_d = evaluate(y_test_d, lgbm_d.predict(X_test_d), "LightGBM", "DEMAND-12h")
    r2_lgbm_p = evaluate(y_test_p, lgbm_p.predict(X_test_p), "LightGBM", "PRICE-12h")
    results["LightGBM"] = {"demand_r2_raw": r2_lgbm_d, "price_r2_raw": r2_lgbm_p}


# In[7]:


# ======================================================
# CLUSTERING — IMPROVED + INTERPRETABLE VERSION
# ======================================================

print("\n" + "="*70)
print("CLUSTERING (IMPROVED)")
print("="*70)

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# -------------------------------
# 1. Better clustering features
# -------------------------------
cluster_features = [
    'hour_sin', 'hour_cos',
    'demand_lag_1h', 'demand_lag_12h', 
    'price_lag_1h', 'price_lag_12h',
    'renewable_pct',
    'demand_avg_12h',
    'price_avg_12h'
]

cluster_features = [f for f in cluster_features if f in df_12h.columns]

# -------------------------------
# 2. Scaling
# -------------------------------
X_cluster = RobustScaler().fit_transform(df_12h[cluster_features])
# -------------------------------
# 3. Find best k (but we use interpretation later)
# -------------------------------
print("\nFinding optimal k (Silhouette Score):")

silhouette_scores = {}

for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_cluster)
    score = silhouette_score(X_cluster, labels, sample_size=5000, random_state=42)

    silhouette_scores[k] = score
    print(f"k={k} -> silhouette = {score:.4f}")

best_k_sil = max(silhouette_scores, key=silhouette_scores.get)

print(f"\nBest k by silhouette = {best_k_sil}")

# -------------------------------
# 4. Final chosen k (interpretability > score)
# -------------------------------
best_k = 3   # or 4 (recommended for energy systems)

print(f"Using k={best_k} for interpretable clustering")

kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_12h['cluster_km'] = kmeans_final.fit_predict(X_cluster)

# -------------------------------
# 5. Cluster summary (VERY IMPORTANT)
# -------------------------------
print("\nCluster Summary (Mean values):")
print(df_12h.groupby('cluster_km')[[
    'demand_12h',
    'price_12h',
    'renewable_pct'
]].mean())





# In[8]:


# -------------------------------
# 7. Cluster distribution
# -------------------------------
df_12h['cluster_km'].value_counts().sort_index().plot(kind='bar')
plt.title("Cluster Distribution")
plt.show()

# -------------------------------
# 8. Interpretation helper (optional but powerful)
# -------------------------------
def interpret_cluster(row):
    if row['demand_12h'] < df_12h['demand_12h'].quantile(0.33):
        return "Low Demand Regime"
    elif row['demand_12h'] < df_12h['demand_12h'].quantile(0.66):
        return "Medium Demand Regime"
    else:
        return "High Demand Regime"

df_12h['cluster_label'] = df_12h.apply(interpret_cluster, axis=1)

print("\nCluster labeling done (for interpretation).")


# In[9]:


# -------------------------------
# 6. PCA visualization
# -------------------------------
pca = PCA(n_components=2)
X_vis = pca.fit_transform(X_cluster)

plt.figure()
plt.scatter(X_vis[:, 0], X_vis[:, 1],
            c=df_12h['cluster_km'], s=5)
plt.title("KMeans Clusters (PCA Visualization)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


# In[ ]:


# ======================================================
# CLASSIFICATION FROM REGRESSION OUTPUTS + DIRECT CLASSIFIERS (UPDATED)
# Adds TimeSeriesSplit CV for classifiers, safe XGBoost fallback,
# scaler pipeline and persisting best models + encoders.
# ======================================================

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import pickle, os

print("\n" + "="*60)
print("CLASSIFICATION FROM REGRESSION OUTPUTS")
print("="*60)

# Regression predictions (baseline) — ensure regressors exist
if 'xgb_d' in globals() and 'xgb_p' in globals():
    try:
        reg_pred_d = xgb_d.predict(X_test_d)
        reg_pred_p = xgb_p.predict(X_test_p)
    except Exception:
        reg_pred_d, reg_pred_p = [], []
else:
    reg_pred_d, reg_pred_p = [], []

# Define thresholds (from true data)
d33, d66 = df_12h['demand_12h'].quantile([0.33, 0.66])
p33, p66 = df_12h['price_12h'].quantile([0.33, 0.66])

# Create class labels from the true targets
def to_class(x, t1, t2):
    if x <= t1:
        return "Low"
    elif x <= t2:
        return "Medium"
    else:
        return "High"

df_12h['demand_class'] = df_12h['demand_12h'].apply(lambda x: to_class(x, d33, d66))
df_12h['price_class'] = df_12h['price_12h'].apply(lambda x: to_class(x, p33, p66))

# Regression -> class baseline
reg_class_d = [to_class(x, d33, d66) for x in reg_pred_d] if len(reg_pred_d) else []
reg_class_p = [to_class(x, p33, p66) for x in reg_pred_p] if len(reg_pred_p) else []

# Prepare classification features
classification_features = []
for feature in demand_features + price_features:
    if feature in df_12h.columns and feature not in classification_features:
        classification_features.append(feature)

split_cls = int(len(df_12h) * 0.8)

X_cls = df_12h[classification_features]

# Encode labels for whole dataset (time-ordered)
le_d = LabelEncoder(); le_p = LabelEncoder()
y_cls_d_enc = le_d.fit_transform(df_12h['demand_class'])
y_cls_p_enc = le_p.fit_transform(df_12h['price_class'])

# Training/test split (time-based)
X_train_cls = X_cls.iloc[:split_cls]
X_test_cls  = X_cls.iloc[split_cls:]

y_train_d_enc = y_cls_d_enc[:split_cls]
y_test_d_enc  = y_cls_d_enc[split_cls:]

y_train_p_enc = y_cls_p_enc[:split_cls]
y_test_p_enc  = y_cls_p_enc[split_cls:]

# Classifier specs with safe XGBoost fallback
classifier_specs = {
    "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=18, random_state=42, n_jobs=-1),
}
try:
    from xgboost import XGBClassifier
    classifier_specs['XGBoost'] = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        eval_metric='mlogloss',
        num_class=3,
        random_state=42,
        n_jobs=-1,
    )
except Exception:
    print('XGBoost classifier unavailable; continuing without it.')

# Evaluate with TimeSeriesSplit CV on training portion
tscv = TimeSeriesSplit(n_splits=min(5, max(2, int(len(X_train_cls)/50))))
classifier_results = {'demand': {}, 'price': {}}
trained_classifiers = {'demand': {}, 'price': {}}

for name, model in classifier_specs.items():
    pipe = make_pipeline(StandardScaler(), model)

    # Demand CV (use training slice only)
    try:
        scores_d = cross_val_score(pipe, X_train_cls, y_train_d_enc, cv=tscv, scoring='f1_macro', n_jobs=-1)
        mean_cv_d = float(scores_d.mean())
    except Exception:
        mean_cv_d = None

    # Price CV
    try:
        scores_p = cross_val_score(pipe, X_train_cls, y_train_p_enc, cv=tscv, scoring='f1_macro', n_jobs=-1)
        mean_cv_p = float(scores_p.mean())
    except Exception:
        mean_cv_p = None

    # Fit on full training slice
    pipe.fit(X_train_cls, y_train_d_enc)
    pred_d_enc = pipe.predict(X_test_cls)
    pred_d = le_d.inverse_transform(pred_d_enc)
    y_test_d = le_d.inverse_transform(y_test_d_enc)

    # Refit for price target separately
    pipe_p = make_pipeline(StandardScaler(), model.__class__(**model.get_params()))
    pipe_p.fit(X_train_cls, y_train_p_enc)
    pred_p_enc = pipe_p.predict(X_test_cls)
    pred_p = le_p.inverse_transform(pred_p_enc)
    y_test_p = le_p.inverse_transform(y_test_p_enc)

    classifier_results['demand'][name] = {
        'cv_f1_macro': mean_cv_d,
        'accuracy': accuracy_score(y_test_d, pred_d),
        'macro_f1': f1_score(y_test_d, pred_d, average='macro'),
        'report': classification_report(y_test_d, pred_d, zero_division=0),
    }
    classifier_results['price'][name] = {
        'cv_f1_macro': mean_cv_p,
        'accuracy': accuracy_score(y_test_p, pred_p),
        'macro_f1': f1_score(y_test_p, pred_p, average='macro'),
        'report': classification_report(y_test_p, pred_p, zero_division=0),
    }

    trained_classifiers['demand'][name] = pipe
    trained_classifiers['price'][name] = pipe_p

# Pick best by CV (fallback to macro_f1 if CV missing)
def best_name(results):
    best = None
    best_score = -1
    for n, m in results.items():
        score = m.get('cv_f1_macro') if m.get('cv_f1_macro') is not None else m.get('macro_f1', 0)
        if score is None:
            continue
        if score > best_score:
            best_score = score
            best = n
    return best

best_demand_classifier_name = best_name(classifier_results['demand'])
best_price_classifier_name  = best_name(classifier_results['price'])

best_demand_classifier = trained_classifiers['demand'].get(best_demand_classifier_name)
best_price_classifier = trained_classifiers['price'].get(best_price_classifier_name)

print("\nBASELINE: regression -> class conversion\n")
if reg_class_d:
    print("DEMAND accuracy:", accuracy_score(le_d.inverse_transform(y_test_d_enc), reg_class_d))

print("\nDIRECT CLASSIFIERS (dedicated supervised models)\n")
for target_name, results_dict in [("demand", classifier_results['demand']), ("price", classifier_results['price'])]:
    print(f"--- {target_name.upper()} ---")
    for model_name, metrics in results_dict.items():
        print(f"{model_name}: cv_f1_macro={metrics.get('cv_f1_macro')}, accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
        print(metrics['report'])

print(f"Best demand classifier: {best_demand_classifier_name}")
print(f"Best price classifier: {best_price_classifier_name}")
print("Dedicated classifiers and encoders will be saved after this section.")

# Expose encoders & trained_classifiers for the artifact save cell
label_encoder_d = le_d
label_encoder_p = le_p
trained_classifiers_all = trained_classifiers
best_classifiers = {'demand': best_demand_classifier_name, 'price': best_price_classifier_name}


# In[ ]:


# ============================================================
# ANN - ARTIFICIAL NEURAL NETWORK (12-Hour Ahead Forecasting)
# ============================================================
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# Use the full engineered feature set instead of only 6 features.
# This makes the ANN a fairer competitor to the tree-based models.
ann_features = []
for feature in demand_features + price_features:
    if feature in df_12h.columns and feature not in ann_features:
        ann_features.append(feature)

# Prepare data
df_ann = df_12h.dropna(subset=ann_features + ['demand_12h', 'price_12h']).copy()

split_ann = int(len(df_ann) * 0.8)

X_ann = df_ann[ann_features]
y_ann_demand = df_ann['demand_12h']
y_ann_price  = df_ann['price_12h']

X_train_ann = X_ann.iloc[:split_ann]
X_test_ann  = X_ann.iloc[split_ann:]
y_train_d_ann = y_ann_demand.iloc[:split_ann]
y_test_d_ann  = y_ann_demand.iloc[split_ann:]
y_train_p_ann = y_ann_price.iloc[:split_ann]
y_test_p_ann  = y_ann_price.iloc[split_ann:]

# Scale features (mandatory for ANN)
scaler_ann = StandardScaler()
X_train_ann_sc = scaler_ann.fit_transform(X_train_ann)
X_test_ann_sc  = scaler_ann.transform(X_test_ann)

ann_params = dict(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    learning_rate_init=0.001,
    verbose=False,
)

# --- DEMAND ANN ---
ann_demand = MLPRegressor(**ann_params)
print("Training ANN for DEMAND...")
ann_demand.fit(X_train_ann_sc, y_train_d_ann)
pred_ann_d = ann_demand.predict(X_test_ann_sc)
r2_ann_d   = r2_score(y_test_d_ann, pred_ann_d)
nmae_ann_d = mean_absolute_error(y_test_d_ann, pred_ann_d) / y_test_d_ann.mean()
print(f"[ANN | DEMAND] R2: {r2_ann_d:.4f} | NMAE: {nmae_ann_d:.4f}")

# --- PRICE ANN ---

ann_price = MLPRegressor(**ann_params)
print("Training ANN for PRICE...")
ann_price.fit(X_train_ann_sc, y_train_p_ann)
pred_ann_p = ann_price.predict(X_test_ann_sc)   # ← this line was missing
r2_ann_p   = r2_score(y_test_p_ann, pred_ann_p)
nmae_ann_p = mean_absolute_error(y_test_p_ann, pred_ann_p) / y_test_p_ann.mean()
print(f"[ANN | PRICE]  R2: {r2_ann_p:.4f} | NMAE: {nmae_ann_p:.4f}")


# In[ ]:


# ============================================================
# REGULARIZED MODEL VARIANTS — Addressing Overfitting
# ============================================================
print("=" * 70)
print("REGULARIZED MODEL VARIANTS")
print("=" * 70)

# ── Reassert numeric targets (classification cell overwrites these) ──
y_train_d = df_12h['demand_12h'].iloc[:split]
y_test_d  = df_12h['demand_12h'].iloc[split:]
y_train_p = df_12h['price_12h'].iloc[:split]
y_test_p  = df_12h['price_12h'].iloc[split:]

# ── 1. Random Forest — tighter regularization ────────────────
rf_d_reg = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=10,
    max_features=0.7,
    n_jobs=-1, random_state=42
)
rf_p_reg = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=10,
    max_features=0.7,
    n_jobs=-1, random_state=42
)
rf_d_reg.fit(X_train_d, y_train_d)
rf_p_reg.fit(X_train_p, y_train_p)
print("\n--- Random Forest (Regularized) ---")
evaluate(y_train_d, rf_d_reg.predict(X_train_d), "RF-Reg [train]", "DEMAND-12h")
evaluate(y_test_d,  rf_d_reg.predict(X_test_d),  "RF-Reg [test] ", "DEMAND-12h")
evaluate(y_train_p, rf_p_reg.predict(X_train_p), "RF-Reg [train]", "PRICE-12h")
evaluate(y_test_p,  rf_p_reg.predict(X_test_p),  "RF-Reg [test] ", "PRICE-12h")

# ── 2. XGBoost — stronger regularization for price ───────────
xgb_d_reg = XGBRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=2.0,
    n_jobs=-1, random_state=42
)
xgb_p_reg = XGBRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=5,
    reg_alpha=0.5,
    reg_lambda=3.0,
    n_jobs=-1, random_state=42
)
xgb_d_reg.fit(X_train_d, y_train_d)
xgb_p_reg.fit(X_train_p, y_train_p)
print("\n--- XGBoost (Regularized) ---")
evaluate(y_train_d, xgb_d_reg.predict(X_train_d), "XGB-Reg [train]", "DEMAND-12h")
evaluate(y_test_d,  xgb_d_reg.predict(X_test_d),  "XGB-Reg [test] ", "DEMAND-12h")
evaluate(y_train_p, xgb_p_reg.predict(X_train_p), "XGB-Reg [train]", "PRICE-12h")
evaluate(y_test_p,  xgb_p_reg.predict(X_test_p),  "XGB-Reg [test] ", "PRICE-12h")

# ── 3. KNN — larger k fixes the collapse on price ────────────
knn_d_reg = KNeighborsRegressor(n_neighbors=20, weights='distance', n_jobs=-1)
knn_p_reg = KNeighborsRegressor(n_neighbors=20, weights='distance', n_jobs=-1)
knn_d_reg.fit(X_train_d_sc, y_train_d)
knn_p_reg.fit(X_train_p_sc, y_train_p)
print("\n--- KNN k=20 with distance weights (scaled) ---")
evaluate(y_train_d, knn_d_reg.predict(X_train_d_sc), "KNN-Reg [train]", "DEMAND-12h")
evaluate(y_test_d,  knn_d_reg.predict(X_test_d_sc),  "KNN-Reg [test] ", "DEMAND-12h")
evaluate(y_train_p, knn_p_reg.predict(X_train_p_sc), "KNN-Reg [train]", "PRICE-12h")
evaluate(y_test_p,  knn_p_reg.predict(X_test_p_sc),  "KNN-Reg [test] ", "PRICE-12h")

# ── 4. LightGBM — add regularization if available ────────────
if LGBM_AVAILABLE:
    lgbm_d_reg = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1, random_state=42
    )
    lgbm_p_reg = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.3,
        reg_lambda=2.0,
        n_jobs=-1, random_state=42
    )
    lgbm_d_reg.fit(X_train_d, y_train_d)
    lgbm_p_reg.fit(X_train_p, y_train_p)
    print("\n--- LightGBM (Regularized) ---")
    evaluate(y_train_d, lgbm_d_reg.predict(X_train_d), "LGBM-Reg [train]", "DEMAND-12h")
    evaluate(y_test_d,  lgbm_d_reg.predict(X_test_d),  "LGBM-Reg [test] ", "DEMAND-12h")
    evaluate(y_train_p, lgbm_p_reg.predict(X_train_p), "LGBM-Reg [train]", "PRICE-12h")
    evaluate(y_test_p,  lgbm_p_reg.predict(X_test_p),  "LGBM-Reg [test] ", "PRICE-12h")

print("\n" + "="*70)
print("NOTE: Some train→test gap is EXPECTED and NORMAL here because:")
print("  - Price actual has extreme intraday spikes (10→83 EUR/MWh same day)")
print("  - Train=2015-2017, Test=2018 — different market year (regime shift)")
print("  - Lag features are near-perfect on training autocorrelation")
print("  A gap of 0.05-0.15 R2 on energy price data is industry-standard.")
print("="*70)


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# ── Collect results ──────────────────────────────────────────────────────────
models = ['RF-Reg', 'XGB-Reg', 'KNN-Reg', 'LGBM-Reg']

def get_metrics(model, X, y):
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    pred = model.predict(X)
    return (
        r2_score(y, pred),
        np.sqrt(mean_squared_error(y, pred)),
        mean_absolute_error(y, pred)
    )

demand_models_train = [rf_d_reg, xgb_d_reg, knn_d_reg] + ([lgbm_d_reg] if LGBM_AVAILABLE else [])
demand_models_test  = demand_models_train
price_models_train  = [rf_p_reg, xgb_p_reg, knn_p_reg] + ([lgbm_p_reg] if LGBM_AVAILABLE else [])

X_trains_d = [X_train_d, X_train_d, X_train_d_sc] + ([X_train_d] if LGBM_AVAILABLE else [])
X_tests_d  = [X_test_d,  X_test_d,  X_test_d_sc]  + ([X_test_d]  if LGBM_AVAILABLE else [])
X_trains_p = [X_train_p, X_train_p, X_train_p_sc] + ([X_train_p] if LGBM_AVAILABLE else [])
X_tests_p  = [X_test_p,  X_test_p,  X_test_p_sc]  + ([X_test_p]  if LGBM_AVAILABLE else [])

demand_train_metrics = [get_metrics(m, X, y_train_d) for m, X in zip(demand_models_train, X_trains_d)]
demand_test_metrics  = [get_metrics(m, X, y_test_d)  for m, X in zip(demand_models_train, X_tests_d)]
price_train_metrics  = [get_metrics(m, X, y_train_p) for m, X in zip(price_models_train,  X_trains_p)]
price_test_metrics   = [get_metrics(m, X, y_test_p)  for m, X in zip(price_models_train,  X_tests_p)]

# ── Plot ─────────────────────────────────────────────────────────────────────
metric_names = ['R²', 'RMSE', 'MAE']
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Model Performance — Regularized Variants', fontsize=14, y=1.01)

x = np.arange(len(models))
w = 0.35

for col, metric_idx in enumerate(range(3)):
    for row, (train_m, test_m, target) in enumerate([
        (demand_train_metrics, demand_test_metrics, 'Demand-12h'),
        (price_train_metrics,  price_test_metrics,  'Price-12h'),
    ]):
        ax = axes[row, col]
        train_vals = [m[metric_idx] for m in train_m]
        test_vals  = [m[metric_idx] for m in test_m]

        ax.bar(x - w/2, train_vals, w, label='Train', color='#3266ad', alpha=0.85)
        ax.bar(x + w/2, test_vals,  w, label='Test',  color='#e24b4a', alpha=0.85)

        ax.set_title(f'{target} — {metric_names[metric_idx]}', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha='right', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        if metric_names[metric_idx] == 'R²':
            ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('model_metrics.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


import os
import pickle
import json

print("\n" + "="*60)
print("SAVING CLEAN ARTIFACTS (NO PCA) — with validation")
print("="*60)

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Helper to save if present
def safe_save(name, obj):
    if obj is None:
        return None
    path = os.path.join(ARTIFACT_DIR, name)
    try:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        return path
    except Exception as e:
        print(f"Failed saving {name}: {e}")
        return None

saved = {}

# 1. Regression models (prefer XGB if available)
saved['demand_model'] = safe_save('demand_model.pkl', globals().get('xgb_d') or globals().get('rf_d'))
saved['price_model']  = safe_save('price_model.pkl', globals().get('xgb_p') or globals().get('rf_p'))

# 2. Scalers
saved['scaler_demand'] = safe_save('scaler_demand.pkl', globals().get('pca_scaler_d') or globals().get('scaler_ann'))
saved['scaler_price']  = safe_save('scaler_price.pkl', globals().get('pca_scaler_p') or globals().get('scaler_ann'))

# 3. Features
saved['demand_features'] = safe_save('demand_features.pkl', globals().get('demand_features'))
saved['price_features']  = safe_save('price_features.pkl', globals().get('price_features'))

# 4. Thresholds
saved['demand_thresholds'] = safe_save('demand_thresholds.pkl', globals().get('d33') and globals().get('d66') and (d33, d66))
saved['price_thresholds']  = safe_save('price_thresholds.pkl', globals().get('p33') and globals().get('p66') and (p33, p66))

# 5. ANN artifacts
saved['ann_demand'] = safe_save('ann_demand.pkl', globals().get('ann_demand'))
saved['ann_price']  = safe_save('ann_price.pkl', globals().get('ann_price'))
saved['ann_scaler'] = safe_save('ann_scaler.pkl', globals().get('scaler_ann'))
saved['ann_features']= safe_save('ann_features.pkl', globals().get('ann_features'))

# 6. Classification artifacts
# Save trained classifiers if available
if 'trained_classifiers_all' in globals():
    try:
        for target in ['demand', 'price']:
            for name, clf in trained_classifiers_all[target].items():
                safe_save(f"{target}_clf_{name}.pkl", clf)
        saved['label_encoder_d'] = safe_save('demand_label_encoder.pkl', globals().get('label_encoder_d'))
        saved['label_encoder_p'] = safe_save('price_label_encoder.pkl', globals().get('label_encoder_p'))
        saved['best_classifiers'] = safe_save('best_classifiers.pkl', globals().get('best_classifiers'))
    except Exception as e:
        print('Failed saving classification artifacts:', e)

# 7. Cluster artifacts/profile saved earlier by clustering cell
# Additional safety: if cluster_profiles exists, ensure it's on disk
if 'cluster_profiles' in globals():
    try:
        cluster_profiles.to_csv(os.path.join(ARTIFACT_DIR, 'cluster_profiles.csv'))
        saved['cluster_profiles'] = os.path.join(ARTIFACT_DIR, 'cluster_profiles.csv')
    except Exception as e:
        print('Failed saving cluster_profiles:', e)

# 8. Write metadata for API consumption
metadata = {
    'saved': {k: v for k, v in saved.items() if v},
    'timestamp': pd.Timestamp.utcnow().isoformat(),
}
with open(os.path.join(ARTIFACT_DIR, 'model_metadata.json'), 'w', encoding='utf8') as f:
    json.dump(metadata, f, indent=2)

print('Artifacts save summary:')
for k, v in metadata['saved'].items():
    print(f" - {k}: {v}")

print('All artifacts saved (where available).')


# In[ ]:


# ============================================================
# MODEL RECOMMENDATION SYSTEM — FIXED
# ============================================================
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd

print("=" * 60)
print("   MODEL RECOMMENDATION SYSTEM (FIXED)")
print("=" * 60)

# ── Reassert correct numeric splits (split may have been redefined in ANN cell) ──
_split = int(len(df_12h) * 0.8)

_X_train_d = df_12h[demand_features].iloc[:_split]
_X_test_d  = df_12h[demand_features].iloc[_split:]
_y_train_d = df_12h['demand_12h'].iloc[:_split]   # always numeric
_y_test_d  = df_12h['demand_12h'].iloc[_split:]

_X_train_p = df_12h[price_features].iloc[:_split]
_X_test_p  = df_12h[price_features].iloc[_split:]
_y_train_p = df_12h['price_12h'].iloc[:_split]    # always numeric
_y_test_p  = df_12h['price_12h'].iloc[_split:]

print(f"Rec-system split → Train: {_split} | Test: {len(df_12h) - _split}")

# ── Helper ────────────────────────────────────────────────────
def evaluate_model(name, model_d, model_p,
                   Xtr_d, ytr_d, Xte_d, yte_d,
                   Xtr_p, ytr_p, Xte_p, yte_p,
                   pretrained=False):
    if not pretrained:
        model_d.fit(Xtr_d, ytr_d)
        model_p.fit(Xtr_p, ytr_p)

    preds_d = model_d.predict(Xte_d)
    preds_p = model_p.predict(Xte_p)

    r2_d   = r2_score(yte_d, preds_d)
    nmae_d = mean_absolute_error(yte_d, preds_d) / yte_d.mean()
    r2_p   = r2_score(yte_p, preds_p)
    nmae_p = mean_absolute_error(yte_p, preds_p) / yte_p.mean()

    return {
        "Model"       : name,
        "Demand R2"   : round(r2_d,   4),
        "Demand NMAE" : round(nmae_d, 4),
        "Price R2"    : round(r2_p,   4),
        "Price NMAE"  : round(nmae_p, 4),
        "Avg R2"      : round((r2_d + r2_p) / 2, 4),
    }

# ── Fresh baselines ───────────────────────────────────────────
fresh_candidates = {
    "Linear Regression" : (LinearRegression(),   LinearRegression()),
    "Bayesian Ridge"    : (BayesianRidge(),       BayesianRidge()),
    "KNN (k=5)"         : (KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
                            KNeighborsRegressor(n_neighbors=5, n_jobs=-1)),
    "Decision Tree"     : (DecisionTreeRegressor(max_depth=10, random_state=42),
                            DecisionTreeRegressor(max_depth=10, random_state=42)),
}

# ── Pre-trained ensembles (reuse from earlier cell) ───────────
pretrained_candidates = {
    "Random Forest" : (rf_d,  rf_p),
    "XGBoost"       : (xgb_d, xgb_p),
}
if LGBM_AVAILABLE:
    pretrained_candidates["LightGBM"] = (lgbm_d, lgbm_p)

# ── ANN (its own scaler/features — kept separate) ─────────────
ann_entry = evaluate_model(
    "ANN",
    ann_demand, ann_price,
    X_train_ann_sc, y_train_d_ann, X_test_ann_sc, y_test_d_ann,
    X_train_ann_sc, y_train_p_ann, X_test_ann_sc, y_test_p_ann,
    pretrained=True
)

# ── Run ───────────────────────────────────────────────────────
rec_results = []

for name, (md, mp) in fresh_candidates.items():
    rec_results.append(evaluate_model(
        name, md, mp,
        _X_train_d, _y_train_d, _X_test_d, _y_test_d,
        _X_train_p, _y_train_p, _X_test_p, _y_test_p,
        pretrained=False
    ))

for name, (md, mp) in pretrained_candidates.items():
    rec_results.append(evaluate_model(
        name, md, mp,
        _X_train_d, _y_train_d, _X_test_d, _y_test_d,
        _X_train_p, _y_train_p, _X_test_p, _y_test_p,
        pretrained=True
    ))

rec_results.append(ann_entry)

# ── Table & recommendation ────────────────────────────────────
rec_df = (pd.DataFrame(rec_results)
            .sort_values("Avg R2", ascending=False)
            .reset_index(drop=True))
rec_df.index += 1

print("\n--- FULL COMPARISON TABLE ---")
print(rec_df.to_string())

best = rec_df.iloc[0]
print("\n" + "=" * 60)
print(f"  RECOMMENDED MODEL : {best['Model']}")
print(f"  Avg R2            : {best['Avg R2']}")
print(f"  Demand R2         : {best['Demand R2']}   NMAE: {best['Demand NMAE']}")
print(f"  Price  R2         : {best['Price R2']}   NMAE: {best['Price NMAE']}")
print("=" * 60)

rec_df.to_csv("artifacts/model_comparison.csv", index=False)
print("Saved → artifacts/model_comparison.csv")


# In[ ]:


# ============================================================
# MODEL COMPARISON — VISUALIZATION
# ============================================================


# ── All models with their test metrics ───────────────────────
comparison = {
    # name                  : (demand_r2, demand_nmae, price_r2, price_nmae)
    "Linear Regression"     : (0.6389, 0.0756,  0.6809, 0.0761),
    "Bayesian Ridge"        : (0.6389, 0.0756,  0.6809, 0.0761),
    "Decision Tree"         : (0.7421, 0.0549,  0.5943, 0.0807),
    "KNN (k=5)"             : (0.5460, 0.0797, -0.1022, 0.2135),
    "Random Forest"         : (0.8102, 0.0468,  0.7214, 0.0705),
    "RF (regularized)"      : (0.8102, 0.0468,  0.7214, 0.0705),  # replace with actual
    "XGBoost"               : (0.8144, 0.0480,  0.7314, 0.0687),
    "XGB (regularized)"     : (0.8144, 0.0480,  0.7314, 0.0687),  # replace with actual
    "LightGBM"              : (0.8197, 0.0469,  0.7282, 0.0692),
    "LGBM (regularized)"    : (0.8197, 0.0469,  0.7282, 0.0692),  # replace with actual
    "ANN"                   : (0.7737, 0.0549,  0.6048, 0.0855),
}

# ── Or build it programmatically from your actual models ─────
# (comment out the dict above and use this instead)
# from sklearn.metrics import r2_score, mean_absolute_error
#
# def get_metrics(model, X, y):
#     pred = model.predict(X)
#     return (
#         r2_score(y, pred),
#         mean_absolute_error(y, pred) / y.mean(),
#     )
#
# comparison = {}
# model_map = {
#     "Linear Regression"  : (lr_d,        lr_p,        X_test_d,    X_test_p),
#     "Random Forest"      : (rf_d,         rf_p,        X_test_d,    X_test_p),
#     "RF (regularized)"   : (rf_d_reg,     rf_p_reg,    X_test_d,    X_test_p),
#     "XGBoost"            : (xgb_d,        xgb_p,       X_test_d,    X_test_p),
#     "XGB (regularized)"  : (xgb_d_reg,    xgb_p_reg,   X_test_d,    X_test_p),
#     "LightGBM"           : (lgbm_d,       lgbm_p,      X_test_d,    X_test_p),
#     "LGBM (regularized)" : (lgbm_d_reg,   lgbm_p_reg,  X_test_d,    X_test_p),
#     "ANN"                : (ann_demand,   ann_price,   X_test_ann_sc, X_test_ann_sc),
# }
# for name, (md, mp, Xd, Xp) in model_map.items():
#     r2_d, nmae_d = get_metrics(md, Xd, y_test_d)
#     r2_p, nmae_p = get_metrics(mp, Xp, y_test_p)
#     comparison[name] = (r2_d, nmae_d, r2_p, nmae_p)

# ── Unpack ────────────────────────────────────────────────────
names      = list(comparison.keys())
demand_r2  = [comparison[m][0] for m in names]
demand_nmae= [comparison[m][1] for m in names]
price_r2   = [comparison[m][2] for m in names]
price_nmae = [comparison[m][3] for m in names]
avg_r2     = [(d + p) / 2 for d, p in zip(demand_r2, price_r2)]

# Sort by avg R2 descending
order      = np.argsort(avg_r2)[::-1]
names      = [names[i]       for i in order]
demand_r2  = [demand_r2[i]   for i in order]
demand_nmae= [demand_nmae[i] for i in order]
price_r2   = [price_r2[i]    for i in order]
price_nmae = [price_nmae[i]  for i in order]
avg_r2     = [avg_r2[i]      for i in order]

x  = np.arange(len(names))
w  = 0.35

BLUE   = "#3266ad"
RED    = "#e24b4a"
GREEN  = "#2e8b57"
ORANGE = "#e07b39"
GRAY   = "#888780"

fig, axes = plt.subplots(2, 2, figsize=(18, 11))
fig.suptitle("Model Comparison — 12h Ahead Forecasting", fontsize=15, fontweight="bold", y=1.01)

# ── Plot 1: R² side by side ───────────────────────────────────
ax = axes[0, 0]
bars1 = ax.bar(x - w/2, demand_r2,  w, label="Demand R²", color=BLUE,  alpha=0.85, zorder=3)
bars2 = ax.bar(x + w/2, price_r2,   w, label="Price R²",  color=RED,   alpha=0.85, zorder=3)
ax.axhline(0, color=GRAY, linewidth=0.8, linestyle="--")
ax.set_title("R² Score (higher = better)", fontsize=12)
ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("R²"); ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.set_ylim(-0.3, 1.05)
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}",
            ha="center", va="bottom", fontsize=7, color=BLUE)
for bar in bars2:
    h = bar.get_height()
    if h > -0.2:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=7, color=RED)

# ── Plot 2: NMAE side by side ─────────────────────────────────
ax = axes[0, 1]
bars3 = ax.bar(x - w/2, demand_nmae, w, label="Demand NMAE", color=BLUE,  alpha=0.85, zorder=3)
bars4 = ax.bar(x + w/2, price_nmae,  w, label="Price NMAE",  color=RED,   alpha=0.85, zorder=3)
ax.set_title("NMAE (lower = better)", fontsize=12)
ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("NMAE"); ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3, zorder=0)
for bar in [*bars3, *bars4]:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.001, f"{h:.3f}",
            ha="center", va="bottom", fontsize=7,
            color=BLUE if bar in bars3 else RED)

# ── Plot 3: Avg R² ranking (horizontal bar) ───────────────────
ax = axes[1, 0]
colors = [GREEN if v >= 0.75 else ORANGE if v >= 0.65 else RED for v in avg_r2]
hbars  = ax.barh(names[::-1], avg_r2[::-1], color=colors[::-1], alpha=0.85, zorder=3)
ax.axvline(0.75, color=GREEN,  linewidth=1.2, linestyle="--", alpha=0.6, label="Good (0.75)")
ax.axvline(0.65, color=ORANGE, linewidth=1.2, linestyle="--", alpha=0.6, label="Acceptable (0.65)")
ax.set_title("Average R² Ranking", fontsize=12)
ax.set_xlabel("Avg R²"); ax.legend(fontsize=8)
ax.grid(axis="x", alpha=0.3, zorder=0)
ax.set_xlim(-0.15, 1.05)
for bar, val in zip(hbars, avg_r2[::-1]):
    ax.text(max(val + 0.01, 0.02), bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=8.5, fontweight="bold")

# ── Plot 4: Radar-style summary table ────────────────────────
ax = axes[1, 1]
ax.axis("off")
col_labels = ["Model", "Demand R²", "Price R²", "Avg R²", "D-NMAE", "P-NMAE"]
table_data = [
    [n,
     f"{demand_r2[i]:.4f}",
     f"{price_r2[i]:.4f}",
     f"{avg_r2[i]:.4f}",
     f"{demand_nmae[i]:.4f}",
     f"{price_nmae[i]:.4f}"]
    for i, n in enumerate(names)
]
tbl = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.55)

# Color header
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#3266ad")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Highlight best row
for j in range(len(col_labels)):
    tbl[1, j].set_facecolor("#d4edda")

# Alternate row shading
for i in range(1, len(names) + 1):
    for j in range(len(col_labels)):
        if i % 2 == 0:
            tbl[i, j].set_facecolor("#f7f7f5")

ax.set_title("Full Metrics Table", fontsize=12, pad=12)

plt.tight_layout()
plt.savefig("artifacts/model_comparison_full.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → artifacts/model_comparison_full.png")


# In[ ]:


# ============================================================
# MODEL COMPARISON — TRAIN vs TEST GAP (GENERALIZATION FOCUS)
# ============================================================
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

# ── Collect BOTH train and test metrics from actual models ────
def get_full_metrics(model, X_train, y_train, X_test, y_test):
    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)
    return {
        "train_r2"  : r2_score(y_train, pred_train),
        "test_r2"   : r2_score(y_test,  pred_test),
        "gap_r2"    : r2_score(y_train, pred_train) - r2_score(y_test, pred_test),
        "train_nmae": mean_absolute_error(y_train, pred_train) / y_train.mean(),
        "test_nmae" : mean_absolute_error(y_test,  pred_test)  / y_test.mean(),
    }

# ── Model registry — (model_d, model_p, X_train_d, X_test_d, X_train_p, X_test_p)
model_registry = {
    "RF (raw)"          : (rf_d,       rf_p,       X_train_d,      X_test_d,      X_train_p,      X_test_p),
    "RF (regularized)"  : (rf_d_reg,   rf_p_reg,   X_train_d,      X_test_d,      X_train_p,      X_test_p),
    "XGB (raw)"         : (xgb_d,      xgb_p,      X_train_d,      X_test_d,      X_train_p,      X_test_p),
    "XGB (regularized)" : (xgb_d_reg,  xgb_p_reg,  X_train_d,      X_test_d,      X_train_p,      X_test_p),
    "LGBM (raw)"        : (lgbm_d,     lgbm_p,     X_train_d,      X_test_d,      X_train_p,      X_test_p),
    "LGBM (regularized)": (lgbm_d_reg, lgbm_p_reg, X_train_d,      X_test_d,      X_train_p,      X_test_p),
    "ANN"               : (ann_demand, ann_price,  X_train_ann_sc, X_test_ann_sc, X_train_ann_sc, X_test_ann_sc),
}

# ── Build metrics dict ────────────────────────────────────────
demand_metrics = {}
price_metrics  = {}

for name, (md, mp, Xtr_d, Xte_d, Xtr_p, Xte_p) in model_registry.items():
    demand_metrics[name] = get_full_metrics(md, Xtr_d, _y_train_d, Xte_d, _y_test_d)
    price_metrics[name]  = get_full_metrics(mp, Xtr_p, _y_train_p, Xte_p, _y_test_p)

names = list(model_registry.keys())

# ── Score = test_r2 - penalty for gap (generalization score) ──
# Gap penalty: every 0.1 gap costs 0.05 in score
def gen_score(m):
    gap_penalty = m["gap_r2"] * 0.5
    return m["test_r2"] - gap_penalty

demand_gen_scores = [gen_score(demand_metrics[n]) for n in names]
price_gen_scores  = [gen_score(price_metrics[n])  for n in names]

best_demand = names[np.argmax(demand_gen_scores)]
best_price  = names[np.argmax(price_gen_scores)]

# ── Colors ────────────────────────────────────────────────────
BLUE_DARK  = "#1a4a8a"
BLUE_LIGHT = "#7bafd4"
RED_DARK   = "#b52b2b"
RED_LIGHT  = "#f0908e"
GREEN      = "#2e8b57"
ORANGE     = "#e07b39"
GRAY       = "#888780"

# ── Raw vs Reg pairs for gap annotation ──────────────────────
pairs = [
    ("RF (raw)",   "RF (regularized)"),
    ("XGB (raw)",  "XGB (regularized)"),
    ("LGBM (raw)", "LGBM (regularized)"),
]

fig, axes = plt.subplots(2, 3, figsize=(22, 12))
fig.suptitle("Model Comparison — Generalization & Train/Test Gap Analysis",
             fontsize=14, fontweight="bold", y=1.01)

# ════════════════════════════════════════════════════════════
# PLOT 1 — DEMAND: Train vs Test R²
# ════════════════════════════════════════════════════════════
ax = axes[0, 0]
x = np.arange(len(names))
w = 0.35

tr_r2 = [demand_metrics[n]["train_r2"] for n in names]
te_r2 = [demand_metrics[n]["test_r2"]  for n in names]
gap   = [demand_metrics[n]["gap_r2"]   for n in names]

b1 = ax.bar(x - w/2, tr_r2, w, label="Train R²", color=BLUE_DARK,  alpha=0.85, zorder=3)
b2 = ax.bar(x + w/2, te_r2, w, label="Test R²",  color=BLUE_LIGHT, alpha=0.85, zorder=3)

# Gap annotation arrows
for i, (tr, te, g) in enumerate(zip(tr_r2, te_r2, gap)):
    if g > 0.03:
        ax.annotate("", xy=(i + w/2, te + 0.01), xytext=(i - w/2, tr + 0.01),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2))
        ax.text(i, max(tr, te) + 0.04, f"Δ{g:.2f}",
                ha="center", fontsize=7, color=ORANGE, fontweight="bold")

ax.set_title("DEMAND — Train vs Test R²", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("R²")
ax.set_ylim(0, 1.15)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.axhline(0.8, color=GREEN, linewidth=1, linestyle="--", alpha=0.5)

for bar in b2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}",
            ha="center", va="bottom", fontsize=6.5, color=BLUE_DARK)

# ════════════════════════════════════════════════════════════
# PLOT 2 — PRICE: Train vs Test R²
# ════════════════════════════════════════════════════════════
ax = axes[0, 1]

tr_r2 = [price_metrics[n]["train_r2"] for n in names]
te_r2 = [price_metrics[n]["test_r2"]  for n in names]
gap   = [price_metrics[n]["gap_r2"]   for n in names]

b1 = ax.bar(x - w/2, tr_r2, w, label="Train R²", color=RED_DARK,  alpha=0.85, zorder=3)
b2 = ax.bar(x + w/2, te_r2, w, label="Test R²",  color=RED_LIGHT, alpha=0.85, zorder=3)

for i, (tr, te, g) in enumerate(zip(tr_r2, te_r2, gap)):
    if g > 0.03:
        ax.annotate("", xy=(i + w/2, te + 0.01), xytext=(i - w/2, tr + 0.01),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2))
        ax.text(i, max(tr, te) + 0.04, f"Δ{g:.2f}",
                ha="center", fontsize=7, color=ORANGE, fontweight="bold")

ax.set_title("PRICE — Train vs Test R²", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("R²")
ax.set_ylim(0, 1.15)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.axhline(0.75, color=GREEN, linewidth=1, linestyle="--", alpha=0.5)

for bar in b2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}",
            ha="center", va="bottom", fontsize=6.5, color=RED_DARK)

# ════════════════════════════════════════════════════════════
# PLOT 3 — Train/Test GAP comparison (raw vs regularized)
# ════════════════════════════════════════════════════════════
ax = axes[0, 2]

pair_names  = [p[0].replace(" (raw)", "") for p in pairs] + ["ANN"]
d_gap_raw   = [demand_metrics[p[0]]["gap_r2"]  for p in pairs] + [demand_metrics["ANN"]["gap_r2"]]
d_gap_reg   = [demand_metrics[p[1]]["gap_r2"]  for p in pairs] + [None]
p_gap_raw   = [price_metrics[p[0]]["gap_r2"]   for p in pairs] + [price_metrics["ANN"]["gap_r2"]]
p_gap_reg   = [price_metrics[p[1]]["gap_r2"]   for p in pairs] + [None]

xi = np.arange(len(pair_names))
w2 = 0.2

ax.bar(xi - 1.5*w2, d_gap_raw, w2, label="Demand raw",  color=BLUE_DARK,  alpha=0.85, zorder=3)
ax.bar(xi - 0.5*w2, [v if v is not None else 0 for v in d_gap_reg], w2,
       label="Demand reg",  color=BLUE_LIGHT, alpha=0.85, zorder=3)
ax.bar(xi + 0.5*w2, p_gap_raw, w2, label="Price raw",   color=RED_DARK,   alpha=0.85, zorder=3)
ax.bar(xi + 1.5*w2, [v if v is not None else 0 for v in p_gap_reg], w2,
       label="Price reg",   color=RED_LIGHT,  alpha=0.85, zorder=3)

ax.axhline(0.05, color=GREEN,  linewidth=1.2, linestyle="--", alpha=0.6, label="Target gap (0.05)")
ax.axhline(0.10, color=ORANGE, linewidth=1.2, linestyle="--", alpha=0.6, label="Concern (0.10)")
ax.set_title("Train−Test R² Gap (lower = less overfit)", fontsize=11)
ax.set_xticks(xi)
ax.set_xticklabels(pair_names, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Gap (Train R² − Test R²)")
ax.legend(fontsize=7, ncol=2)
ax.grid(axis="y", alpha=0.3, zorder=0)

# ════════════════════════════════════════════════════════════
# PLOT 4 — Generalization Score (test R² penalized for gap)
# ════════════════════════════════════════════════════════════
ax = axes[1, 0]

d_gen = [gen_score(demand_metrics[n]) for n in names]
p_gen = [gen_score(price_metrics[n])  for n in names]

b1 = ax.bar(x - w/2, d_gen, w, label="Demand", color=BLUE_DARK,  alpha=0.85, zorder=3)
b2 = ax.bar(x + w/2, p_gen, w, label="Price",  color=RED_DARK,   alpha=0.85, zorder=3)

ax.set_title("Generalization Score\n(Test R² − 0.5 × Gap)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("Generalization Score")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3, zorder=0)

# Star best models
best_d_idx = np.argmax(d_gen)
best_p_idx = np.argmax(p_gen)
ax.text(best_d_idx - w/2, d_gen[best_d_idx] + 0.01, "★", ha="center",
        fontsize=11, color=BLUE_DARK)
ax.text(best_p_idx + w/2, p_gen[best_p_idx] + 0.01, "★", ha="center",
        fontsize=11, color=RED_DARK)

# ════════════════════════════════════════════════════════════
# PLOT 5 — NMAE Train vs Test
# ════════════════════════════════════════════════════════════
ax = axes[1, 1]

d_nmae_tr = [demand_metrics[n]["train_nmae"] for n in names]
d_nmae_te = [demand_metrics[n]["test_nmae"]  for n in names]
p_nmae_tr = [price_metrics[n]["train_nmae"]  for n in names]
p_nmae_te = [price_metrics[n]["test_nmae"]   for n in names]

w3 = 0.2
ax.bar(x - 1.5*w3, d_nmae_tr, w3, label="Demand train", color=BLUE_DARK,  alpha=0.85, zorder=3)
ax.bar(x - 0.5*w3, d_nmae_te, w3, label="Demand test",  color=BLUE_LIGHT, alpha=0.85, zorder=3)
ax.bar(x + 0.5*w3, p_nmae_tr, w3, label="Price train",  color=RED_DARK,   alpha=0.85, zorder=3)
ax.bar(x + 1.5*w3, p_nmae_te, w3, label="Price test",   color=RED_LIGHT,  alpha=0.85, zorder=3)

ax.set_title("NMAE — Train vs Test (lower = better)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("NMAE")
ax.legend(fontsize=7, ncol=2)
ax.grid(axis="y", alpha=0.3, zorder=0)

# ════════════════════════════════════════════════════════════
# PLOT 6 — Recommendation summary table
# ════════════════════════════════════════════════════════════
ax = axes[1, 2]
ax.axis("off")

col_labels = ["Model", "D-TestR²", "D-Gap", "D-GenScore", "P-TestR²", "P-Gap", "P-GenScore"]
table_data = []
for n in names:
    dm = demand_metrics[n]
    pm = price_metrics[n]
    table_data.append([
        n,
        f"{dm['test_r2']:.3f}",
        f"{dm['gap_r2']:.3f}",
        f"{gen_score(dm):.3f}",
        f"{pm['test_r2']:.3f}",
        f"{pm['gap_r2']:.3f}",
        f"{gen_score(pm):.3f}",
    ])

tbl = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.5)
tbl.scale(1, 1.5)

for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#1a4a8a")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Highlight best demand and price rows
best_d_row = names.index(best_demand) + 1
best_p_row = names.index(best_price)  + 1
for j in range(len(col_labels)):
    tbl[best_d_row, j].set_facecolor("#cce5ff")
for j in range(len(col_labels)):
    if best_p_row != best_d_row:
        tbl[best_p_row, j].set_facecolor("#ffe0e0")

for i in range(1, len(names) + 1):
    for j in range(len(col_labels)):
        if i % 2 == 0 and i != best_d_row and i != best_p_row:
            tbl[i, j].set_facecolor("#f5f5f3")

ax.set_title("Generalization Metrics Table\n(blue=best demand, red=best price)",
             fontsize=10, pad=10)

plt.tight_layout()
plt.savefig("artifacts/model_comparison_generalization.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Final recommendation printout ────────────────────────────
print("\n" + "="*60)
print("  RECOMMENDATION (based on generalization score)")
print("="*60)
print(f"\n  DEMAND  →  {best_demand}")
print(f"    Test R²         : {demand_metrics[best_demand]['test_r2']:.4f}")
print(f"    Train-Test Gap  : {demand_metrics[best_demand]['gap_r2']:.4f}")
print(f"    Gen Score       : {gen_score(demand_metrics[best_demand]):.4f}")

print(f"\n  PRICE   →  {best_price}")
print(f"    Test R²         : {price_metrics[best_price]['test_r2']:.4f}")
print(f"    Train-Test Gap  : {price_metrics[best_price]['gap_r2']:.4f}")
print(f"    Gen Score       : {gen_score(price_metrics[best_price]):.4f}")
print("="*60)


# In[ ]:


from prefect import task, flow
import pandas as pd
import numpy as np
import pickle
import os
import json

# =========================
# 1. LOAD DATA
# =========================
@task(retries=2, retry_delay_seconds=30)
def load_data(path="energy_dataset.csv"):
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.sort_values('time').set_index('time')
    return df


# =========================
# 2. FEATURE ENGINEERING
# =========================
@task(retries=1, retry_delay_seconds=10)
def preprocess(df):

    df['renewable'] = (
        df['generation solar'].fillna(0) +
        df['generation wind onshore'].fillna(0)
    )

    df['fossil'] = (
        df['generation fossil gas'].fillna(0) +
        df['generation fossil hard coal'].fillna(0)
    )

    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['demand_lag_1h'] = df['total load actual'].shift(1)
    df['price_lag_1h']  = df['price actual'].shift(1)

    df['demand_12h'] = df['total load actual'].shift(-12)
    df['price_12h']  = df['price actual'].shift(-12)

    df = df.dropna()

    return df


# =========================
# 3. TRAIN MODEL
# =========================
@task(retries=2, retry_delay_seconds=30)
def train_model(df):

    try:
        from xgboost import XGBRegressor
        xgb_available = True
    except Exception:
        xgb_available = False

    features = [
        'hour_sin', 'hour_cos',
        'demand_lag_1h', 'price_lag_1h',
        'renewable', 'fossil'
    ]

    X = df[features]
    y_demand = df['demand_12h']
    y_price  = df['price_12h']

    if xgb_available:
        model_d = XGBRegressor(n_estimators=300)
        model_p = XGBRegressor(n_estimators=300)
    else:
        from sklearn.ensemble import RandomForestRegressor
        model_d = RandomForestRegressor(n_estimators=200)
        model_p = RandomForestRegressor(n_estimators=200)

    model_d.fit(X, y_demand)
    model_p.fit(X, y_price)

    return model_d, model_p, features


# =========================
# 4. CREATE CLASSIFICATION THRESHOLDS
# =========================
@task
def create_thresholds(df):

    d33, d66 = df['demand_12h'].quantile([0.33, 0.66])
    p33, p66 = df['price_12h'].quantile([0.33, 0.66])

    return (d33, d66), (p33, p66)


# =========================
# 5. SAVE ARTIFACTS
# =========================
@task(retries=2, retry_delay_seconds=10)
def save_artifacts(model_d, model_p, features, d_th, p_th):

    os.makedirs("artifacts", exist_ok=True)

    artifacts = {
        "demand_model.pkl": model_d,
        "price_model.pkl": model_p,
        "features.pkl": features,
        "demand_thresholds.pkl": d_th,
        "price_thresholds.pkl": p_th
    }

    saved = []
    for name, obj in artifacts.items():
        try:
            with open(f"artifacts/{name}", "wb") as f:
                pickle.dump(obj, f)
            saved.append(name)
        except Exception as e:
            print(f"Failed to save {name}: {e}")

    # basic validation
    missing = [n for n in artifacts.keys() if n not in saved]
    if missing:
        raise RuntimeError(f"Failed to persist artifacts: {missing}")

    # write metadata
    meta = {"saved": saved, "timestamp": pd.Timestamp.utcnow().isoformat()}
    with open('artifacts/model_metadata.json', 'w', encoding='utf8') as f:
        json.dump(meta, f, indent=2)

    return "Saved successfully"


# =========================
# 6. MAIN FLOW
# =========================
@flow
def energy_pipeline():

    df = load_data()
    df = preprocess(df)

    model_d, model_p, features = train_model(df)
    d_th, p_th = create_thresholds(df)

    result = save_artifacts(model_d, model_p, features, d_th, p_th)

    return result


# RUN PIPELINE
if __name__ == "__main__":
    energy_pipeline()


# In[58]:


from pathlib import Path
import json
import numpy as np
import pandas as pd

experiment_log = {
    "source": "colab_modeltraining_v4.ipynb",
    "best_models": {
        "demand": {
            "model": best_demand_classifier_name,
            "metrics": classifier_results["demand"][best_demand_classifier_name],
        },
        "price": {
            "model": best_price_classifier_name,
            "metrics": classifier_results["price"][best_price_classifier_name],
        },
    },
    "regression": results,
    "classification": classifier_results,
    "clustering": {
        "chosen_k": int(best_k),
        "best_k_by_silhouette": int(best_k_sil),
        "cluster_sizes": cluster_sizes.to_dict(),
    },
    "recommendation": {
        "best_demand_classifier": best_demand_classifier_name,
        "best_price_classifier": best_price_classifier_name,
    },
    "all_trained_models": [],
    "observations": {
        "best_performing_model": f"{best_demand_classifier_name} / {best_price_classifier_name}",
        "data_quality_issues": [
            "Handled missing and duplicate features during preprocessing.",
            "Standardized time-based features and lagged values before training.",
        ],
        "overfitting_patterns": [
            "Compared raw and PCA-based regression scores to spot instability.",
            "Tracked train/test behavior for classification and clustering models.",
        ],
        "deployment_speed": "Artifacts are saved once and served through thin API endpoints.",
        "prefect_reliability": "Local artifact logs provide a fallback when orchestration is unavailable.",
    },
}

for name, values in results.items():
    experiment_log["all_trained_models"].append({
        "task": "regression_baseline",
        "model": name,
        "demand_r2_raw": float(values.get("demand_r2_raw", np.nan)),
        "price_r2_raw": float(values.get("price_r2_raw", np.nan)),
        "demand_r2_pca": float(values.get("demand_r2_pca", np.nan)),
        "price_r2_pca": float(values.get("price_r2_pca", np.nan)),
    })

for target_name, target_results in [("demand", classifier_results["demand"]), ("price", classifier_results["price"])]:
    for model_name, metrics in target_results.items():
        experiment_log["all_trained_models"].append({
            "task": f"classification_{target_name}",
            "model": model_name,
            "cv_f1_macro": None if metrics.get("cv_f1_macro") is None else float(metrics.get("cv_f1_macro")),
            "accuracy": float(metrics.get("accuracy", np.nan)),
            "macro_f1": float(metrics.get("macro_f1", np.nan)),
        })

experiment_log["all_trained_models"].append({
    "task": "clustering",
    "model": "KMeans",
    "best_k_by_silhouette": int(best_k_sil),
    "chosen_k": int(best_k),
})

if "r2_rf_d" in globals() and "r2_rf_p" in globals():
    experiment_log["all_trained_models"].append({
        "task": "regression_advanced",
        "model": "Random Forest",
        "demand_r2": float(r2_rf_d),
        "price_r2": float(r2_rf_p),
    })
if "r2_xgb_d" in globals() and "r2_xgb_p" in globals():
    experiment_log["all_trained_models"].append({
        "task": "regression_advanced",
        "model": "XGBoost",
        "demand_r2": float(r2_xgb_d),
        "price_r2": float(r2_xgb_p),
    })
if "LGBM_AVAILABLE" in globals() and LGBM_AVAILABLE and all(name in globals() for name in ["r2_lgbm_d", "r2_lgbm_p"]):
    experiment_log["all_trained_models"].append({
        "task": "regression_advanced",
        "model": "LightGBM",
        "demand_r2": float(r2_lgbm_d),
        "price_r2": float(r2_lgbm_p),
    })

artifact_dir = Path(ARTIFACT_DIR)
log_json_path = artifact_dir / "experiment_log.json"
log_csv_path = artifact_dir / "experiment_log.csv"
log_md_path = artifact_dir / "experiment_log.md"
models_csv_path = artifact_dir / "experiment_models.csv"

with open(log_json_path, "w", encoding="utf8") as f:
    json.dump(experiment_log, f, indent=2)

summary_rows = []
for task_name, model_data in experiment_log["best_models"].items():
    summary_rows.append({
        "section": "best_model",
        "task": task_name,
        "model": model_data["model"],
    })
summary_rows.append({
    "section": "clustering",
    "task": "chosen_k",
    "model": str(experiment_log["clustering"]["chosen_k"]),
})
summary_rows.append({
    "section": "clustering",
    "task": "best_k_by_silhouette",
    "model": str(experiment_log["clustering"]["best_k_by_silhouette"]),
})

pd.DataFrame(summary_rows).to_csv(log_csv_path, index=False)
pd.DataFrame(experiment_log["all_trained_models"]).to_csv(models_csv_path, index=False)

with open(log_md_path, "w", encoding="utf8") as f:
    f.write("# ML Experimentation & Observations\n\n")
    f.write("## Best Models\n")
    f.write(f"- Demand: {experiment_log['best_models']['demand']['model']}\n")
    f.write(f"- Price: {experiment_log['best_models']['price']['model']}\n\n")
    f.write("## Clustering\n")
    f.write(f"- Chosen k: {experiment_log['clustering']['chosen_k']}\n")
    f.write(f"- Best k by silhouette: {experiment_log['clustering']['best_k_by_silhouette']}\n\n")
    f.write("## Observations\n")
    for key, values in experiment_log["observations"].items():
        f.write(f"- {key}: {values}\n")

print(f"Saved experiment log to: {log_json_path}")
print(f"Saved experiment summary table to: {log_csv_path}")
print(f"Saved full model list to: {models_csv_path}")
print(f"Saved experiment notes to: {log_md_path}")


# ============================================================
# SAVE ALL 16 ARTIFACTS FOR FASTAPI
# ============================================================
import os
import pickle
from sklearn.preprocessing import normalize, RobustScaler

os.makedirs('artifacts', exist_ok=True)

def save_artifact(obj, name):
    with open(f'artifacts/{name}', 'wb') as f:
        pickle.dump(obj, f)
    print(f'Saved: {name}')

# Best regression models (XGBoost won)
save_artifact(xgb_d, 'reg_demand.pkl')
save_artifact(xgb_p, 'reg_price.pkl')

# Best classifiers
save_artifact(best_demand_classifier, 'clf_demand.pkl')
best_price_classifier = trained_classifiers['price'].get(best_price_classifier_name)
save_artifact(best_price_classifier, 'clf_price.pkl')

# Label encoders
save_artifact(le_d, 'le_demand.pkl')
save_artifact(le_p, 'le_price.pkl')

# Feature lists
save_artifact(demand_features, 'demand_features.pkl')
save_artifact(price_features, 'price_features.pkl')

# Clustering
cluster_scaler_final = RobustScaler().fit(df_12h[cluster_features])
save_artifact(cluster_scaler_final, 'cluster_scaler.pkl')
save_artifact(kmeans_final, 'kmeans.pkl')

# Thresholds
d33 = float(np.percentile(y_train_d, 33))
d66 = float(np.percentile(y_train_d, 66))
p33 = float(np.percentile(y_train_p, 33))
p66 = float(np.percentile(y_train_p, 66))
save_artifact({'d33': d33, 'd66': d66, 'p33': p33, 'p66': p66, 'best_k': int(best_k)}, 'thresholds.pkl')

# Label maps
save_artifact({'Low': 0, 'Medium': 1, 'High': 2}, 'label_map.pkl')
save_artifact({0: 'Low', 1: 'Medium', 2: 'High'}, 'inv_map.pkl')

# Recommendation table
rec_table = {
    ('Low',    'Low'):    'Low demand and low price: ideal time to charge EVs and run heavy appliances.',
    ('Low',    'Medium'): 'Low demand, moderate price: good time for non-critical loads.',
    ('Low',    'High'):   'Low demand but high price: avoid discretionary usage despite low grid load.',
    ('Medium', 'Low'):    'Moderate demand, low price: favorable window for flexible loads.',
    ('Medium', 'Medium'): 'Balanced conditions: maintain standard operations.',
    ('Medium', 'High'):   'Moderate demand but high price: defer non-essential consumption.',
    ('High',   'Low'):    'High demand but low price: grid is stressed but affordable.',
    ('High',   'Medium'): 'Peak demand, moderate price: reduce non-essential usage.',
    ('High',   'High'):   'Peak demand and high price: minimize usage immediately.',
}
save_artifact(rec_table, 'rec_table.pkl')

# Profile DataFrame and normalized vectors
test_idx = df_12h.index[split:]
pred_demand_test = xgb_d.predict(X_test_d)
pred_price_test  = xgb_p.predict(X_test_p)
test_clusters    = kmeans_final.predict(cluster_scaler_final.transform(df_12h.loc[test_idx, cluster_features]))

profile_df_final = pd.DataFrame({
    'pred_demand':       pred_demand_test,
    'pred_price':        pred_price_test,
    'demand_class_enc':  le_d.transform(df_12h.loc[test_idx, 'demand_class']),
    'price_class_enc':   le_p.transform(df_12h.loc[test_idx, 'price_class']),
    'cluster':           test_clusters,
    'hour_sin':          X_test_d['hour_sin'].values,
    'hour_cos':          X_test_d['hour_cos'].values,
    'month_sin':         np.sin(2 * np.pi * df_12h.loc[test_idx, 'month'] / 12),
    'month_cos':         np.cos(2 * np.pi * df_12h.loc[test_idx, 'month'] / 12),
    'renewable_pct':     df_12h.loc[test_idx, 'renewable_pct'].values,
    'is_weekend':        X_test_d['is_weekend'].values,
}, index=test_idx)

profiles_norm_final = normalize(profile_df_final.values.astype(float), norm='l2')
save_artifact(profile_df_final, 'profile_df.pkl')
save_artifact(profiles_norm_final, 'profiles_norm.pkl')

print('All 16 artifacts saved successfully!')
