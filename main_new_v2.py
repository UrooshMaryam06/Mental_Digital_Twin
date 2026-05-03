# ============================================================
# FILE: main.py  (FastAPI application)
#
# HOW THE PIPELINE WORKS:
#   1. Run model_training.py ONCE → saves artifacts/
#   2. Start this app: uvicorn main:app --reload
#   3. All artifacts are loaded ONCE at startup into memory.
#   4. Every request hits a thin prediction layer — no training
#      happens at request time.
#
# ENDPOINTS:
#   GET  /health                  — liveness check
#   POST /predict/demand          — 12h-ahead demand regression
#   POST /predict/price           — 12h-ahead price regression
#   POST /predict/both            — demand + price in one call
#   POST /classify/demand         — demand regime Low/Med/High
#   POST /classify/price          — price regime Low/Med/High
#   POST /cluster                 — which energy regime cluster
#   POST /recommend               — full recommendation (all 3)
#   GET  /recommend/by_index/{i}  — recommendation from test-set index
#   GET  /models/compare          — R2 comparison of all models
#   GET  /clusters/profiles       — cluster profile table
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
import pandas as pd
import pickle
import os
import re
import ast
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

app = FastAPI(
    title="Electricity Demand & Price Forecasting API",
    description="MLOps Pipeline for Energy Grid Management",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# STARTUP: LOAD ALL ARTIFACTS
# ============================================================
# Every artifact was serialised by model_training.py.
# We load them once here so every request is fast.
# ============================================================

ARTIFACTS = "artifacts"

def load(name):
    path = os.path.join(ARTIFACTS, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Artifact not found: {path}\n"
            f"Run model_training.py first to generate it."
        )
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_optional(name, default):
    """Load an artifact if present, otherwise return `default`."""
    path = os.path.join(ARTIFACTS, name)
    if not os.path.exists(path):
        return default
    with open(path, 'rb') as f:
        return pickle.load(f)

print("Loading artifacts...")
reg_demand      = load('reg_demand.pkl')
reg_price       = load('reg_price.pkl')
clf_demand      = load('clf_demand.pkl')
clf_price       = load('clf_price.pkl')
le_demand       = load('le_demand.pkl')
le_price        = load('le_price.pkl')
kmeans_model    = load('kmeans.pkl')
cluster_scaler  = load('cluster_scaler.pkl')
profiles_norm   = load('profiles_norm.pkl')
profile_df      = load('profile_df.pkl')
demand_features = load('demand_features.pkl')
price_features  = load('price_features.pkl')
label_map       = load('label_map.pkl')
# `inv_map` may not be present in older artifact sets — load optionally and build a fallback.
inv_map = load_optional('inv_map.pkl', None)
if inv_map is None:
    try:
        if isinstance(label_map, dict):
            inv_map = {int(v): k for k, v in label_map.items()}
        elif hasattr(le_demand, 'classes_'):
            inv_map = {i: c for i, c in enumerate(le_demand.classes_)}
        else:
            inv_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    except Exception:
        inv_map = {0: 'Low', 1: 'Medium', 2: 'High'}
REC_TABLE       = load('rec_table.pkl')
thresholds      = load('thresholds.pkl')
print("All artifacts loaded.")


def get_cluster_count() -> int:
    """Return the trained cluster count with a safe fallback."""
    return int(thresholds.get('best_k', getattr(kmeans_model, 'n_clusters', 0)))

try:
    print("Loading historical dataset for feature extraction...")
    HIST_DF = pd.read_csv('energy_dataset.csv')
    HIST_DF['time'] = pd.to_datetime(HIST_DF['time'], utc=True)
    HIST_DF = HIST_DF.sort_values('time').set_index('time')
    print("Historical dataset loaded successfully.")
except Exception as e:
    print(f"Warning: Failed to load energy_dataset.csv. Lag feature extraction will fall back to defaults. Error: {e}")
    HIST_DF = pd.DataFrame()

assoc_rules = load_optional('association_rules.pkl', pd.DataFrame())
# Load and sanitise association rules (CSV takes precedence if present)
def _sanitize_assoc_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    def _safe_parse(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return []
        if isinstance(x, (list, tuple, set)):
            return sorted([str(v).strip() for v in list(x) if str(v).strip()])
        if isinstance(x, str):
            s = x.strip()
            try:
                if s.startswith(('[', '(', '{')):
                    val = ast.literal_eval(s)
                    if isinstance(val, (list, tuple, set)):
                        return sorted([str(v).strip() for v in val if str(v).strip()])
            except Exception:
                pass
            s = s.replace('\n', ',').replace('|', ',')
            s = re.sub(r"^[\[\(\{\]\)\}\s\'\"]+|[\[\(\{\]\)\}\s\'\"]+$", '', s)
            parts = [p.strip() for p in re.split(r'[;,\\/]|,', s) if p and p.strip()]
            return sorted(list(dict.fromkeys(parts)))
        return [str(x)]

    try:
        if 'antecedents' in df.columns:
            df['antecedents'] = df['antecedents'].apply(_safe_parse)
        if 'consequents' in df.columns:
            df['consequents'] = df['consequents'].apply(_safe_parse)
    except Exception:
        pass
    return df

csv_path = os.path.join(ARTIFACTS, 'association_rules.csv')
if os.path.exists(csv_path):
    try:
        assoc_rules = pd.read_csv(csv_path)
        assoc_rules = _sanitize_assoc_df(assoc_rules)
    except Exception:
        assoc_rules = None
else:
    # sanitize the pickle-loaded rules if present
    try:
        assoc_rules = _sanitize_assoc_df(assoc_rules)
    except Exception:
        pass


# ============================================================
# PYDANTIC MODELS
# ============================================================
# These define exactly what JSON the client sends.
# Field(...) = required. Field(None) = optional.
# Every field maps directly to a feature the ML model uses.
# ============================================================

class PredictionInput(BaseModel):
    timestamp: str = Field(..., description="Target time (e.g., '2018-05-01 12:00:00')")
    generation_solar: float = Field(0.0, alias="generation solar")
    generation_wind_onshore: float = Field(0.0, alias="generation wind onshore")
    generation_nuclear: float = Field(0.0, alias="generation nuclear")
    generation_fossil_gas: float = Field(0.0, alias="generation fossil gas")
    generation_fossil_hard_coal: float = Field(0.0, alias="generation fossil hard coal")
    generation_hydro_water_reservoir: float = Field(0.0, alias="generation hydro water reservoir")
    forecast_wind_onshore_day_ahead: Optional[float] = Field(None, alias="forecast wind onshore day ahead")
    forecast_solar_day_ahead: Optional[float] = Field(None, alias="forecast solar day ahead")
    total_load_forecast: Optional[float] = Field(None, alias="total load forecast")
    price_day_ahead: Optional[float] = Field(None, alias="price day ahead")

    class Config:
        populate_by_name = True

DemandInput = PredictionInput
PriceInput = PredictionInput
BothInput = PredictionInput


class AssociationQuery(BaseModel):
    demand_level: Optional[str] = Field(None, description="LOW, MED, or HIGH")
    price_level: Optional[str] = Field(None, description="LOW, MED, or HIGH")
    renewable_level: Optional[str] = Field(None, description="LOW, MED, or HIGH")
    top_n: int = Field(5, ge=1, le=50)


# ============================================================
# PIPELINE HELPERS
# ============================================================
# These functions transform raw user input into the exact
# feature matrix the models expect — same transformations
# that were applied during training.
# ============================================================

def extract_features(raw_input: dict) -> dict:
    """Extract and compute all required features from raw inputs and historical data."""
    out = dict(raw_input)
    
    # 1. Time features
    try:
        ts = pd.to_datetime(out.get('timestamp', '2018-01-01 00:00:00'), utc=True)
    except Exception:
        ts = pd.to_datetime('2018-01-01 00:00:00', utc=True)
        
    out['hour'] = ts.hour
    out['day_of_week'] = ts.dayofweek
    out['month'] = ts.month
    out['is_weekend'] = 1 if ts.dayofweek >= 5 else 0
    out['hour_sin'] = np.sin(2 * np.pi * out['hour'] / 24)
    out['hour_cos'] = np.cos(2 * np.pi * out['hour'] / 24)
    out['month_sin'] = np.sin(2 * np.pi * out['month'] / 12)
    out['month_cos'] = np.cos(2 * np.pi * out['month'] / 12)

    # 2. Aggregations
    solar = out.get('generation solar', out.get('generation_solar', 0.0))
    wind = out.get('generation wind onshore', out.get('generation_wind_onshore', 0.0))
    hydro = out.get('generation hydro water reservoir', out.get('generation_hydro_water_reservoir', 0.0))
    gas = out.get('generation fossil gas', out.get('generation_fossil_gas', 0.0))
    coal = out.get('generation fossil hard coal', out.get('generation_fossil_hard_coal', 0.0))
    nuclear = out.get('generation nuclear', out.get('generation_nuclear', 0.0))

    renewable = solar + wind + hydro
    fossil = gas + coal
    total_gen = renewable + fossil + nuclear
    renewable_pct = (renewable / total_gen * 100.0) if total_gen > 0 else 0.0

    out['renewable'] = renewable
    out['fossil'] = fossil
    out['nuclear'] = nuclear
    out['renewable_pct'] = renewable_pct

    # 3. Lags and Rolling Averages from historical data
    if not HIST_DF.empty:
        idx_1h = ts - pd.Timedelta(hours=1)
        idx_24h = ts - pd.Timedelta(hours=24)
        idx_168h = ts - pd.Timedelta(hours=168)
        
        # Helper to safely extract float from dataframe
        def get_hist_val(idx, col, default=0.0):
            try:
                return float(HIST_DF.loc[idx, col])
            except KeyError:
                subset = HIST_DF.loc[:idx, col]
                if not subset.empty:
                    return float(subset.iloc[-1])
                return default
                
        out['demand_lag_1h'] = get_hist_val(idx_1h, 'total load actual', 28000.0)
        out['demand_lag_24h'] = get_hist_val(idx_24h, 'total load actual', 28000.0)
        out['demand_lag_168h'] = get_hist_val(idx_168h, 'total load actual', 28000.0)
        
        out['price_lag_1h'] = get_hist_val(idx_1h, 'price actual', 50.0)
        out['price_lag_24h'] = get_hist_val(idx_24h, 'price actual', 50.0)
        
        # 24h rolling average
        try:
            window_start = ts - pd.Timedelta(hours=24)
            window_end = ts - pd.Timedelta(hours=1)
            mask = (HIST_DF.index >= window_start) & (HIST_DF.index <= window_end)
            if mask.sum() > 0:
                out['demand_avg_24h'] = float(HIST_DF.loc[mask, 'total load actual'].mean())
                out['price_avg_24h'] = float(HIST_DF.loc[mask, 'price actual'].mean())
            else:
                out['demand_avg_24h'] = out['demand_lag_1h']
                out['price_avg_24h'] = out['price_lag_1h']
        except Exception:
            out['demand_avg_24h'] = 28000.0
            out['price_avg_24h'] = 50.0

        idx_12h = ts - pd.Timedelta(hours=12)
        out['demand_lag_12h'] = get_hist_val(idx_12h, 'total load actual', 28000.0)
        out['price_lag_12h']  = get_hist_val(idx_12h, 'price actual', 50.0)
    else:
        out['demand_lag_1h'] = 28000.0
        out['demand_lag_24h'] = 28000.0
        out['demand_lag_168h'] = 28000.0
        out['demand_lag_12h'] = 28000.0
        out['price_lag_1h'] = 50.0
        out['price_lag_24h'] = 50.0
        out['price_lag_12h'] = 50.0
        out['demand_avg_24h'] = 28000.0
        out['price_avg_24h'] = 50.0

    # Forecast fields — use provided values or fall back to computed estimates
    out['forecast wind onshore day ahead'] = raw_input.get('forecast wind onshore day ahead',
                                              raw_input.get('forecast_wind_onshore_day_ahead',
                                              out.get('renewable', 0.0) * 0.4))
    out['forecast solar day ahead']        = raw_input.get('forecast solar day ahead',
                                              raw_input.get('forecast_solar_day_ahead',
                                              out.get('renewable', 0.0) * 0.3))
    out['total load forecast']             = raw_input.get('total load forecast',
                                              raw_input.get('total_load_forecast',
                                              out.get('demand_lag_1h', 28000.0)))

    return out

def build_feature_row(raw: dict, feature_list: list) -> pd.DataFrame:
    # Instead of the complex alias_map loop, use a simple comprehension
    # This ensures that the DataFrame columns exactly match what the model was trained on
    row = {feat: raw.get(feat, 0.0) for feat in feature_list}
    
    # Ensure no None values leak through
    for key in row:
        if row[key] is None:
            row[key] = 0.0
            
    return pd.DataFrame([row])[feature_list]


def predict_demand(raw: dict) -> float:
    raw = extract_features(raw)
    X   = build_feature_row(raw, demand_features)
    return float(reg_demand.predict(X)[0])


def predict_price(raw: dict) -> float:
    raw = extract_features(raw)
    X   = build_feature_row(raw, price_features)
    return float(reg_price.predict(X)[0])


def classify_demand(raw: dict) -> str:
    raw = extract_features(raw)
    X   = build_feature_row(raw, demand_features)
    enc = clf_demand.predict(X)[0]
    return str(le_demand.inverse_transform([enc])[0])


def classify_price(raw: dict) -> str:
    raw = extract_features(raw)
    X   = build_feature_row(raw, price_features)
    enc = clf_price.predict(X)[0]
    return str(le_price.inverse_transform([enc])[0])


def get_cluster(raw: dict) -> int:
    cols = ['hour_sin', 'hour_cos', 'renewable_pct', 'demand_lag_1h', 'price_lag_1h'] # <-- CHECK THIS
    raw  = extract_features(raw)
    vec  = np.array([[raw.get(c, 0.0) for c in cols]])
    vec_scaled = cluster_scaler.transform(vec) # This fails if shapes don't match
    return int(kmeans_model.predict(vec_scaled)[0])


def recommend_from_raw(raw: dict, k: int = 5) -> dict:
    """
    Full recommendation pipeline for a single new input:
      1. Regression  → predicted demand & price
      2. Classifier  → demand class & price class
      3. Clustering  → which regime cluster
      4. Build profile vector, L2-normalise
      5. Cosine similarity against all stored test-set profiles
      6. Top-k neighbours → majority vote → recommendation
    """
    raw = extract_features(raw)

    pred_d = predict_demand(raw)
    pred_p = predict_price(raw)
    cls_d  = classify_demand(raw)
    cls_p  = classify_price(raw)
    cl     = get_cluster(raw)

    query_vec = np.array([[
    pred_d,
    pred_p,
    label_map.get(cls_d, 1),
    label_map.get(cls_p, 1),
    cl,
    raw.get('hour_sin',      0.0),
    raw.get('hour_cos',      0.0),
    raw.get('month_sin',     0.0),
    raw.get('month_cos',     0.0),
    raw.get('renewable_pct', 0.0),
    raw.get('is_weekend',    0),
]], dtype=float)

    query_norm = normalize(query_vec, norm='l2')
    sims       = cosine_similarity(query_norm, profiles_norm)[0]
    top_k      = np.argsort(sims)[-k:][::-1]
    neighbors  = profile_df.iloc[top_k]

    dom_d  = inv_map[int(neighbors['demand_class_enc'].mode()[0])]
    dom_p  = inv_map[int(neighbors['price_class_enc'].mode()[0])]
    agree  = ((neighbors['demand_class_enc'] == label_map[dom_d]) &
               (neighbors['price_class_enc']  == label_map[dom_p])).sum()

    return {
        "predicted_demand_12h_MW"  : round(pred_d, 1),
        "predicted_price_12h_EUR"  : round(pred_p, 2),
        "demand_class"             : dom_d,
        "price_class"              : dom_p,
        "cluster"                  : cl,
        "recommendation"           : REC_TABLE.get((dom_d, dom_p), "No recommendation available."),
        "confidence"               : round(float(agree / k), 2),
        "similar_hours"            : [str(t) for t in profile_df.index[top_k].tolist()],
        "top_similarities"         : [round(float(s), 4) for s in sims[top_k].tolist()],
    }


# ============================================================
# APP
# ============================================================

# ============================================================
# HEALTH
# ============================================================

@app.get("/health", tags=["Health"])
def health():
    return {
        "status"            : "ok",
        "demand_features"   : len(demand_features),
        "price_features"    : len(price_features),
        "test_set_profiles" : len(profile_df),
        "n_clusters"        : get_cluster_count(),
    }


# ============================================================
# REGRESSION ENDPOINTS
# ============================================================

@app.post("/predict/demand", tags=["Regression"])
def predict_demand_endpoint(data: DemandInput):
    """
    Predict total electricity demand 12 hours from now (MW).
    Uses the best tuned XGBoost regressor.
    """
    raw  = data.dict(by_alias=True)
    pred = predict_demand(raw)
    return {
        "predicted_demand_12h_MW": round(pred, 1),
        "unit": "MW",
        "horizon": "12 hours",
    }


@app.post("/predict/price", tags=["Regression"])
def predict_price_endpoint(data: PriceInput):
    """
    Predict electricity spot price 12 hours from now (EUR/MWh).
    Uses the best tuned XGBoost regressor.
    """
    raw  = data.dict(by_alias=True)
    pred = predict_price(raw)
    return {
        "predicted_price_12h_EUR": round(pred, 2),
        "unit": "EUR/MWh",
        "horizon": "12 hours",
    }


@app.post("/predict/both", tags=["Regression"])
def predict_both(data: BothInput):
    """
    Predict both demand and price 12h ahead in a single call.
    More efficient than calling /predict/demand and /predict/price
    separately because cyclic features are computed once.
    """
    raw    = data.dict(by_alias=True)
    demand = round(predict_demand(raw), 1)
    price  = round(predict_price(raw),  2)
    return {
        "predicted_demand_12h_MW": demand,
        "predicted_price_12h_EUR": price,
        "unit_demand": "MW",
        "unit_price":  "EUR/MWh",
        "horizon":     "12 hours",
    }


# ============================================================
# CLASSIFICATION ENDPOINTS
# ============================================================

@app.post("/classify/demand", tags=["Classification"])
def classify_demand_endpoint(data: DemandInput):
    """
    Classify whether 12h-ahead demand will be Low, Medium, or High.
    Thresholds are the 33rd and 66th percentiles of the training data.
    """
    raw = data.dict(by_alias=True)
    return {
        "demand_class"  : classify_demand(raw),
        "thresholds_MW" : {
            "low_below"   : round(thresholds['d33'], 0),
            "high_above"  : round(thresholds['d66'], 0),
        },
    }


@app.post("/classify/price", tags=["Classification"])
def classify_price_endpoint(data: PriceInput):
    """
    Classify whether 12h-ahead price will be Low, Medium, or High.
    """
    raw = data.dict(by_alias=True)
    return {
        "price_class"       : classify_price(raw),
        "thresholds_EUR_MWh": {
            "low_below"  : round(thresholds['p33'], 2),
            "high_above" : round(thresholds['p66'], 2),
        },
    }


# ============================================================
# CLUSTERING ENDPOINT
# ============================================================

@app.post("/cluster", tags=["Clustering"])
def cluster_endpoint(data: DemandInput):
    """
    Assign this hour to an energy regime cluster.
    The cluster ID indicates which group of historically similar
    hours this hour belongs to (based on time-of-day, renewable
    mix, and recent demand/price levels).
    Interpretation: see /clusters/profiles for what each cluster means.
    """
    raw = data.dict(by_alias=True)
    cl  = get_cluster(raw)
    return {
        "cluster_id"  : cl,
        "n_clusters"  : get_cluster_count(),
        "note"        : "See /clusters/profiles for mean demand/price per cluster.",
    }


@app.get("/clusters/profiles", tags=["Clustering"])
def cluster_profiles():
    """
    Mean demand and price 12h-ahead per cluster on the test set.
    Use this to interpret what each cluster number means
    (e.g., cluster 0 = overnight low renewable, cluster 2 = weekday morning peak).
    """
    cols  = ['demand_class_enc', 'price_class_enc', 'cluster',
             'pred_demand', 'pred_price', 'renewable_pct']
    avail = [c for c in cols if c in profile_df.columns]
    # Ensure 'cluster' appears only once in the selection to avoid
    # pandas grouping errors when duplicate column names are present.
    if 'cluster' not in avail:
        sel = avail + ['cluster']
    else:
        sel = avail
    prof  = profile_df[sel].groupby('cluster').mean().round(2)
    return prof.to_dict(orient='index')


# ============================================================
# RECOMMENDATION ENDPOINTS
# ============================================================

@app.post("/recommend", tags=["Recommendation"])
def recommend(data: BothInput, k: int = 5):
    """
    Full recommendation pipeline in one call.

    Steps performed internally:
      1. Predict demand and price 12h ahead (regression)
      2. Classify demand and price regime (Low/Medium/High)
      3. Assign energy regime cluster (KMeans)
      4. Build L2-normalised profile vector
      5. Cosine similarity against all stored test-set profiles
      6. Top-k most similar hours → majority vote on regime
      7. Lookup recommendation from (demand_class, price_class) table

    Returns predicted values, regime classification, cluster,
    a human-readable recommendation, confidence score, and
    the timestamps of the most similar historical hours.
    """
    raw = data.dict(by_alias=True)
    try:
        result = recommend_from_raw(raw, k=k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


@app.get("/recommend/by_index/{idx}", tags=["Recommendation"])
def recommend_by_index(idx: int, k: int = 5):
    """
    Get a recommendation for a specific hour in the test set by
    its positional index (0-based). Useful for testing and
    validating the recommendation engine against known outcomes.
    """
    n = len(profiles_norm)
    if idx < 0 or idx >= n:
        raise HTTPException(
            status_code=400,
            detail=f"Index {idx} out of range. Test set has {n} rows (0..{n-1})."
        )

    query_vec       = profiles_norm[idx].reshape(1, -1)
    sims            = cosine_similarity(query_vec, profiles_norm)[0]
    sims[idx]       = -1
    top_k           = np.argsort(sims)[-k:][::-1]
    neighbors       = profile_df.iloc[top_k]
    dom_d           = inv_map[int(neighbors['demand_class_enc'].mode()[0])]
    dom_p           = inv_map[int(neighbors['price_class_enc'].mode()[0])]
    agree           = ((neighbors['demand_class_enc'] == label_map[dom_d]) &
                       (neighbors['price_class_enc']  == label_map[dom_p])).sum()

    row = profile_df.iloc[idx]
    return {
        "index"                    : idx,
        "timestamp"                : str(profile_df.index[idx]),
        "predicted_demand_12h_MW"  : round(float(row['pred_demand']), 1),
        "predicted_price_12h_EUR"  : round(float(row['pred_price']),  2),
        "demand_class"             : inv_map[int(row['demand_class_enc'])],
        "price_class"              : inv_map[int(row['price_class_enc'])],
        "cluster"                  : int(row['cluster']),
        "recommendation"           : REC_TABLE.get((dom_d, dom_p), "N/A"),
        "confidence"               : round(float(agree / k), 2),
        "similar_hours"            : [str(t) for t in profile_df.index[top_k].tolist()],
        "top_similarities"         : [round(float(s), 4) for s in sims[top_k].tolist()],
    }


# -------------------- Association rules (from artifacts) --------------------
@app.get("/associations/top", tags=["Association Rules"])
def get_top_associations(n: int = 10):
    try:
        if assoc_rules is None or len(assoc_rules) == 0:
            # fallback: try reading CSV on demand
            csv_path = os.path.join(ARTIFACTS, 'association_rules.csv')
            if not os.path.exists(csv_path):
                raise HTTPException(status_code=404, detail="Association rules not available. Run association mining.")
            df = pd.read_csv(csv_path)
            df = _sanitize_assoc_df(df)
        else:
            df = assoc_rules
        top = df.head(n).copy()
        # Ensure numeric types
        for c in ['support', 'confidence', 'lift']:
            if c in top.columns:
                try:
                    top[c] = top[c].astype(float)
                except Exception:
                    pass
        return top[['antecedents','consequents','support','confidence','lift']].to_dict(orient='records')
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"associations/top error: {e}")


@app.post("/associations/query", tags=["Association Rules"])
def query_associations(query: AssociationQuery):
    try:
        from src.association_rules_endpoint import query_rules
        if assoc_rules is None or len(assoc_rules) == 0:
            csv_path = os.path.join(ARTIFACTS, 'association_rules.csv')
            if not os.path.exists(csv_path):
                raise HTTPException(status_code=404, detail="Association rules not available. Run association mining.")
            df = pd.read_csv(csv_path)
            df = _sanitize_assoc_df(df)
        else:
            df = assoc_rules
        results = query_rules(df, demand_level=query.demand_level, price_level=query.price_level, renewable_level=query.renewable_level, top_n=query.top_n)
        return {'query': query.dict(), 'rules_found': len(results), 'rules': results}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/associations/debug', tags=["Association Rules"])
def associations_debug():
    """Return basic diagnostics about the association_rules.csv file for debugging."""
    csv_path = os.path.join(ARTIFACTS, 'association_rules.csv')
    if not os.path.exists(csv_path) and (assoc_rules is None or len(assoc_rules) == 0):
        raise HTTPException(status_code=404, detail='association_rules.csv missing')
    info = {}
    try:
        if assoc_rules is not None and len(assoc_rules) > 0:
            df = assoc_rules
        else:
            df = pd.read_csv(csv_path)
        info['rows'] = int(len(df))
        info['columns'] = df.columns.tolist()
        first = df.iloc[0].to_dict()
        sample = {}
        for k, v in first.items():
            sample[k] = {'repr': str(v)[:300], 'type': type(v).__name__}
        info['first_row_sample'] = sample
    except Exception as e:
        info['error'] = str(e)
    return info


# ============================================================
# MODEL INFO ENDPOINT
# ============================================================

@app.get("/models/compare", tags=["Info"])
def model_comparison():
    """
    Returns a summary of what models were trained and the
    feature lists used. Run model_training.py to see actual
    R2 scores printed to console during training.
    """
    return {
        "regression_model_demand" : type(reg_demand).__name__,
        "regression_model_price"  : type(reg_price).__name__,
        "classifier_demand"       : type(clf_demand).__name__,
        "classifier_price"        : type(clf_price).__name__,
        "clustering_model"        : type(kmeans_model).__name__,
        "n_clusters"              : get_cluster_count(),
        "demand_features_count"   : len(demand_features),
        "price_features_count"    : len(price_features),
        "demand_features"         : demand_features,
        "price_features"          : price_features,
        "thresholds"              : {
            "demand_low_below_MW"    : round(float(thresholds['d33']), 0) if np.isfinite(float(thresholds['d33'])) else 0,
            "demand_high_above_MW"   : round(float(thresholds['d66']), 0) if np.isfinite(float(thresholds['d66'])) else 0,
            "price_low_below_EUR"    : round(float(thresholds['p33']), 2) if np.isfinite(float(thresholds['p33'])) else 0,
            "price_high_above_EUR"   : round(float(thresholds['p66']), 2) if np.isfinite(float(thresholds['p66'])) else 0,
        },
    }


@app.get("/models/compare_metrics", tags=["Info"])
def model_comparison_metrics():
    """Return per-model performance metrics read from artifacts/model_comparison.csv."""
    csv_path = os.path.join(ARTIFACTS, 'model_comparison.csv')
    if not os.path.exists(csv_path):
                return []
    df = pd.read_csv(csv_path)
    # Convert to dict: { model_name: {metric: value, ...}, ... }
    out = {}
    for _, row in df.iterrows():
        out[row['Model']] = {
            'demand_r2': float(row.get('Demand R2', 0)),
            'demand_nmae': float(row.get('Demand NMAE', 0)),
            'price_r2': float(row.get('Price R2', 0)),
            'price_nmae': float(row.get('Price NMAE', 0)),
            'avg_r2': float(row.get('Avg R2', 0)),
        }
    return out


@app.get("/recommend", tags=["Info"])
def recommend_best_model():
    """Return the best model based on Avg R2 from model_comparison.csv."""
    csv_path = os.path.join(ARTIFACTS, 'model_comparison.csv')
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="model_comparison.csv not found in artifacts")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise HTTPException(status_code=404, detail="No model comparison data available")
    best = df.loc[df['Avg R2'].idxmax()]
    return {
        'best_model': best['Model'],
        'best_score': float(best['Avg R2']),
        'reason': 'Highest Avg R2 across demand and price'
    }


@app.get("/")
def home():
    return {"status": "Online", "message": "Energy API is running. Visit /docs for the UI."}

@app.post("/predict/all")
def predict_all(input_data: BothInput):
    # Convert input to dictionary
    data_dict = input_data.model_dump(by_alias=True)
    
    # 1. Prepare features for Demand
    # We filter the dictionary to only include features the model was trained on
    d_feats = {f: data_dict.get(f, 0) for f in demand_features}
    d_df = pd.DataFrame([d_feats])
    
    # 2. Prepare features for Price
    p_feats = {f: data_dict.get(f, 0) for f in price_features}
    p_df = pd.DataFrame([p_feats])
    
    # 3. Predict
    pred_d = float(reg_demand.predict(d_df)[0])
    pred_p = float(reg_price.predict(p_df)[0])
    
    # 4. Classification logic using your thresholds
    d33, d66 = thresholds['d33'], thresholds['d66']
    p33, p66 = thresholds['p33'], thresholds['p66']
    
    def get_label(val, t1, t2):
        if val <= t1: return "Low"
        if val <= t2: return "Medium"
        return "High"
    
    d_class = get_label(pred_d, d33, d66)
    p_class = get_label(pred_p, p33, p66)
    
    # 5. Get Recommendation from your REC_TABLE
    recommendation = REC_TABLE.get((d_class, p_class), "Maintain standard operations.")
    
    return {
        "predictions": {
            "demand_mw": round(pred_d, 2),
            "price_eur": round(pred_p, 2)
        },
        "status": {
            "demand_level": d_class,
            "price_level": p_class
        },
        "recommendation": recommendation
    }


@app.get("/", tags=["Home"])
def read_root():
    return {
        "Project": "Energy Forecasting API",
        "Status": "Running",
        "Endpoints": ["/docs", "/redoc", "/predict/all"]
    }
