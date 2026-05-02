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
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

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
inv_map         = load('inv_map.pkl')
REC_TABLE       = load('rec_table.pkl')
thresholds      = load('thresholds.pkl')
assoc_rules     = load_optional('association_rules.pkl', pd.DataFrame())
print("All artifacts loaded.")

# ============================================================
# PYDANTIC MODELS
# ============================================================
# These define exactly what JSON the client sends.
# Field(...) = required. Field(None) = optional.
# Every field maps directly to a feature the ML model uses.
# ============================================================

class DemandInput(BaseModel):
    hour:                          int   = Field(..., ge=0, le=23,  description="Hour of day (0-23)")
    day_of_week:                   int   = Field(..., ge=0, le=6,   description="Day of week (0=Mon, 6=Sun)")
    month:                         int   = Field(..., ge=1, le=12,  description="Month (1-12)")
    is_weekend:                    int   = Field(..., ge=0, le=1)
    demand_lag_1h:                 float = Field(..., description="Demand 1 hour ago (MW)")
    demand_lag_24h:                float = Field(..., description="Demand 24 hours ago (MW)")
    demand_lag_168h:               float = Field(..., description="Demand 168 hours ago (MW)")
    price_lag_1h:                  float = Field(..., description="Price 1 hour ago (EUR/MWh)")
    price_lag_24h:                 float = Field(..., description="Price 24 hours ago (EUR/MWh)")
    renewable:                     float = Field(..., description="Total renewable generation (MW)")
    fossil:                        float = Field(..., description="Total fossil generation (MW)")
    nuclear:                       float = Field(..., description="Nuclear generation (MW)")
    renewable_pct:                 float = Field(..., description="Renewable as % of total generation")
    demand_avg_24h:                float = Field(..., description="Rolling 24h mean demand (MW)")
    price_avg_24h:                 float = Field(..., description="Rolling 24h mean price (EUR/MWh)")
    forecast_wind_onshore_day_ahead: Optional[float] = Field(None, alias="forecast wind onshore day ahead")
    forecast_solar_day_ahead:      Optional[float]   = Field(None, alias="forecast solar day ahead")
    total_load_forecast:           Optional[float]   = Field(None, alias="total load forecast")

    class Config:
        populate_by_name = True


class PriceInput(BaseModel):
    hour:                          int   = Field(..., ge=0, le=23)
    day_of_week:                   int   = Field(..., ge=0, le=6)
    month:                         int   = Field(..., ge=1, le=12)
    is_weekend:                    int   = Field(..., ge=0, le=1)
    price_lag_1h:                  float
    price_lag_24h:                 float
    demand_lag_1h:                 float
    demand_lag_24h:                float
    renewable:                     float
    fossil:                        float
    nuclear:                       float
    renewable_pct:                 float
    price_avg_24h:                 float
    demand_avg_24h:                float
    forecast_wind_onshore_day_ahead: Optional[float] = Field(None, alias="forecast wind onshore day ahead")
    forecast_solar_day_ahead:      Optional[float]   = Field(None, alias="forecast solar day ahead")
    total_load_forecast:           Optional[float]   = Field(None, alias="total load forecast")
    price_day_ahead:               Optional[float]   = Field(None, alias="price day ahead")

    class Config:
        populate_by_name = True


class BothInput(BaseModel):
    """Combined input — contains all fields needed for both models."""
    hour:                          int   = Field(..., ge=0, le=23)
    day_of_week:                   int   = Field(..., ge=0, le=6)
    month:                         int   = Field(..., ge=1, le=12)
    is_weekend:                    int   = Field(..., ge=0, le=1)
    demand_lag_1h:                 float
    demand_lag_24h:                float
    demand_lag_168h:               float
    price_lag_1h:                  float
    price_lag_24h:                 float
    renewable:                     float
    fossil:                        float
    nuclear:                       float
    renewable_pct:                 float
    demand_avg_24h:                float
    price_avg_24h:                 float
    forecast_wind_onshore_day_ahead: Optional[float] = Field(None, alias="forecast wind onshore day ahead")
    forecast_solar_day_ahead:      Optional[float]   = Field(None, alias="forecast solar day ahead")
    total_load_forecast:           Optional[float]   = Field(None, alias="total load forecast")
    price_day_ahead:               Optional[float]   = Field(None, alias="price day ahead")

    class Config:
        populate_by_name = True


class AssociationQuery(BaseModel):
    demand_level: Optional[str] = Field(None, description="LOW, MED, or HIGH")
    price_level: Optional[str] = Field(None, description="LOW, MED, or HIGH")
    renewable_level: Optional[str] = Field(None, description="LOW, MED, or HIGH")
    top_n: int = Field(5, ge=1, le=20)


# ============================================================
# PIPELINE HELPERS
# ============================================================
# These functions transform raw user input into the exact
# feature matrix the models expect — same transformations
# that were applied during training.
# ============================================================

def add_cyclic_time(row: dict) -> dict:
    """Add hour_sin, hour_cos, month_sin, month_cos from raw hour/month."""
    row['hour_sin']  = np.sin(2 * np.pi * row['hour']  / 24)
    row['hour_cos']  = np.cos(2 * np.pi * row['hour']  / 24)
    row['month_sin'] = np.sin(2 * np.pi * row['month'] / 12)
    row['month_cos'] = np.cos(2 * np.pi * row['month'] / 12)
    return row


def build_feature_row(raw: dict, feature_list: list) -> pd.DataFrame:
    """
    Build a single-row DataFrame aligned to feature_list.
    Missing optional features are filled with 0 (safe default).
    The aliases (spaces in names) are handled by mapping the
    underscore versions to their original names.
    """
    alias_map = {
        'forecast_wind_onshore_day_ahead': 'forecast wind onshore day ahead',
        'forecast_solar_day_ahead':        'forecast solar day ahead',
        'total_load_forecast':             'total load forecast',
        'price_day_ahead':                 'price day ahead',
    }
    row = {}
    for feat in feature_list:
        # Try direct match first
        if feat in raw:
            row[feat] = raw[feat]
            continue
        # Try underscore alias
        for alias, original in alias_map.items():
            if original == feat and alias in raw:
                row[feat] = raw[alias] if raw[alias] is not None else 0.0
                break
        else:
            row[feat] = 0.0   # optional feature not supplied
    return pd.DataFrame([row])[feature_list]


def predict_demand(raw: dict) -> float:
    raw = add_cyclic_time(raw)
    X   = build_feature_row(raw, demand_features)
    return float(reg_demand.predict(X)[0])


def predict_price(raw: dict) -> float:
    raw = add_cyclic_time(raw)
    X   = build_feature_row(raw, price_features)
    return float(reg_price.predict(X)[0])


def classify_demand(raw: dict) -> str:
    raw = add_cyclic_time(raw)
    X   = build_feature_row(raw, demand_features)
    enc = clf_demand.predict(X)[0]
    return str(le_demand.inverse_transform([enc])[0])


def classify_price(raw: dict) -> str:
    raw = add_cyclic_time(raw)
    X   = build_feature_row(raw, price_features)
    enc = clf_price.predict(X)[0]
    return str(le_price.inverse_transform([enc])[0])


def get_cluster(raw: dict) -> int:
    cols = ['hour_sin', 'hour_cos', 'renewable_pct', 'demand_lag_1h', 'price_lag_1h']
    raw  = add_cyclic_time(raw)
    vec  = np.array([[raw.get(c, 0.0) for c in cols]])
    vec_scaled = cluster_scaler.transform(vec)
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
    raw = add_cyclic_time(raw)

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
        raw.get('hour_sin',     0.0),
        raw.get('hour_cos',     0.0),
        raw.get('month_sin',    0.0),
        raw.get('month_cos',    0.0),
        raw.get('renewable_pct',0.0),
        raw.get('is_weekend',   0),
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

app = FastAPI(
    title="Energy Forecast API",
    description=(
        "12-hour ahead demand and price forecasting.\n\n"
        "**Pipeline**: run `model_training.py` first to generate "
        "`artifacts/`, then start this server.\n\n"
        "All models are loaded once at startup. Endpoints return "
        "regression forecasts, classification regimes, cluster "
        "assignments, and cosine-similarity-based recommendations."
    ),
    version="1.0.0",
)


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
        "n_clusters"        : int(thresholds['best_k']),
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
        "n_clusters"  : int(thresholds['best_k']),
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
    prof  = profile_df[avail + ['cluster']].groupby('cluster').mean().round(2)
    return prof.to_dict(orient='index')


# ============================================================
# ASSOCIATION RULES ENDPOINTS
# ============================================================

@app.post("/associations/query", tags=["Association Rules"])
def get_associations(query: AssociationQuery):
    """
    Find association rules matching a given energy state.
    For example: given demand=HIGH and renewable=LOW, what price rules apply?
    """
    try:
        if assoc_rules is None or len(assoc_rules) == 0:
            raise HTTPException(status_code=404, detail="Association rules are not available. Run src/association_rules_mining.py first.")
        from src.association_rules_endpoint import query_rules
        results = query_rules(
            assoc_rules,
            demand_level=query.demand_level,
            price_level=query.price_level,
            renewable_level=query.renewable_level,
            top_n=query.top_n
        )
        return {
            'query': query.dict(),
            'rules_found': len(results),
            'rules': results
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/associations/top", tags=["Association Rules"])
def get_top_associations(n: int = 10):
    """Return top N association rules by lift."""
    if assoc_rules is None or len(assoc_rules) == 0:
        raise HTTPException(status_code=404, detail="Association rules are not available. Run src/association_rules_mining.py first.")
    top = assoc_rules.head(n).copy()
    top['antecedents'] = top['antecedents'].apply(lambda x: sorted(list(x)))
    top['consequents'] = top['consequents'].apply(lambda x: sorted(list(x)))
    return top[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_dict(orient='records')


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
        "n_clusters"              : int(thresholds['best_k']),
        "demand_features_count"   : len(demand_features),
        "price_features_count"    : len(price_features),
        "demand_features"         : demand_features,
        "price_features"          : price_features,
        "thresholds"              : {
            "demand_low_below_MW"    : round(thresholds['d33'], 0),
            "demand_high_above_MW"   : round(thresholds['d66'], 0),
            "price_low_below_EUR"    : round(thresholds['p33'], 2),
            "price_high_above_EUR"   : round(thresholds['p66'], 2),
        },
    }
