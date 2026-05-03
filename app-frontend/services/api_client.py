"""
Centralized API client for the Electricity Forecasting FastAPI backend.
"""
import requests
import streamlit as st
from utils.config import API_BASE_URL, API_TIMEOUT


def _get(endpoint: str, params: dict = None) -> dict | None:
    try:
        r = requests.get(
            f"{API_BASE_URL}{endpoint}",
            params=params,
            timeout=API_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.warning(f"FastAPI backend is not reachable at {API_BASE_URL}. Make sure the API container is running.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def _post(endpoint: str, payload: dict) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE_URL}{endpoint}",
            json=payload,
            timeout=API_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.warning(f"FastAPI backend is not reachable at {API_BASE_URL}.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


# --- Health ---
@st.cache_data(ttl=30)
def get_health() -> dict | None:
    return _get("/health")


# --- Predictions ---
def predict_both(features: dict) -> dict | None:
    return _post("/predict/both", features)

def predict_demand(features: dict) -> dict | None:
    return _post("/predict/demand", features)

def predict_price(features: dict) -> dict | None:
    return _post("/predict/price", features)


# --- Classification ---
def classify_demand(features: dict) -> dict | None:
    return _post("/classify/demand", features)

def classify_price(features: dict) -> dict | None:
    return _post("/classify/price", features)


# --- Clustering ---
def get_cluster(features: dict) -> dict | None:
    return _post("/cluster", features)

@st.cache_data(ttl=300)
def get_cluster_profiles() -> dict | None:
    return _get("/clusters/profiles")


# --- Model comparison ---
@st.cache_data(ttl=300)
def get_model_comparison() -> dict | None:
    """GET /models/compare — returns model registry info."""
    return _get("/models/compare")

@st.cache_data(ttl=300)
def get_model_registry() -> dict | None:
    return _get("/models/compare")

@st.cache_data(ttl=300)
def get_recommendation() -> dict | None:
    """GET /recommend/by_index/0 — returns a sample recommendation."""
    return _get("/recommend/by_index/0")

def recommend_by_index(idx: int, k: int = 5) -> dict | None:
    return _get(f"/recommend/by_index/{idx}", params={"k": k})

def recommend_full(features: dict, k: int = 5) -> dict | None:
    return _post(f"/recommend?k={k}", features)


# --- Association rules ---
@st.cache_data(ttl=300)
def get_top_associations(n: int = 15) -> dict | None:
    rules = _get("/associations/top", params={"n": n})
    if rules is not None:
        return rules
    dbg = _get("/associations/debug")
    if dbg is not None:
        return {"rules": [], "debug": dbg}
    return None

def query_associations(demand_level=None, price_level=None,
                       renewable_level=None, top_n=5) -> dict | None:
    payload = {"top_n": top_n}
    if demand_level:    payload["demand_level"]    = demand_level
    if price_level:     payload["price_level"]     = price_level
    if renewable_level: payload["renewable_level"] = renewable_level
    result = _post("/associations/query", payload)
    if result is not None:
        return result
    dbg = _get("/associations/debug")
    if dbg is not None:
        return {"rules": [], "rules_found": 0, "debug": dbg}
    return None