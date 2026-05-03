"""
Centralized API client for the Electricity Forecasting FastAPI backend.
All endpoints from `main_new_v2` are wrapped here.
"""

import requests
import streamlit as st
from utils.config import API_BASE_URL, API_TIMEOUT


def _get(endpoint: str, params: dict = None) -> dict | None:
    """
    Make a GET request to the API. Returns parsed JSON or None on failure.
    Displays a Streamlit warning if the API is unreachable.
    """
    try:
        r = requests.get(
            f"{API_BASE_URL}{endpoint}",
            params=params,
            timeout=API_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.warning("FastAPI backend is not reachable. Start it with: uvicorn main_new_v2:app --reload")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def _post(endpoint: str, payload: dict) -> dict | None:
    """Make a POST request to the API."""
    try:
        r = requests.post(
            f"{API_BASE_URL}{endpoint}",
            json=payload,
            timeout=API_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.warning("FastAPI backend is not reachable.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


# ── Health ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def get_health() -> dict | None:
    """GET /health — returns model load status and version info."""
    return _get("/health")


# ── Predictions ───────────────────────────────────────────────────────────────
def predict_both(features: dict) -> dict | None:
    """
    POST /predict/both
    Input: dict with keys matching FeaturesInput Pydantic model
    Returns: { demand_prediction, price_prediction }
    """
    return _post("/predict/both", features)


def predict_demand(features: dict) -> dict | None:
    """POST /predict/demand — returns { demand_prediction }"""
    return _post("/predict/demand", features)


def predict_price(features: dict) -> dict | None:
    """POST /predict/price — returns { price_prediction }"""
    return _post("/predict/price", features)


# ── Classification ────────────────────────────────────────────────────────────
def classify_demand(features: dict) -> dict | None:
    """POST /classify/demand — returns { demand_class: 'LOW'|'MED'|'HIGH' }"""
    return _post("/classify/demand", features)


def classify_price(features: dict) -> dict | None:
    """POST /classify/price — returns { price_class: 'LOW'|'MED'|'HIGH' }"""
    return _post("/classify/price", features)


# ── Clustering ───────────────────────────────────────────────────────────────
def get_cluster(features: dict) -> dict | None:
    """POST /cluster — returns { cluster_id, cluster_label }"""
    return _post("/cluster", features)


@st.cache_data(ttl=300)
def get_cluster_profiles() -> dict | None:
    """GET /clusters/profiles — returns cluster summary statistics."""
    return _get("/clusters/profiles")


# ── Model comparison ──────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_model_comparison() -> dict | None:
    """GET /models/compare_metrics — returns metrics for all trained models."""
    return _get("/models/compare_metrics")


@st.cache_data(ttl=300)
def get_model_registry() -> dict | None:
    """GET /models/compare — returns model registry, feature lists, and thresholds."""
    return _get("/models/compare")


@st.cache_data(ttl=300)
def get_recommendation() -> dict | None:
    """GET /recommend — returns best model name + reason."""
    return _get("/recommend")


def recommend_by_index(idx: int, k: int = 5) -> dict | None:
    """GET /recommend/by_index/{idx} — return recommendation details for a test row."""
    return _get(f"/recommend/by_index/{idx}", params={"k": k})


def recommend_full(features: dict, k: int = 5) -> dict | None:
    """POST /recommend — return the full recommendation pipeline output."""
    return _post(f"/recommend?k={k}", features)


# ── Association rules ─────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_top_associations(n: int = 15) -> dict | None:
    """GET /associations/top — returns top N rules by lift."""
    rules = _get("/associations/top", params={"n": n})
    if rules is not None:
        return rules

    # Graceful fallback so UI can still render diagnostics if rules endpoint fails.
    dbg = _get("/associations/debug")
    if dbg is not None:
        return {"rules": [], "debug": dbg}
    return None


def query_associations(demand_level=None, price_level=None,
                        renewable_level=None, top_n=5) -> dict | None:
    """POST /associations/query — find rules matching a given energy state."""
    payload = {"top_n": top_n}
    if demand_level:    payload["demand_level"] = demand_level
    if price_level:     payload["price_level"]  = price_level
    if renewable_level: payload["renewable_level"] = renewable_level
    result = _post("/associations/query", payload)
    if result is not None:
        return result
    dbg = _get("/associations/debug")
    if dbg is not None:
        return {"rules": [], "rules_found": 0, "debug": dbg}
    return None
