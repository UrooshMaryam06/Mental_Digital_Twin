import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_TIMEOUT  = 10

COLORS = {
    "bg_primary":    "#0f1117",
    "bg_secondary":  "#1a1d27",
    "bg_card":       "#1e2130",
    "border":        "#2e3347",
    "text_primary":  "#e8eaf0",
    "text_secondary":"#8b91a8",
    "accent_amber":  "#f5a623",
    "accent_teal":   "#26c6da",
    "accent_red":    "#ef5350",
    "accent_green":  "#66bb6a",
    "demand_color":  "#26c6da",
    "price_color":   "#f5a623",
    "low_color":     "#66bb6a",
    "med_color":     "#ffa726",
    "high_color":    "#ef5350",
}

DATA_PATH = os.getenv("DATA_PATH", "../energy_dataset.csv")

MODEL_NAMES = [
    "Linear Regression", "Bayesian Ridge", "KNN",
    "Decision Tree", "Random Forest", "XGBoost", "SVR", "ANN",
]

GENERATION_FEATURES = [
    "generation biomass", "generation fossil gas",
    "generation fossil hard coal",
    "generation hydro run-of-river and poundage",
    "generation hydro water reservoir", "generation nuclear",
    "generation solar", "generation wind onshore",
]