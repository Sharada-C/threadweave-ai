"""
ThreadWeave AI - Global configuration
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# --- Input files ---
SALES_HISTORY_CSV = os.path.join(DATA_DIR, "fashion_sales_history.csv")
SUPPLIER_DATA_CSV = os.path.join(DATA_DIR, "supplier_data.csv")
INVENTORY_CSV = os.path.join(DATA_DIR, "inventory.csv")
PRODUCT_CATALOG_CSV = os.path.join(DATA_DIR, "product_catalog.csv")

# --- Output files (agent results) ---
TREND_SCORES_CSV = os.path.join(OUTPUT_DIR, "trend_scores.csv")
DEMAND_FORECAST_CSV = os.path.join(OUTPUT_DIR, "demand_forecast.csv")
RISK_ANALYSIS_CSV = os.path.join(OUTPUT_DIR, "risk_analysis.csv")
COST_OPTIMIZATION_CSV = os.path.join(OUTPUT_DIR, "cost_optimization.csv")

# --- LLM config ---
OLLAMA_MODEL = os.environ.get("THREADWEAVE_LLM_MODEL", "qwen2.5-coder:7b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_TIMEOUT_SECONDS = 8          # fail fast and fall back rather than hang
LLM_TEMPERATURE = 0.2

# --- Forecasting config ---
FORECAST_HORIZON_DAYS = 30
FORECAST_MODELS = [
    "ARIMA",
    "SARIMA",
    "Holt-Winters",
    "Prophet",
    "XGBoost Lag",
    "Random Forest",
    "Linear Regression",
]

# --- Risk config ---
SERVICE_LEVEL_Z = 1.645  # z-score for 95% service level

# --- Cost/procurement config ---
ORDERING_COST_INR = 500.0
HOLDING_COST_PCT = 0.18
OVERAGE_COST_PCT = 0.30
DEFAULT_BUDGET_MULTIPLIER = 0.85  # fallback if LLM/portfolio step unavailable

# --- Agent dependency graph (Supervisor uses this) ---
AGENT_DEPENDENCIES = {
    "trend": [],
    "demand": ["trend"],
    "risk": ["demand"],
    "cost": ["trend", "demand", "risk"],
}

AGENT_OUTPUT_FILES = {
    "trend": TREND_SCORES_CSV,
    "demand": DEMAND_FORECAST_CSV,
    "risk": RISK_ANALYSIS_CSV,
    "cost": COST_OPTIMIZATION_CSV,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
