"""
ThreadWeave AI - Forecasting model library

Implements the seven forecasting models named in the report:
ARIMA, SARIMA, Holt-Winters, Prophet, XGBoost (lag features), Random Forest,
Linear Regression.

Each forecast_* function takes a weekly pandas Series (indexed by date,
ascending) and a horizon in weeks, and returns a numpy array of length
`horizon_weeks` with the forecasted weekly units.

All functions are defensive: on failure (e.g. too little history, model
non-convergence) they fall back to a naive forecast (last value / recent
mean) rather than raising, so the pipeline never crashes on a single
product.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
    _PROPHET_AVAILABLE = True
except Exception:
    _PROPHET_AVAILABLE = False


def _naive_forecast(series: pd.Series, horizon_weeks: int) -> np.ndarray:
    recent_mean = series.tail(8).mean() if len(series) >= 1 else 0.0
    return np.full(horizon_weeks, max(recent_mean, 0.0))


def forecast_arima(series: pd.Series, horizon_weeks: int) -> np.ndarray:
    from statsmodels.tsa.arima.model import ARIMA
    try:
        model = ARIMA(series.values, order=(2, 1, 2))
        fit = model.fit()
        forecast = fit.forecast(steps=horizon_weeks)
        return np.clip(forecast, 0, None)
    except Exception:
        return _naive_forecast(series, horizon_weeks)


def forecast_sarima(series: pd.Series, horizon_weeks: int, seasonal_period: int = 52) -> np.ndarray:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    try:
        sp = seasonal_period if len(series) >= 2 * seasonal_period else 0
        order = (1, 1, 1)
        seasonal_order = (1, 1, 0, sp) if sp else (0, 0, 0, 0)
        model = SARIMAX(series.values, order=order, seasonal_order=seasonal_order,
                         enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
        forecast = fit.forecast(steps=horizon_weeks)
        return np.clip(forecast, 0, None)
    except Exception:
        return _naive_forecast(series, horizon_weeks)


def forecast_holt_winters(series: pd.Series, horizon_weeks: int, seasonal_period: int = 52) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    try:
        sp = seasonal_period if len(series) >= 2 * seasonal_period else None
        seasonal = "add" if sp else None
        model = ExponentialSmoothing(series.values, trend="add",
                                      seasonal=seasonal, seasonal_periods=sp)
        fit = model.fit()
        forecast = fit.forecast(horizon_weeks)
        return np.clip(forecast, 0, None)
    except Exception:
        return _naive_forecast(series, horizon_weeks)


def forecast_prophet(series: pd.Series, horizon_weeks: int) -> np.ndarray:
    if not _PROPHET_AVAILABLE:
        return _naive_forecast(series, horizon_weeks)
    try:
        df = pd.DataFrame({"ds": series.index, "y": series.values})
        model = Prophet(weekly_seasonality=False, yearly_seasonality=True, daily_seasonality=False)
        model.fit(df)
        future = model.make_future_dataframe(periods=horizon_weeks, freq="W")
        fcst = model.predict(future)
        forecast = fcst["yhat"].tail(horizon_weeks).values
        return np.clip(forecast, 0, None)
    except Exception:
        return _naive_forecast(series, horizon_weeks)


def _make_lag_features(series: pd.Series, lags=(1, 2, 4, 8)):
    df = pd.DataFrame({"y": series.values})
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["rolling_mean_4"] = df["y"].shift(1).rolling(4).mean()
    df["t"] = np.arange(len(df))
    df = df.dropna()
    return df, lags


def _recursive_tree_forecast(model_cls, series: pd.Series, horizon_weeks: int, lags=(1, 2, 4, 8)):
    try:
        df, lag_list = _make_lag_features(series, lags)
        feature_cols = [c for c in df.columns if c != "y"]
        X, y = df[feature_cols], df["y"]
        model = model_cls()
        model.fit(X, y)

        history = list(series.values)
        preds = []
        t_counter = len(series)
        for _ in range(horizon_weeks):
            row = {}
            for lag in lag_list:
                row[f"lag_{lag}"] = history[-lag]
            row["rolling_mean_4"] = np.mean(history[-4:])
            row["t"] = t_counter
            X_next = pd.DataFrame([row])[feature_cols]
            pred = max(float(model.predict(X_next)[0]), 0.0)
            preds.append(pred)
            history.append(pred)
            t_counter += 1
        return np.array(preds)
    except Exception:
        return _naive_forecast(series, horizon_weeks)


def forecast_xgboost(series: pd.Series, horizon_weeks: int) -> np.ndarray:
    from xgboost import XGBRegressor
    model_cls = lambda: XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.1, verbosity=0)
    return _recursive_tree_forecast(model_cls, series, horizon_weeks)


def forecast_random_forest(series: pd.Series, horizon_weeks: int) -> np.ndarray:
    from sklearn.ensemble import RandomForestRegressor
    model_cls = lambda: RandomForestRegressor(n_estimators=150, max_depth=4, random_state=42)
    return _recursive_tree_forecast(model_cls, series, horizon_weeks)


def forecast_linear_regression(series: pd.Series, horizon_weeks: int) -> np.ndarray:
    from sklearn.linear_model import LinearRegression
    try:
        t = np.arange(len(series)).reshape(-1, 1)
        y = series.values
        model = LinearRegression().fit(t, y)
        future_t = np.arange(len(series), len(series) + horizon_weeks).reshape(-1, 1)
        forecast = model.predict(future_t)
        return np.clip(forecast, 0, None)
    except Exception:
        return _naive_forecast(series, horizon_weeks)


MODEL_DISPATCH = {
    "ARIMA": forecast_arima,
    "SARIMA": forecast_sarima,
    "Holt-Winters": forecast_holt_winters,
    "Prophet": forecast_prophet,
    "XGBoost Lag": forecast_xgboost,
    "Random Forest": forecast_random_forest,
    "Linear Regression": forecast_linear_regression,
}


def run_model(model_name: str, series: pd.Series, horizon_weeks: int) -> np.ndarray:
    fn = MODEL_DISPATCH.get(model_name, forecast_arima)
    return fn(series, horizon_weeks)


def compute_series_statistics(series: pd.Series) -> dict:
    """Descriptive stats used for model-selection reasoning (heuristic and/or LLM)."""
    values = series.values.astype(float)
    mean = float(np.mean(values)) if len(values) else 0.0
    std = float(np.std(values)) if len(values) else 0.0
    cv = std / mean if mean > 0 else 0.0

    t = np.arange(len(values))
    if len(values) >= 2:
        slope = float(np.polyfit(t, values, 1)[0])
    else:
        slope = 0.0

    seasonality_strength = 0.0
    if len(values) >= 104:  # need >= 2 years of weekly data to assess yearly seasonality
        try:
            s1 = pd.Series(values[:-52])
            s2 = pd.Series(values[52:])
            seasonality_strength = float(np.corrcoef(s1, s2)[0, 1])
            if np.isnan(seasonality_strength):
                seasonality_strength = 0.0
        except Exception:
            seasonality_strength = 0.0

    return {
        "mean": round(mean, 2),
        "std": round(std, 2),
        "coefficient_of_variation": round(cv, 3),
        "trend_slope": round(slope, 4),
        "seasonality_strength": round(seasonality_strength, 3),
        "n_weeks": len(values),
    }


def select_model_heuristic(stats: dict) -> str:
    """Deterministic fallback model-selection rule, used when the LLM is
    unavailable or its answer can't be parsed. Mirrors the reasoning an LLM
    would be prompted to apply from the descriptive statistics."""
    if stats["n_weeks"] < 20:
        return "Linear Regression"  # too little history for anything fancier
    if stats["seasonality_strength"] > 0.5 and abs(stats["trend_slope"]) > 0.01:
        return "Holt-Winters"
    if stats["seasonality_strength"] > 0.5:
        return "SARIMA"
    if stats["coefficient_of_variation"] > 0.6:
        return "XGBoost Lag"
    if abs(stats["trend_slope"]) > 0.05:
        return "Linear Regression"
    return "ARIMA"
