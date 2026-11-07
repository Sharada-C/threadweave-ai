"""
ThreadWeave AI - Demand Agent

Aggregates daily sales into weekly series per product, computes descriptive
statistics, selects the most suitable forecasting model (LLM-assisted, with
a deterministic heuristic fallback), runs it, and applies the Trend Agent's
multiplier to produce a final 30-day forecast.
"""
import math
import numpy as np
import pandas as pd

from .. import config, data_loader, llm_utils
from ..models import forecasting_models as fm

WEEKS_FOR_30D = math.ceil(config.FORECAST_HORIZON_DAYS / 7)  # 5 weeks


def _weekly_series(sales_df: pd.DataFrame, product_id: str) -> pd.Series:
    prod_sales = sales_df[sales_df["Product_ID"] == product_id].copy()
    prod_sales = prod_sales.set_index("Date").sort_index()
    weekly = prod_sales["Units_Sold"].resample("W").sum()
    return weekly


def _llm_select_model(product_name: str, stats: dict) -> str:
    heuristic_pick = fm.select_model_heuristic(stats)
    system = ("You are the Demand Forecasting Agent inside ThreadWeave AI. Choose exactly "
              "ONE forecasting model from this list based on the statistics given: "
              f"{', '.join(config.FORECAST_MODELS)}. "
              "Reply with ONLY the exact model name from the list, nothing else.")
    prompt = (f"Product: {product_name}\n"
              f"Mean weekly units: {stats['mean']}\n"
              f"Std dev: {stats['std']}\n"
              f"Coefficient of variation: {stats['coefficient_of_variation']}\n"
              f"Trend slope: {stats['trend_slope']}\n"
              f"Seasonality strength: {stats['seasonality_strength']}\n"
              f"History length (weeks): {stats['n_weeks']}\n"
              f"Which model?")
    text, used_llm = llm_utils.safe_llm_call(prompt, system=system, fallback=heuristic_pick)
    if not used_llm:
        return heuristic_pick
    # validate LLM answer is actually one of our known models
    text_clean = text.strip().strip(".").strip()
    for model_name in config.FORECAST_MODELS:
        if model_name.lower() == text_clean.lower() or model_name.lower() in text_clean.lower():
            return model_name
    return heuristic_pick  # unparseable answer -> fall back


def _fallback_explanation(product_name: str, model_name: str, forecast_30d: int) -> str:
    return (f"{product_name}: selected {model_name} based on historical sales patterns; "
            f"projected {forecast_30d} units over the next 30 days.")


def _llm_explanation(product_name: str, model_name: str, stats: dict, forecast_30d: int,
                      trend_mult: float) -> str:
    system = ("You are the Demand Forecasting Agent inside ThreadWeave AI. Write ONE short "
              "business-friendly sentence (max 30 words) explaining the forecast decision. "
              "No preamble, just the sentence.")
    prompt = (f"Product: {product_name}\n"
              f"Model used: {model_name}\n"
              f"Trend multiplier applied: {trend_mult}\n"
              f"Coefficient of variation: {stats['coefficient_of_variation']}\n"
              f"Final 30-day forecast: {forecast_30d} units\n"
              "Explain this forecast.")
    fallback = _fallback_explanation(product_name, model_name, forecast_30d)
    text, used_llm = llm_utils.safe_llm_call(prompt, system=system, fallback=fallback)
    return text if used_llm else fallback


def run(product_filter: list = None) -> pd.DataFrame:
    print("[Demand Agent] Starting...")
    sales_df = data_loader.load_sales_history()
    catalog_df = data_loader.filter_products(data_loader.load_product_catalog(), product_filter)

    if data_loader.output_exists("trend"):
        trend_df = data_loader.load_output("trend")
    else:
        print("[Demand Agent] WARNING: trend_scores.csv not found - assuming neutral trend (50) "
              "for all products. Run the Trend Agent first for accurate results.")
        trend_df = None

    rows = []
    for _, prod in catalog_df.iterrows():
        pid, pname = prod["Product_ID"], prod["Product_Name"]

        weekly = _weekly_series(sales_df, pid)
        stats = fm.compute_series_statistics(weekly)
        model_name = _llm_select_model(pname, stats)

        weekly_forecast = fm.run_model(model_name, weekly, WEEKS_FOR_30D)
        base_30d = float(np.sum(weekly_forecast) * (config.FORECAST_HORIZON_DAYS /
                                                      (WEEKS_FOR_30D * 7)))

        if trend_df is not None and pid in trend_df["Product_ID"].values:
            trend_score = trend_df.loc[trend_df["Product_ID"] == pid, "Trend_Score"].values[0]
        else:
            trend_score = 50.0
        trend_mult = round(1 + (trend_score - 50) / 100, 3)  # +/- up to ~0.5

        final_forecast = int(max(round(base_30d * trend_mult), 0))
        explanation = _llm_explanation(pname, model_name, stats, final_forecast, trend_mult)

        rows.append({
            "Product_ID": pid,
            "Product_Name": pname,
            "Selected_Model": model_name,
            "Mean_Weekly_Units": stats["mean"],
            "Coefficient_of_Variation": stats["coefficient_of_variation"],
            "Trend_Slope": stats["trend_slope"],
            "Seasonality_Strength": stats["seasonality_strength"],
            "Trend_Multiplier_Applied": trend_mult,
            "Final_Forecast_30d": final_forecast,
            "Explanation": explanation,
        })
        print(f"  - {pid} {pname}: model={model_name}, forecast_30d={final_forecast}")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(config.DEMAND_FORECAST_CSV, index=False)
    print(f"[Demand Agent] Done -> {config.DEMAND_FORECAST_CSV} ({len(result_df)} products)")
    return result_df


if __name__ == "__main__":
    run()
