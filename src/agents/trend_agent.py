"""
ThreadWeave AI - Trend Agent

Tracks product trends by combining internal sales momentum with (optional)
external signals from Google Trends (pytrends) and web search (SerpAPI).
Produces a Trend Score (0-100), a direction, and an LLM-generated insight
for every product. Falls back to a default score of 50 if online signals
or the LLM are unavailable, per the report's robustness design.
"""
import os
import numpy as np
import pandas as pd

from .. import config, data_loader, llm_utils

DEFAULT_TREND_SCORE = 50


def _sales_momentum_score(sales_df: pd.DataFrame, product_id: str) -> float:
    """Compares last-30-day sales to the prior-30-day sales for a product
    (summed across all stores) and maps the % change onto a 0-100 score
    centered at 50."""
    prod_sales = sales_df[sales_df["Product_ID"] == product_id].sort_values("Date")
    if prod_sales.empty:
        return DEFAULT_TREND_SCORE

    max_date = prod_sales["Date"].max()
    last_30_start = max_date - pd.Timedelta(days=30)
    prior_30_start = max_date - pd.Timedelta(days=60)

    last_30 = prod_sales[prod_sales["Date"] > last_30_start]["Units_Sold"].sum()
    prior_30 = prod_sales[(prod_sales["Date"] > prior_30_start) &
                           (prod_sales["Date"] <= last_30_start)]["Units_Sold"].sum()

    if prior_30 <= 0:
        pct_change = 0.0
    else:
        pct_change = (last_30 - prior_30) / prior_30

    # Map pct_change in [-1, +1]-ish range onto a 0-100 score centered at 50
    score = 50 + np.clip(pct_change, -1.0, 1.0) * 45
    return float(np.clip(score, 1, 99))


def _external_signal_score(product_name: str) -> float:
    """
    Optional external trend signal via pytrends (Google Trends). Returns
    None if unavailable (no network, library missing, or rate-limited) so
    the caller can fall back to sales-momentum-only scoring.
    """
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl="en-US", tz=330, timeout=(3, 5))
        pytrends.build_payload([product_name], timeframe="today 3-m", geo="IN")
        data = pytrends.interest_over_time()
        if data.empty:
            return None
        recent = data[product_name].tail(4).mean()
        return float(np.clip(recent, 0, 100))
    except Exception:
        return None


def _classify_direction(score: float) -> str:
    if score >= 60:
        return "Increasing"
    if score >= 40:
        return "Stable"
    return "Declining"


def _fallback_insight(product_name: str, direction: str) -> str:
    templates = {
        "Increasing": f"{product_name} is gaining traction based on recent sales momentum.",
        "Stable": f"{product_name} shows consistent demand with no major shift in popularity.",
        "Declining": f"{product_name} is losing momentum; consider clearance or markdown planning.",
    }
    return templates[direction]


def _llm_insight(product_name: str, category: str, score: float, direction: str,
                  used_external_signal: bool) -> str:
    system = ("You are the Trend Intelligence Agent inside ThreadWeave AI, a fashion "
              "retail supply chain platform. Given a product's trend score, write ONE "
              "short, business-friendly sentence (max 25 words) explaining the trend "
              "and what action a merchandiser should consider. No preamble, just the sentence.")
    prompt = (f"Product: {product_name} (Category: {category})\n"
              f"Trend Score: {score}/100\n"
              f"Direction: {direction}\n"
              f"External web signal used: {used_external_signal}\n"
              f"Write the insight sentence now.")
    text, used_llm = llm_utils.safe_llm_call(
        prompt, system=system, fallback=_fallback_insight(product_name, direction)
    )
    return text if used_llm else _fallback_insight(product_name, direction)


def run(product_filter: list = None, use_external_signals: bool = True) -> pd.DataFrame:
    print("[Trend Agent] Starting...")
    sales_df = data_loader.load_sales_history()
    catalog_df = data_loader.load_product_catalog()
    catalog_df = data_loader.filter_products(catalog_df, product_filter)

    rows = []
    for _, prod in catalog_df.iterrows():
        pid, pname, cat = prod["Product_ID"], prod["Product_Name"], prod["Category"]

        momentum_score = _sales_momentum_score(sales_df, pid)

        external_score = None
        if use_external_signals:
            external_score = _external_signal_score(pname)

        if external_score is not None:
            final_score = round(0.6 * momentum_score + 0.4 * external_score, 1)
            used_external = True
        else:
            final_score = round(momentum_score, 1)
            used_external = False

        direction = _classify_direction(final_score)
        insight = _llm_insight(pname, cat, final_score, direction, used_external)

        rows.append({
            "Product_ID": pid,
            "Product_Name": pname,
            "Category": cat,
            "Trend_Score": final_score,
            "Trend_Direction": direction,
            "Used_External_Signal": used_external,
            "Insight": insight,
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(config.TREND_SCORES_CSV, index=False)
    print(f"[Trend Agent] Done -> {config.TREND_SCORES_CSV} ({len(result_df)} products)")
    return result_df


if __name__ == "__main__":
    run()
