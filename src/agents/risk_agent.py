"""
ThreadWeave AI - Risk Agent

Computes a stochastic risk profile per product: stockout probability
(normal distribution over lead-time demand), safety stock at a 95% service
level, Value-at-Risk (95th percentile demand surplus), supplier delay risk,
and a weighted composite risk score (0-10). Generates an LLM mitigation
strategy with a templated fallback.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from .. import config, data_loader, llm_utils


def _fallback_mitigation(risk_score: float, stockout_prob: float) -> str:
    if risk_score > 7:
        return "High risk: increase safety stock, diversify suppliers, and expedite reorder."
    if risk_score > 4:
        return "Moderate risk: monitor supplier lead times closely and keep buffer stock topped up."
    return "Low risk: maintain current reorder policy and standard monitoring cadence."


def _llm_mitigation(product_name: str, stockout_prob: float, delay_risk: float,
                     composite_risk: float) -> str:
    system = ("You are the Risk Assessment Agent inside ThreadWeave AI. Write ONE short, "
              "actionable mitigation sentence (max 30 words) for a merchandiser. No preamble.")
    prompt = (f"Product: {product_name}\n"
              f"Stockout probability: {stockout_prob}%\n"
              f"Supplier delay risk: {delay_risk}%\n"
              f"Composite risk score (0-10): {composite_risk}\n"
              "What should the business do?")
    fallback = _fallback_mitigation(composite_risk, stockout_prob)
    text, used_llm = llm_utils.safe_llm_call(prompt, system=system, fallback=fallback)
    return text if used_llm else fallback


def run(product_filter: list = None) -> pd.DataFrame:
    print("[Risk Agent] Starting...")
    supplier_df = data_loader.filter_products(data_loader.load_supplier_data(), product_filter)
    inventory_df = data_loader.load_inventory()
    demand_df = data_loader.load_output("demand") if data_loader.output_exists("demand") else None

    if demand_df is None:
        raise RuntimeError("risk_agent requires demand_forecast.csv - run the Demand Agent first.")

    rows = []
    for _, sup in supplier_df.iterrows():
        pid, pname = sup["Product_ID"], sup["Product_Name"]
        lead_time = float(sup["Lead_Time_Days"])
        reliability = float(sup["Reliability_Score"])

        demand_row = demand_df.loc[demand_df["Product_ID"] == pid]
        if demand_row.empty:
            continue
        forecast_30d = float(demand_row["Final_Forecast_30d"].values[0])
        daily_demand = forecast_30d / 30.0
        cv = float(demand_row["Coefficient_of_Variation"].values[0]) if \
            "Coefficient_of_Variation" in demand_row.columns else 0.3
        demand_std = daily_demand * max(cv, 0.1)

        # Lead-time demand distribution
        lead_time_demand_mean = daily_demand * lead_time
        lead_time_demand_std = demand_std * np.sqrt(max(lead_time, 1))

        # Available stock for this product, summed across stores
        prod_inventory = inventory_df[inventory_df["Product_ID"] == pid]
        available_stock = float(prod_inventory["Usable_Stock"].sum()) if not prod_inventory.empty else 0.0

        # Stockout probability: P(lead-time demand > available stock)
        if lead_time_demand_std > 0:
            z_score = (available_stock - lead_time_demand_mean) / lead_time_demand_std
            stockout_prob = round(float((1 - norm.cdf(z_score)) * 100), 1)
        else:
            stockout_prob = 0.0 if available_stock >= lead_time_demand_mean else 100.0

        # Safety stock at 95% service level
        safety_stock_95 = round(config.SERVICE_LEVEL_Z * lead_time_demand_std, 1)

        # Value-at-Risk: 95th percentile of demand surplus over lead time
        var_95 = round(float(lead_time_demand_mean + config.SERVICE_LEVEL_Z * lead_time_demand_std), 1)

        # Supplier delay risk from reliability score (0-100 -> higher unreliability = higher risk)
        supplier_delay_risk = round(max(0.0, 100.0 - reliability), 1)

        # Composite risk score (0-10), weighted blend
        composite_risk = round(np.clip(
            0.40 * (stockout_prob / 10) +
            0.35 * (supplier_delay_risk / 10) +
            0.25 * (var_95 / (daily_demand + 1) / 3),
            0, 10), 2)

        mitigation = _llm_mitigation(pname, stockout_prob, supplier_delay_risk, composite_risk)

        rows.append({
            "Product_ID": pid,
            "Product_Name": pname,
            "Available_Stock": available_stock,
            "Lead_Time_Days": lead_time,
            "Stockout_Probability_%": stockout_prob,
            "Safety_Stock_95pct": safety_stock_95,
            "Value_at_Risk_95": var_95,
            "Supplier_Delay_Risk_%": supplier_delay_risk,
            "Composite_Risk_Score": composite_risk,
            "Mitigation_Strategy": mitigation,
        })
        print(f"  - {pid} {pname}: stockout_prob={stockout_prob}%, composite_risk={composite_risk}")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(config.RISK_ANALYSIS_CSV, index=False)
    print(f"[Risk Agent] Done -> {config.RISK_ANALYSIS_CSV} ({len(result_df)} products)")
    return result_df


if __name__ == "__main__":
    run()
