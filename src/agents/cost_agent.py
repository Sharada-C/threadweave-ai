"""
ThreadWeave AI - Cost Agent

Three-phase procurement optimization:
  Phase 1: LLM (or fallback constant) sets a portfolio-wide budget multiplier.
  Phase 2: Compute Q_EOQ, Q_Newsvendor, Q_Lagrangian per product.
  Phase 3: Blend the three quantities (LLM-tuned weights, or fixed fallback
           weights) and generate a final business decision - Action,
           Priority, Risk Flag, Business Insight.
"""
import numpy as np
import pandas as pd

from .. import config, data_loader, llm_utils
from ..optimization import procurement_models as opt

FALLBACK_BLEND_WEIGHTS = {"eoq": 0.3, "newsvendor": 0.4, "lagrangian": 0.3}


def _phase1_budget_multiplier(total_unconstrained_cost: float) -> float:
    system = ("You are the Cost Optimization Agent inside ThreadWeave AI. Given the total "
              "unconstrained procurement cost across the portfolio, respond with ONLY a "
              "single decimal number between 0.5 and 1.0 representing the fraction of that "
              "spend the business should commit to this cycle (a budget multiplier). "
              "No words, just the number.")
    prompt = f"Total unconstrained procurement cost: INR {total_unconstrained_cost:,.0f}\nBudget multiplier?"
    text, used_llm = llm_utils.safe_llm_call(prompt, system=system,
                                              fallback=str(config.DEFAULT_BUDGET_MULTIPLIER))
    if used_llm:
        try:
            val = float(text.strip())
            if 0.3 <= val <= 1.0:
                return val
        except ValueError:
            pass
    return config.DEFAULT_BUDGET_MULTIPLIER


def _decide_action_fallback(trend_score: float, composite_risk: float) -> tuple:
    if composite_risk > 7 or trend_score < 35:
        return "HOLD_STOCK", "LOW"
    if trend_score > 65 and composite_risk < 4:
        return "PROCURE_FULL", "HIGH"
    return "PROCURE_PARTIAL", "MEDIUM"


def _risk_flag(composite_risk: float) -> str:
    if composite_risk > 7:
        return "RED"
    if composite_risk > 4:
        return "AMBER"
    return "GREEN"


def _fallback_insight(product_name: str, action: str, trend_score: float, risk_score: float) -> str:
    return (f"{action.replace('_', ' ').title()} recommended for {product_name} based on "
            f"trend score {trend_score} and risk score {risk_score}.")


def _llm_business_decision(product_name: str, trend_score: float, risk_score: float,
                            recommended_qty: int, total_cost: float) -> tuple:
    """Returns (action, priority, insight). Falls back to deterministic rules
    if the LLM is unavailable or gives an unparseable action."""
    fallback_action, fallback_priority = _decide_action_fallback(trend_score, risk_score)
    fallback_insight = _fallback_insight(product_name, fallback_action, trend_score, risk_score)

    system = ("You are the Cost Optimization Agent inside ThreadWeave AI. Based on the trend "
              "score and risk score, choose an Action from exactly: PROCURE_FULL, "
              "PROCURE_PARTIAL, or HOLD_STOCK. Then choose a Priority from exactly: HIGH, "
              "MEDIUM, or LOW. Then write ONE short business insight sentence (max 25 words). "
              "Respond in this exact format on three lines:\nACTION: <action>\nPRIORITY: <priority>\nINSIGHT: <sentence>")
    prompt = (f"Product: {product_name}\n"
              f"Trend score: {trend_score}/100\n"
              f"Composite risk score: {risk_score}/10\n"
              f"Recommended order quantity: {recommended_qty}\n"
              f"Total cost: INR {total_cost:,.0f}")
    text, used_llm = llm_utils.safe_llm_call(prompt, system=system, fallback="")

    if used_llm and text:
        try:
            lines = {line.split(":", 1)[0].strip().upper(): line.split(":", 1)[1].strip()
                      for line in text.splitlines() if ":" in line}
            action = lines.get("ACTION", "").upper()
            priority = lines.get("PRIORITY", "").upper()
            insight = lines.get("INSIGHT", "")
            if action in ("PROCURE_FULL", "PROCURE_PARTIAL", "HOLD_STOCK") and \
               priority in ("HIGH", "MEDIUM", "LOW") and insight:
                return action, priority, insight
        except Exception:
            pass

    return fallback_action, fallback_priority, fallback_insight


def run(product_filter: list = None) -> pd.DataFrame:
    print("[Cost Agent] Starting...")
    supplier_df = data_loader.filter_products(data_loader.load_supplier_data(), product_filter)
    catalog_df = data_loader.load_product_catalog()

    if not (data_loader.output_exists("demand") and data_loader.output_exists("risk")):
        raise RuntimeError("cost_agent requires demand_forecast.csv and risk_analysis.csv - "
                            "run the Demand and Risk Agents first.")
    demand_df = data_loader.load_output("demand")
    risk_df = data_loader.load_output("risk")
    trend_df = data_loader.load_output("trend") if data_loader.output_exists("trend") else None

    # ---- Phase 2: compute unconstrained EOQ / Newsvendor per product ----
    phase2_rows = []
    for _, sup in supplier_df.iterrows():
        pid, pname = sup["Product_ID"], sup["Product_Name"]
        unit_cost = float(sup["Unit_Cost_INR"])

        demand_row = demand_df.loc[demand_df["Product_ID"] == pid]
        if demand_row.empty:
            continue
        forecast_30d = float(demand_row["Final_Forecast_30d"].values[0])
        cv = float(demand_row["Coefficient_of_Variation"].values[0]) if \
            "Coefficient_of_Variation" in demand_row.columns else 0.3
        annual_demand = forecast_30d * 12
        daily_demand = forecast_30d / 30.0
        demand_std = daily_demand * max(cv, 0.1) * 30  # monthly std approx

        catalog_row = catalog_df.loc[catalog_df["Product_ID"] == pid]
        price = float(catalog_row["Base_Price_INR"].values[0]) if not catalog_row.empty else unit_cost * 2

        q_eoq = opt.eoq(annual_demand, config.ORDERING_COST_INR, unit_cost, config.HOLDING_COST_PCT)
        q_news = opt.newsvendor_quantity(forecast_30d, demand_std, price, unit_cost,
                                          salvage_value_pct=1 - config.OVERAGE_COST_PCT)

        trend_score = 50.0
        if trend_df is not None and pid in trend_df["Product_ID"].values:
            trend_score = float(trend_df.loc[trend_df["Product_ID"] == pid, "Trend_Score"].values[0])
        priority_weight = max(trend_score, 5.0)  # higher trend -> gets more budget priority

        phase2_rows.append({
            "product_id": pid, "product_name": pname, "unit_cost": unit_cost,
            "q_eoq": q_eoq, "q_news": q_news,
            "unconstrained_qty": max(q_eoq, q_news),
            "priority_weight": priority_weight,
            "price": price, "trend_score": trend_score,
        })

    total_unconstrained_cost = sum(r["unconstrained_qty"] * r["unit_cost"] for r in phase2_rows)

    # ---- Phase 1: LLM (or fallback) sets portfolio budget multiplier ----
    budget_multiplier = _phase1_budget_multiplier(total_unconstrained_cost)
    total_budget = total_unconstrained_cost * budget_multiplier
    print(f"  Budget multiplier: {budget_multiplier} -> total budget INR {total_budget:,.0f} "
          f"(unconstrained was INR {total_unconstrained_cost:,.0f})")

    lagrangian_qtys = opt.lagrangian_relaxation(phase2_rows, total_budget)

    # ---- Phase 3: blend quantities + LLM business decision ----
    rows = []
    for r in phase2_rows:
        pid, pname = r["product_id"], r["product_name"]
        q_lag = lagrangian_qtys.get(pid, r["unconstrained_qty"])
        blended_qty = int(round(
            FALLBACK_BLEND_WEIGHTS["eoq"] * r["q_eoq"] +
            FALLBACK_BLEND_WEIGHTS["newsvendor"] * r["q_news"] +
            FALLBACK_BLEND_WEIGHTS["lagrangian"] * q_lag
        ))
        total_cost = round(blended_qty * r["unit_cost"], 2)

        risk_row = risk_df.loc[risk_df["Product_ID"] == pid]
        composite_risk = float(risk_row["Composite_Risk_Score"].values[0]) if not risk_row.empty else 5.0

        action, priority, insight = _llm_business_decision(
            pname, r["trend_score"], composite_risk, blended_qty, total_cost)
        risk_flag = _risk_flag(composite_risk)

        rows.append({
            "Product_ID": pid,
            "Product_Name": pname,
            "Q_EOQ": int(round(r["q_eoq"])),
            "Q_Newsvendor": int(round(r["q_news"])),
            "Q_Lagrangian": int(round(q_lag)),
            "Recommended_Qty": blended_qty,
            "Total_Cost_INR": total_cost,
            "Action": action,
            "Priority": priority,
            "Risk_Flag": risk_flag,
            "Business_Insight": insight,
        })
        print(f"  - {pid} {pname}: qty={blended_qty}, action={action}, priority={priority}")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(config.COST_OPTIMIZATION_CSV, index=False)
    print(f"[Cost Agent] Done -> {config.COST_OPTIMIZATION_CSV} ({len(result_df)} products)")
    return result_df


if __name__ == "__main__":
    run()
