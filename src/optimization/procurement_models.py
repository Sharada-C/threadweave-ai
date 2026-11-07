"""
ThreadWeave AI - Procurement optimization models

Implements the three OR techniques named in the report:
  1. Economic Order Quantity (EOQ)
  2. Newsvendor Model (demand-uncertainty aware)
  3. Lagrangian Relaxation (budget-constrained across the product portfolio)
"""
import numpy as np
from scipy.stats import norm


def eoq(annual_demand: float, ordering_cost: float, unit_cost: float,
        holding_cost_pct: float) -> float:
    """Classic Economic Order Quantity. Harris (1913)."""
    holding_cost = max(unit_cost * holding_cost_pct, 0.01)
    annual_demand = max(annual_demand, 0.0)
    q = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
    return float(q)


def newsvendor_quantity(mean_demand: float, std_demand: float,
                         price: float, unit_cost: float,
                         salvage_value_pct: float = 0.3) -> float:
    """
    Newsvendor model: order quantity that maximizes expected profit under
    demand uncertainty for a single selling period.

    Critical ratio = (price - cost) / (price - salvage_value)
    Q* = mean + z(critical_ratio) * std
    """
    underage_cost = max(price - unit_cost, 0.01)          # cost of understocking (lost margin)
    overage_cost = max(unit_cost * (1 - salvage_value_pct), 0.01)  # cost of overstocking
    critical_ratio = underage_cost / (underage_cost + overage_cost)
    critical_ratio = min(max(critical_ratio, 0.01), 0.99)
    z = norm.ppf(critical_ratio)
    q = mean_demand + z * std_demand
    return float(max(q, 0.0))


def lagrangian_relaxation(products: list, total_budget: float,
                           max_iterations: int = 60, tol: float = 1e-3) -> dict:
    """
    Budget-constrained procurement across a portfolio of products using
    Lagrangian Relaxation on the total spend constraint.

    Each product dict must contain:
      - "product_id"
      - "unconstrained_qty": the quantity we'd order with no budget limit
                              (e.g. max(EOQ, Newsvendor) per product)
      - "unit_cost"
      - "priority_weight": relative importance (e.g. derived from margin,
                            trend score) used to allocate scarce budget

    Returns a dict {product_id: adjusted_quantity} that respects the
    portfolio budget as closely as possible via bisection on the Lagrange
    multiplier (lambda) applied to each product's spend.
    """
    if not products:
        return {}

    lam_lo, lam_ho = 0.0, 5.0  # search range for the multiplier
    best_result = None

    for _ in range(max_iterations):
        lam = (lam_lo + lam_ho) / 2.0
        result = {}
        total_spend = 0.0
        for p in products:
            # higher lambda -> quantity shrinks faster for low-priority products
            penalty = 1.0 / (1.0 + lam / max(p["priority_weight"], 0.01))
            qty = p["unconstrained_qty"] * penalty
            qty = max(qty, 0.0)
            result[p["product_id"]] = qty
            total_spend += qty * p["unit_cost"]

        best_result = result
        if abs(total_spend - total_budget) / max(total_budget, 1) < tol:
            break
        if total_spend > total_budget:
            lam_lo = lam
        else:
            lam_ho = lam

    return {pid: int(round(q)) for pid, q in best_result.items()}
