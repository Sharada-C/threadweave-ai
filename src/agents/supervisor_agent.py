"""
ThreadWeave AI - Supervisor Agent

Parses a natural-language query, determines which agents are needed and in
what order (via recursive dependency resolution over
config.AGENT_DEPENDENCIES), runs them (skipping any whose output CSV already
exists, unless --refresh is set), applies any product filter, and presents
the combined results in a clear tabular format.
"""
import re
import pandas as pd

from .. import config, data_loader, llm_utils
from . import trend_agent, demand_agent, risk_agent, cost_agent

AGENT_RUNNERS = {
    "trend": trend_agent.run,
    "demand": demand_agent.run,
    "risk": risk_agent.run,
    "cost": cost_agent.run,
}

_INTENT_KEYWORDS = {
    "cost": ["cost", "procure", "procurement", "budget", "order quantity", "purchase", "spend"],
    "risk": ["risk", "stockout", "delay", "safety stock", "disrupt"],
    "demand": ["demand", "forecast", "predict", "sales projection", "how much will sell"],
    "trend": ["trend", "popular", "popularity", "viral", "gaining", "losing"],
}


def _resolve_dependencies(target_agents: list) -> list:
    """Recursively expands target_agents into a full ordered execution plan
    that respects config.AGENT_DEPENDENCIES (topological order)."""
    resolved = []

    def visit(agent):
        if agent in resolved:
            return
        for dep in config.AGENT_DEPENDENCIES[agent]:
            visit(dep)
        resolved.append(agent)

    for agent in target_agents:
        visit(agent)
    return resolved


def _parse_intent_heuristic(query: str) -> list:
    query_lower = query.lower()
    matched = set()
    for agent, keywords in _INTENT_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            matched.add(agent)
    if not matched:
        matched.add("cost")  # cost is the "final" agent - a sensible default for vague asks
    return list(matched)


def _parse_intent_llm(query: str) -> list:
    system = ("You are the Supervisor Agent inside ThreadWeave AI. Given a user query about "
              "fashion retail supply chain, decide which of these agents must run: trend, "
              "demand, risk, cost. Reply with ONLY a comma-separated list of the needed agent "
              "names from that set (lowercase), nothing else.")
    fallback_agents = _parse_intent_heuristic(query)
    text, used_llm = llm_utils.safe_llm_call(query, system=system,
                                              fallback=",".join(fallback_agents))
    agents = [a.strip().lower() for a in text.split(",") if a.strip().lower() in config.AGENT_DEPENDENCIES]
    return agents if agents else fallback_agents


def _parse_product_filter(query: str, catalog_df: pd.DataFrame) -> list:
    """Looks for explicit Product_ID mentions (e.g. P012) or product-name
    substrings in the query."""
    ids_found = re.findall(r"\bP\d{3}\b", query.upper())
    names_found = []
    query_lower = query.lower()
    for name in catalog_df["Product_Name"].unique():
        if name.lower() in query_lower:
            names_found.append(name)
    return list(set(ids_found + names_found))


def run_query(query: str, refresh: bool = False, use_llm_intent: bool = True) -> pd.DataFrame:
    print(f"\n[Supervisor] Query: \"{query}\"")
    catalog_df = data_loader.load_product_catalog()
    product_filter = _parse_product_filter(query, catalog_df)
    if product_filter:
        print(f"[Supervisor] Product filter detected: {product_filter}")

    target_agents = _parse_intent_llm(query) if use_llm_intent else _parse_intent_heuristic(query)
    execution_plan = _resolve_dependencies(target_agents)
    print(f"[Supervisor] Target agents: {target_agents} -> execution plan: {execution_plan}")

    for agent in execution_plan:
        if not refresh and data_loader.output_exists(agent):
            print(f"[Supervisor] Skipping '{agent}' agent - output already exists "
                  f"({config.AGENT_OUTPUT_FILES[agent]}). Use refresh=True to force re-run.")
            continue
        print(f"[Supervisor] Running '{agent}' agent...")
        AGENT_RUNNERS[agent](product_filter=product_filter if product_filter else None)

    # Assemble final combined view from whichever outputs exist
    final_agent = execution_plan[-1] if execution_plan else "cost"
    result_df = data_loader.load_output(final_agent)
    if product_filter:
        result_df = data_loader.filter_products(result_df, product_filter)

    print(f"\n[Supervisor] === Results ({final_agent} agent) ===")
    print(result_df.to_string(index=False))
    return result_df


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Find the total procurement cost for all products"
    run_query(q)
