"""
ThreadWeave AI - Entry point

Usage:
    python main.py --query "Find the total procurement cost for all products"
    python main.py --query "What is the risk profile for Denim Jacket?" --refresh
    python main.py --agent cost          # run one agent directly, skipping the Supervisor's NL parsing
    python main.py --agent all --refresh # force a full pipeline re-run
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents import supervisor_agent, trend_agent, demand_agent, risk_agent, cost_agent

DIRECT_AGENT_RUNNERS = {
    "trend": trend_agent.run,
    "demand": demand_agent.run,
    "risk": risk_agent.run,
    "cost": cost_agent.run,
}


def main():
    parser = argparse.ArgumentParser(description="ThreadWeave AI - Fashion Retail Supply Chain Intelligence")
    parser.add_argument("--query", type=str, default=None,
                         help="Natural language query for the Supervisor Agent")
    parser.add_argument("--agent", type=str, choices=["trend", "demand", "risk", "cost", "all"],
                         default=None, help="Run a specific agent directly (bypasses NL parsing)")
    parser.add_argument("--refresh", action="store_true",
                         help="Force re-run of agents even if output CSVs already exist")
    parser.add_argument("--no-llm-intent", action="store_true",
                         help="Use keyword heuristic instead of LLM for intent parsing")
    args = parser.parse_args()

    if args.agent:
        if args.agent == "all":
            for name in ["trend", "demand", "risk", "cost"]:
                DIRECT_AGENT_RUNNERS[name]()
        else:
            DIRECT_AGENT_RUNNERS[args.agent]()
        return

    query = args.query or "Find the total procurement cost for all products"
    supervisor_agent.run_query(query, refresh=args.refresh, use_llm_intent=not args.no_llm_intent)


if __name__ == "__main__":
    main()
