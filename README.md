# ThreadWeave AI — Autonomous Multi-Agent Platform for Fashion Retail Supply Chain Optimization

Working implementation matching the architecture in the IIM Mumbai summer internship report:
a Supervisor Agent orchestrating four specialist agents (Trend, Demand, Risk, Cost), each
combining a quantitative Python model with local LLM reasoning via Ollama.

## Project layout

```
threadweave_project/
├── data/                        # input CSVs (already populated with synthetic data)
│   ├── fashion_sales_history.csv
│   ├── supplier_data.csv
│   ├── inventory.csv
│   └── product_catalog.csv
├── outputs/                     # agent-generated CSVs land here
├── src/
│   ├── config.py                # paths, model name, dependency graph, tunables
│   ├── llm_utils.py              # Ollama wrapper with connectivity check + fallback
│   ├── data_loader.py
│   ├── models/
│   │   └── forecasting_models.py # ARIMA, SARIMA, Holt-Winters, Prophet, XGBoost, RF, LinReg
│   ├── optimization/
│   │   └── procurement_models.py # EOQ, Newsvendor, Lagrangian Relaxation
│   └── agents/
│       ├── trend_agent.py
│       ├── demand_agent.py
│       ├── risk_agent.py
│       ├── cost_agent.py
│       └── supervisor_agent.py
├── main.py                      # CLI entry point
└── requirements.txt
```

## Setup

```bash
# 1. Install Ollama (https://ollama.com) and pull the model
ollama pull qwen2.5-coder:7b
ollama serve                     # leave running in a terminal/background

# 2. Python environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# (Optional) Prophet - uncomment in requirements.txt, or install separately.
# The pipeline runs fine without it; the Demand Agent just won't pick "Prophet".

# (Optional) SerpAPI + pytrends already in requirements.txt for the Trend Agent's
# external signal. Without a key, the Trend Agent uses sales-momentum only.
```

## Running it

```bash
# Natural-language query through the Supervisor Agent (recommended)
python main.py --query "Find the total procurement cost for all products"
python main.py --query "What is the risk profile for Denim Jacket?"
python main.py --query "Show demand forecast for Silk Saree"

# Force a full re-run even if outputs/*.csv already exist
python main.py --query "procurement cost for all products" --refresh

# Run a single agent directly (bypasses NL intent parsing)
python main.py --agent trend
python main.py --agent demand
python main.py --agent risk
python main.py --agent cost

# Run the entire pipeline once, in order
python main.py --agent all --refresh
```

## How it behaves with vs. without Ollama running

Every agent follows the report's "robustness" principle: every LLM call has a Python
fallback, so the pipeline **never fails outright** because of the LLM.

| | Ollama running | Ollama NOT running |
|---|---|---|
| Trend Agent | LLM writes the insight sentence | Templated insight based on direction |
| Demand Agent | LLM picks the forecasting model + explains it | Deterministic heuristic on descriptive stats (seasonality, CV, trend slope) |
| Risk Agent | LLM writes the mitigation strategy | Templated mitigation based on risk tier |
| Cost Agent | LLM sets the budget multiplier + the Action/Priority/Insight | Fixed 0.85 multiplier + rule-based Action/Priority + templated insight |

`llm_utils.is_llm_available()` does a fast connectivity check against
`http://localhost:11434` before ever touching LangChain, so a missing Ollama server
degrades gracefully instead of hanging or crashing.

This has been tested end-to-end **without** Ollama running (fallback paths only) across
all 50 products in the dataset in ~18 seconds — confirming the pipeline, dependency
resolution, and output schemas are all correct. With Ollama running, the same commands
will additionally produce natural-language reasoning per the report's design.

## Supervisor Agent behavior

- Parses the query for **intent keywords** (or asks the LLM to classify it) to determine
  which of `trend`, `demand`, `risk`, `cost` are needed.
- Expands that into a full **execution plan** using `config.AGENT_DEPENDENCIES`
  (`cost` depends on `risk` depends on `demand` depends on `trend`), so e.g. asking only
  about "procurement cost" still runs the full chain in the correct order.
- **Skips** any agent whose output CSV already exists on disk, unless `--refresh` is passed.
- Detects an explicit **product filter** from Product_IDs (e.g. `P016`) or product names
  mentioned in the query, and narrows all agents to just those products.

## Notes on the synthetic data

The bundled dataset (50 products × 5 stores × 5 years daily) has fairly uniform strong
yearly seasonality baked into every category, so the Demand Agent's heuristic model
selector currently picks **Holt-Winters** for most products — that's an honest reflection
of the heuristic on this particular synthetic data, not a bug. Swap in real (or more
varied synthetic) sales history and you'll see the model mix diversify, matching the
report's stated distribution (ARIMA / XGBoost Lag / etc.).
