---
name: talon-analyze
description: On-demand short-term options analysis for a single ticker using Talon's pipeline (technicals, options chain, scoring, optional chart vision) with deterministic, real-quote-based pricing. Research only; not financial advice.
entrypoints:
  quick: python -m skill.analyze {ticker}
  deep: python -m skill.analyze {ticker} deep
---

# Talon Analyze Skill

Runs Talon's full analysis pipeline for ONE ticker on demand and returns a
structured trade idea (or a clear "no setup" reason). It reuses the same
guardrails as the scheduled alerts:

- Prices (entry/target/stop) are derived in code from the chosen option's
  real bid/ask and mid - the model never supplies dollar values.
- The contract must match a real, liquid candidate in the configured DTE
  window or the result is rejected.
- The result reports whether the setup passes the live alert gates
  (score floor + minimum confidence) or is informational only.

## Entry points

- `quick` - `python -m skill.analyze <TICKER>`: scan -> detect -> options ->
  score -> deterministically-priced synthesis.
- `deep` - `python -m skill.analyze <TICKER> deep`: same, plus the vision
  chart-analyst pass (Claude vision) for an extra confirmation signal.

## Output

Human-readable summary followed by a JSON block with: `ticker`, `ok`,
`stage`, and (when a setup exists) `direction`, `setup_type`, `score`,
`gate_pass`, `vision`, and a `card` object (`contract`, `entry_low/high`,
`target`, `stop`, `sell_by`, `confidence`, `rationale`).

## Requirements

- A populated `talon/.env` (Alpaca keys for real-time data, Anthropic key
  for the LLM steps). Without market-data keys it falls back to delayed
  yfinance data.

## Disclaimer

Research and educational tooling only. Talon never executes trades and this
skill does not constitute investment advice.
