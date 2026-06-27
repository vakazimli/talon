# Talon — Options Trading Assistant

Local-first, self-improving short-term options trading assistant.

## Quick Start

```bash
# Create and activate virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Copy and fill in your API keys
cp .env.example .env

# Initialize the database
python -m scripts.setup_db

# Run (starts in shadow mode by default)
python -m src.main
```

## Architecture

```
SCAN → FILTER → ANALYZE → SCORE → ALERT → TRACK → REVIEW → LEARN → repeat
```

- **Scanner**: pulls market data (yfinance) and options chains (Tradier), computes technicals locally
- **Scoring**: weighted multi-signal scoring with source reliability feedback
- **Synthesis**: LLM generates concise trade cards only for high-scoring setups
- **Delivery**: Telegram bot sends trade cards and accepts feedback commands
- **Learning**: tracks outcomes, updates source reliability scores, runs postmortems

## Modes

- `shadow` — log everything and still send Telegram trade cards (tagged
  `SHADOW`) at a lower score threshold (`min_score_to_alert_shadow`), but
  place no trades. Default.
- `paper` — simulated trades via Alpaca paper trading
- `live` — real alerts at the full score threshold (requires explicit activation)

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/status` | System status, mode, daily P&L, budget |
| `/feedback <id> <good\|bad\|late\|skip>` | Rate an alert |
| `/note <id> <text>` | Add a note to an alert |
| `/review` | Today's performance summary |
| `/weekly` | Weekly postmortem |
| `/sources` | Source reliability rankings |
| `/pause` / `/resume` | Pause/resume scanning |
| `/mode <shadow\|paper\|live>` | Switch operating mode |
| `/budget` | Today's API spend vs budget |
| `/last` | Last alert sent |
| `/watchlist` | Current watchlist |
