# MBM CS2 daily engine (rebuild v1)

Standalone predictive model for tier-S/A Bo3 matches. No market inputs.
Map-count Elo (K=24) + two frozen heads trained on data through 2025:
winner = P(team1 wins series), total = P(series goes 3 maps) for the O/U 2.5 market.
Rest of 2026 is the forward test. Board covers Bo3 only (heads are Bo3-calibrated).

The total head barely clears its base rate on the 2026 holdout. It is logged and
tripwired, and should not be bet. Winner head is the product.

## Setup (one time, ~5 min)
1. Private GitHub repo, upload this folder (keep .github).
2. Enable Actions. Runs twice daily (EU + NA slates), auto-discovers upcoming matches
   from bo3.gg, grades everything, commits state.
3. Optional Telegram: secrets TG_TOKEN and TG_CHAT.

## Manual run
    pip install -r requirements.txt && python daily_cs2.py

## Tripwires
Every run prints last-200 log-loss edge vs base rate for BOTH heads and flags
TRIPWIRE the moment either goes non-positive.
