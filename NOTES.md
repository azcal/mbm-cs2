# CS2 rebuild v1 (bo3.gg, tier S/A)

Data pulled 2026-07-23: 6,531 finished tier-S/A matches (2020->now), 4,917 maps with
player stats (2024-07->now), 327 team names. All from api.bo3.gg (no key needed).

Model: map-count Elo (K=24, updates scaled by sqrt(maps)), two heads fit on <=2025 Bo3:
- winner head: logistic on Elo gap (intercept absorbs the team1 listing/seeding edge, base ~0.552)
- total head:  logistic on |gap| for P(series goes 3 maps), base 3-map rate 0.428

Frozen 2026 Bo3 holdout (n=623):
- winner: LL 0.6255 / Brier 0.2180 / hit 64.4%  (base 0.6853, raw Elo 0.6264)
- O/U 2.5: LL 0.6833 (base 0.6843, iid-from-Elo 0.6867). Thin. Paper only.

Lineup layer (player-strength gap + roster continuity from 49k player-map rows) was
tested ONCE on the frozen holdout and failed (winner 0.6284, total 0.6981). Shelved.
Lineups change in only 7.3% of team-matches; with a one-season training window the
features add noise. Revisit with 3+ years of player history or as live-day flags only.

Files: matches.csv, games.csv, teams.csv, pstats.jsonl.gz, eval_2026.csv, spec.json,
elo_current.json (ratings as of pull date).
