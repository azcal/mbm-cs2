#!/usr/bin/env python3
"""
Regenerate the CS2 backfill so it reflects the rating the board actually ships.

`logs/predictions.csv` carries 623 rows tagged `v1-backfill`, produced by the map-count
Elo. The board has shipped round-margin (MOV) Elo since 2026-07-26, so the dashboard has
been reporting a rating that no longer runs.

This runs inside the repo on purpose. It could not be done from outside: `matches.csv` and
`teams.csv` are authoritative here, and rebuilding the match list from a fresh bo3.gg pull
reproduced the AUC to within 0.0002 but not the log loss, because a slightly different
match set shifts the Elo trajectory. Rebuilding in place removes that difference entirely.

It touches nothing except `logs/predictions.csv`, and only the rows tagged `*-backfill`.
Live rows are left exactly as logged.

    python cs2_backfill.py            # writes the file
    python cs2_backfill.py --dry-run  # prints what it would do and writes nothing
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import daily_cs2 as D                     # reuse the runner's own map index and Elo


def main(dry):
    m = pd.read_csv(D.MATCHES)
    m["date"] = pd.to_datetime(m.date, format="mixed", utc=True)
    m = m.sort_values("date").reset_index(drop=True)
    g = pd.read_csv(D.GAMES) if os.path.exists(D.GAMES) else pd.DataFrame()
    if not len(g):
        raise SystemExit("data/games.csv missing; the MOV rating needs the per-map rows")

    by_mid = D.build_map_index(m, g)
    elo, gaps = D.compute_elo(m, by_mid)
    mw, base = D.fit_head(m, gaps)

    d = m.assign(gap=gaps)
    d = d[d.bo_type == 3].copy()
    d["y"] = (d.winner_team_id == d.team1_id).astype(int)
    te = d.date.dt.year == D.SPEC.get("holdout_year", 2026)
    if te.sum() < 100:
        raise SystemExit(f"only {int(te.sum())} holdout rows found; check the season split")
    rows = d[te]
    p = mw.predict_proba(rows[["gap"]].values)[:, 1]
    pe = 1 / (1 + 10 ** (-rows.gap.values / 400))
    y = rows.y.values

    print(f"\nregenerated holdout, n={len(y)}")
    print(f"  v2-mov head  LL={log_loss(y, np.clip(p, 1e-6, 1-1e-6)):.4f}  "
          f"AUC={roc_auc_score(y, p):.4f}  hit={np.mean((p > .5) == (y == 1)):.3f}")
    print(f"  raw MOV elo  LL={log_loss(y, np.clip(pe, 1e-6, 1-1e-6)):.4f}")
    print(f"  base rate    LL={log_loss(y, np.full(len(y), base)):.4f}")

    tn = {}
    if os.path.exists(D.TEAMS):
        t = pd.read_csv(D.TEAMS)
        tn = dict(zip(t.team_id.astype(int), t.name.astype(str)))
    miss = sum(1 for i in rows.team1_id if int(i) not in tn)
    if miss:
        print(f"  {miss} team ids not in teams.csv, falling back to the id as the name")

    seed = pd.DataFrame({
        "ts": pd.to_datetime(rows.start_date, format="mixed", utc=True).dt.strftime("%Y-%m-%dT%H:%M"),
        "match_id": rows.id.astype(str),
        "team1": [tn.get(int(i), str(i)) for i in rows.team1_id],
        "team2": [tn.get(int(i), str(i)) for i in rows.team2_id],
        "p_team1": p, "p_elo": pe, "bo": 3,
        "model": "v2-mov-backfill", "y": y.astype(float)})

    path = os.path.join(D.LOGS, "predictions.csv")
    if os.path.exists(path):
        log = pd.read_csv(path, dtype={"match_id": str})
        live = log[~log.model.astype(str).str.contains("backfill", case=False)]
        print(f"  keeping {len(live)} live rows, replacing "
              f"{len(log) - len(live)} backfill rows with {len(seed)}")
        out = pd.concat([seed, live], ignore_index=True)
    else:
        out = seed
    out = out.sort_values("ts").reset_index(drop=True)
    if dry:
        print("\ndry run, nothing written")
        return
    out.to_csv(path, index=False)
    print(f"\nwrote {path}: {len(out)} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args().dry_run)
