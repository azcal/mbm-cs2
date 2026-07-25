"""
MBM CS2 daily runner (rebuild v1, frozen spec).
- Incrementally refreshes finished tier-S/A matches from bo3.gg
- Recomputes map-count Elo, fits frozen heads (train <= 2025; rest of 2026 = forward test)
- Auto-discovers upcoming Bo3 matches (48h horizon), boards p(win) and p(3 maps)
- Grades all logged predictions, prints rolling metrics + tripwires for both heads
Usage: python daily_cs2.py
"""
import json, os, datetime as dt
import numpy as np, pandas as pd, requests
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
H = {"User-Agent": "Mozilla/5.0"}
API = "https://api.bo3.gg/api/v1"
MATCHES = os.path.join(HERE, "data", "matches.csv")
TEAMS = os.path.join(HERE, "data", "teams.csv")
LOGS = os.path.join(HERE, "logs")
SPEC = json.load(open(os.path.join(HERE, "spec.json")))
K = SPEC["K"]

MIN_EDGE = 0.05  # required margin: threshold odds pay 5% above model breakeven

def min_odds(p):
    import math
    p = min(max(float(p), 0.01), 0.99)
    return math.ceil((1 + MIN_EDGE) / p * 100) / 100.0

FIELDS = ("id", "slug", "start_date", "team1_id", "team2_id", "team1_score",
          "team2_score", "winner_team_id", "bo_type", "tier", "tournament_id")

def refresh_matches():
    m = pd.read_csv(MATCHES)
    known = set(m.id.astype(int))
    new = []
    for page in range(20):
        r = requests.get(f"{API}/matches", headers=H, timeout=60, params={
            "page[limit]": 100, "page[offset]": page * 100, "sort": "-start_date",
            "filter[matches.status][in]": "finished",
            "filter[matches.tier][in]": "s,a"}).json()
        res = r.get("results", [])
        if not res:
            break
        hits = 0
        for x in res:
            if int(x["id"]) in known:
                hits += 1
            elif x.get("team1_id") and x.get("team2_id") and x.get("winner_team_id"):
                new.append({k: x.get(k) for k in FIELDS})
        if hits > 50:
            break
    if new:
        n = pd.DataFrame(new)
        n["date"] = pd.to_datetime(n.start_date)
        m = pd.concat([m, n], ignore_index=True).drop_duplicates("id")
        m["date"] = pd.to_datetime(m.date, format="mixed", utc=True)
        m = m.sort_values("date").reset_index(drop=True)
        m.to_csv(MATCHES, index=False)
    else:
        m["date"] = pd.to_datetime(m.date, format="mixed", utc=True)
    print(f"matches: {len(m)} (+{len(new)} new)")
    return m

def team_names(ids):
    t = pd.read_csv(TEAMS) if os.path.exists(TEAMS) else pd.DataFrame(columns=["team_id", "name"])
    missing = [int(i) for i in ids if int(i) not in set(t.team_id.astype(int))]
    for i in range(0, len(missing), 50):
        b = ",".join(map(str, missing[i:i+50]))
        r = requests.get(f"{API}/teams", headers=H, timeout=60,
                         params={"filter[teams.id][in]": b, "page[limit]": 60}).json()
        t = pd.concat([t, pd.DataFrame([{"team_id": x["id"], "name": x["name"]}
                                        for x in r.get("results", [])])], ignore_index=True)
    if missing:
        t.drop_duplicates("team_id").to_csv(TEAMS, index=False)
    return dict(zip(t.team_id.astype(int), t.name.astype(str)))

def compute_elo(m):
    elo = defaultdict(lambda: 1500.0)
    gaps = np.zeros(len(m))
    for i, r in enumerate(m.itertuples()):
        gaps[i] = elo[r.team1_id] - elo[r.team2_id]
        s1, s2 = r.team1_score, r.team2_score
        n = (s1 or 0) + (s2 or 0)
        if not np.isfinite(n) or n <= 0:
            continue
        p = 1 / (1 + 10 ** (-(elo[r.team1_id] - elo[r.team2_id]) / 400))
        d = K * (s1 - p * n) / max(n ** 0.5, 1)
        elo[r.team1_id] += d
        elo[r.team2_id] -= d
    return elo, gaps

def fit_heads(m, gaps):
    d = m.assign(gap=gaps)
    d = d[d.bo_type == 3].copy()
    d["y"] = (d.winner_team_id == d.team1_id).astype(int)
    d["maps3"] = ((d.team1_score + d.team2_score) == 3).astype(int)
    tr = d.date.dt.year <= 2025
    mw = LogisticRegression(max_iter=1000).fit(d.loc[tr, ["gap"]].values, d.y[tr])
    mt = LogisticRegression(max_iter=1000).fit(np.abs(d.loc[tr, ["gap"]].values), d.maps3[tr])
    return mw, mt, float(d.y[tr].mean()), float(d.maps3[tr].mean())

def fetch_upcoming(hours=48):
    r = requests.get(f"{API}/matches", headers=H, timeout=60, params={
        "page[limit]": 40, "sort": "start_date",
        "filter[matches.status][in]": "upcoming,current",
        "filter[matches.tier][in]": "s,a"}).json()
    out = []
    lim = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=hours)
    for x in r.get("results", []):
        if x.get("bo_type") != 3 or not x.get("team1_id") or not x.get("team2_id"):
            continue
        try:
            ts = pd.Timestamp(x["start_date"])
        except Exception:
            continue
        if ts <= lim:
            out.append(x)
    return out

def run():
    m = refresh_matches()
    elo, gaps = compute_elo(m)
    mw, mt, base_w, base_t = fit_heads(m, gaps)
    up = fetch_upcoming()
    tn = team_names(set(m.team1_id) | set(m.team2_id) |
                    {x["team1_id"] for x in up} | {x["team2_id"] for x in up})

    alerts = []
    os.makedirs(LOGS, exist_ok=True)
    pred_path = os.path.join(LOGS, "predictions.csv")
    log = pd.read_csv(pred_path, dtype={"match_id": str}) if os.path.exists(pred_path) else \
        pd.DataFrame(columns=["ts", "match_id", "team1", "team2", "p_team1", "p_3maps", "p_elo", "model"])

    if up:
        rows = []
        print("\n=== MBM CS2 board (Bo3, next 48h) ===")
        for x in up:
            g = elo[x["team1_id"]] - elo[x["team2_id"]]
            pw = mw.predict_proba([[g]])[0, 1]
            p3 = mt.predict_proba([[abs(g)]])[0, 1]
            pe = 1 / (1 + 10 ** (-g / 400))
            n1, n2 = tn.get(int(x["team1_id"]), x["team1_id"]), tn.get(int(x["team2_id"]), x["team2_id"])
            rows.append({"ts": x["start_date"][:16], "match_id": str(x["id"]), "team1": n1,
                         "team2": n2, "p_team1": pw, "p_3maps": p3, "p_elo": pe, "model": "v1"})
            print(f"{x['start_date'][:16]}  {n1} vs {n2}  p({n1})={pw:.3f}  p(3maps)={p3:.3f}  "
                  f"value: {n1} >= {min_odds(pw):.2f} / {n2} >= {min_odds(1-pw):.2f}  (3maps: no action, benched)")
        b = pd.DataFrame(rows)
        log = pd.concat([log[~log.match_id.isin(b.match_id)], b], ignore_index=True)
    else:
        print("no upcoming Bo3 tier-S/A matches in horizon")

    fin = m.assign(match_id=m.id.astype(str)).set_index("match_id")
    log["y"] = log.match_id.map((fin.winner_team_id == fin.team1_id).astype(float))
    log["maps3"] = log.match_id.map(((fin.team1_score + fin.team2_score) == 3).astype(float))
    log.loc[~log.match_id.isin(fin.index), ["y", "maps3"]] = np.nan
    log.to_csv(pred_path, index=False)
    dec = log[log.y.notna()]
    if len(dec) >= 30:
        yw = dec.y.astype(int).values
        y3 = dec.maps3.astype(int).values
        print(f"\nrolling grade (n={len(dec)}):")
        for lbl, p, y, base in (("winner head", dec.p_team1, yw, base_w),
                                ("elo floor  ", dec.p_elo, yw, base_w),
                                ("total head ", dec.p_3maps, y3, base_t)):
            pc = np.clip(p.values.astype(float), 1e-6, 1 - 1e-6)
            print(f"{lbl}: LL={log_loss(y, pc):.4f}  Brier={brier_score_loss(y, pc):.4f}  "
                  f"hit={np.mean((pc > .5) == (y == 1)):.3f}  (base LL={log_loss(y, np.full(len(y), base)):.4f})")
        for lbl, p, y, base in (("winner", dec.p_team1, yw, base_w), ("total", dec.p_3maps, y3, base_t)):
            w = dec.tail(200)
            if len(w) >= 200:
                yy = (w.y if lbl == "winner" else w.maps3).astype(int).values
                pp = np.clip((w.p_team1 if lbl == "winner" else w.p_3maps).values.astype(float), 1e-6, 1 - 1e-6)
                edge = log_loss(yy, np.full(len(yy), base)) - log_loss(yy, pp)
                print(f"last-200 {lbl} edge vs base: {edge:+.4f}  [{'OK' if edge > 0 else 'TRIPWIRE: bench review'}]")
                if lbl == "winner" and edge <= 0:
                    alerts.append("CS2 winner")
        dec.to_csv(os.path.join(LOGS, "graded.csv"), index=False)

    tg, ch = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT")
    if tg and ch and (up or alerts):
        txt = "\n".join(f"{r['team1']} vs {r['team2']}: {r['p_team1']:.3f} | {r['team1']} >= {min_odds(r['p_team1']):.2f} / {r['team2']} >= {min_odds(1-r['p_team1']):.2f}" for r in rows) if up else "no upcoming matches"
        if alerts:
            txt += "\n\u26a0 TRIPWIRE: " + ", ".join(alerts) + " below base rate, stop betting this board"
        requests.post(f"https://api.telegram.org/bot{tg}/sendMessage",
                      data={"chat_id": ch, "text": f"MBM CS2\n{txt}"}, timeout=20)

if __name__ == "__main__":
    run()
