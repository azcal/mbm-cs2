"""
MBM CS2 daily runner (rebuild v2, frozen spec).
- Incrementally refreshes finished tier-S/A matches AND their map rows from bo3.gg
- Elo: round-margin (MOV) updates where map data exists, map-count elsewhere
- Fits the frozen winner head (train <= 2025; rest of 2026 = forward test)
- Auto-discovers upcoming Bo3 matches (48h horizon), boards p(win)
- Grades all logged predictions, prints rolling metrics + tripwire
- Series-length head RETIRED 2026-07-26 (see NOTES); no longer computed or boarded
Usage: python daily_cs2.py
"""
import json, os, datetime as dt
import numpy as np, pandas as pd, requests
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
import tg_delta
H = {"User-Agent": "Mozilla/5.0"}
API = "https://api.bo3.gg/api/v1"
MATCHES = os.path.join(HERE, "data", "matches.csv")
GAMES = os.path.join(HERE, "data", "games.csv")
TEAMS = os.path.join(HERE, "data", "teams.csv")
LOGS = os.path.join(HERE, "logs")
SPEC = json.load(open(os.path.join(HERE, "spec.json")))
K = SPEC["K"]
KR = SPEC.get("Kr", 14.0)          # per-map MOV update rate

MIN_EDGE = 0.05  # required margin: threshold odds pay 5% above model breakeven

def min_odds(p):
    import math
    p = min(max(float(p), 0.01), 0.99)
    return math.ceil((1 + MIN_EDGE) / p * 100) / 100.0

FIELDS = ("id", "slug", "start_date", "team1_id", "team2_id", "team1_score",
          "team2_score", "winner_team_id", "bo_type", "tier", "tournament_id")
GFIELDS = ("id", "match_id", "begin_at", "map_name", "number", "winner_clan_score",
           "loser_clan_score", "winner_clan_name", "loser_clan_name", "status")


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


def refresh_games(m):
    """Pull recent finished map rows. The /games endpoint covers every tier, so only
    about one row in ten belongs to a tier-S/A match; page until we stop seeing new
    relevant ids. Historical backfill is deliberately not attempted: extending round
    data from 2024-07 back to 2020 was measured at +0.0003 log loss, which is noise."""
    if os.path.exists(GAMES):
        g = pd.read_csv(GAMES)
    else:
        g = pd.DataFrame(columns=list(GFIELDS))
    known = set(g.id.astype(int)) if len(g) else set()
    want = set(m.id.astype(int))
    new, stale_pages = [], 0
    for page in range(25):
        try:
            r = requests.get(f"{API}/games", headers=H, timeout=60, params={
                "page[limit]": 100, "page[offset]": page * 100, "sort": "-begin_at"}).json()
            res = r.get("results", [])
        except Exception as e:
            print(f"  games fetch failed on page {page}: {e}")
            break
        if not res:
            break
        added = 0
        for x in res:
            if x.get("status") != "finished":
                continue
            if int(x["id"]) in known or int(x.get("match_id") or -1) not in want:
                continue
            new.append({k: x.get(k) for k in GFIELDS})
            known.add(int(x["id"]))
            added += 1
        stale_pages = stale_pages + 1 if added == 0 else 0
        if stale_pages >= 3:
            break
    if new:
        g = pd.concat([g, pd.DataFrame(new)], ignore_index=True).drop_duplicates("id")
        g.to_csv(GAMES, index=False)
    print(f"map rows: {len(g)} (+{len(new)} new)")
    return g


def build_map_index(m, g):
    """Attach each map to (winner_team_id, loser_team_id).

    The /games rows carry clan NAMES, not team ids. Sweeps (2-0) are unambiguous
    because the map winner must be the series winner, so they are used to vote a
    name -> team_id table, which then resolves the 2-1 series."""
    if not len(g):
        return {}
    g = g.copy()
    for c in ("winner_clan_score", "loser_clan_score"):
        g[c] = pd.to_numeric(g[c], errors="coerce")
    g = g[(g.status == "finished") & g.winner_clan_score.notna() & g.loser_clan_score.notna()]
    g = g[(g.winner_clan_score > g.loser_clan_score) & (g.winner_clan_score >= 13)]
    g["winner_clan_name"] = g.winner_clan_name.astype(str).str.strip()
    g["loser_clan_name"] = g.loser_clan_name.astype(str).str.strip()
    mi = m.set_index("id")
    g = g[g.match_id.isin(mi.index)]

    votes = defaultdict(Counter)
    for r in g.itertuples():
        row = mi.loc[r.match_id]
        if min(float(row.team1_score), float(row.team2_score)) != 0:
            continue
        w = row.winner_team_id
        l = row.team2_id if w == row.team1_id else row.team1_id
        votes[r.winner_clan_name][w] += 1
        votes[r.loser_clan_name][l] += 1
    name2id = {n: c.most_common(1)[0][0] for n, c in votes.items()}

    rows = []
    for r in g.itertuples():
        row = mi.loc[r.match_id]
        w, l = name2id.get(r.winner_clan_name), name2id.get(r.loser_clan_name)
        ids = {row.team1_id, row.team2_id}
        if w in ids and l in ids and w != l:
            pass
        elif w in ids and l not in ids:
            l = (ids - {w}).pop()
        elif l in ids and w not in ids:
            w = (ids - {l}).pop()
        else:
            continue
        rows.append((r.match_id, r.number, w, l, float(r.winner_clan_score), float(r.loser_clan_score)))
    if not rows:
        return {}
    MP = pd.DataFrame(rows, columns=["match_id", "number", "win_id", "lose_id", "ws", "ls"])
    MP = MP.sort_values(["match_id", "number"])
    idx = {k: v for k, v in MP.groupby("match_id")}
    print(f"map index: {len(MP)} maps across {len(idx)} matches "
          f"({len(idx)/max(len(m),1)*100:.1f}% of history)")
    return idx


def compute_elo(m, by_mid):
    """Round-margin Elo where map rows exist, map-count Elo elsewhere.

    MOV form: d = KR * ln(margin + 1) * (1 - expected), applied per map in series
    order. Validated on 2025 (+0.0043, t=1.83) and the 2026 holdout (+0.0036, AUC
    0.6924 -> 0.6973) against the v1 map-count rule."""
    elo = defaultdict(lambda: 1500.0)
    gaps = np.zeros(len(m))
    mov_used = 0
    for i, r in enumerate(m.itertuples()):
        gaps[i] = elo[r.team1_id] - elo[r.team2_id]
        s1, s2 = r.team1_score, r.team2_score
        n = (s1 or 0) + (s2 or 0)
        if not np.isfinite(n) or n <= 0:
            continue
        mp = by_mid.get(r.id)
        if mp is not None and len(mp) == int(n):
            for gm in mp.itertuples():
                w, l = gm.win_id, gm.lose_id
                p = 1 / (1 + 10 ** (-(elo[w] - elo[l]) / 400))
                d = KR * np.log((gm.ws - gm.ls) + 1.0) * (1.0 - p)
                elo[w] += d
                elo[l] -= d
            mov_used += 1
            continue
        p = 1 / (1 + 10 ** (-(elo[r.team1_id] - elo[r.team2_id]) / 400))
        d = K * (s1 - p * n) / max(n ** 0.5, 1)
        elo[r.team1_id] += d
        elo[r.team2_id] -= d
    print(f"elo: {mov_used}/{len(m)} matches updated with round margins, rest map-count")
    return elo, gaps


def fit_head(m, gaps):
    d = m.assign(gap=gaps)
    d = d[d.bo_type == 3].copy()
    d["y"] = (d.winner_team_id == d.team1_id).astype(int)
    tr = d.date.dt.year <= 2025
    mw = LogisticRegression(max_iter=1000).fit(d.loc[tr, ["gap"]].values, d.y[tr])
    return mw, float(d.y[tr].mean())


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


def run():
    m = refresh_matches()
    g = refresh_games(m)
    by_mid = build_map_index(m, g)
    elo, gaps = compute_elo(m, by_mid)
    mw, base_w = fit_head(m, gaps)
    up = fetch_upcoming()
    tn = team_names(set(m.team1_id) | set(m.team2_id) |
                    {x["team1_id"] for x in up} | {x["team2_id"] for x in up})

    alerts = []
    rows = []
    os.makedirs(LOGS, exist_ok=True)
    pred_path = os.path.join(LOGS, "predictions.csv")
    log = pd.read_csv(pred_path, dtype={"match_id": str}) if os.path.exists(pred_path) else \
        pd.DataFrame(columns=["ts", "match_id", "team1", "team2", "p_team1", "p_elo", "model"])

    if up:
        print("\n=== MBM CS2 board (Bo3, next 48h) ===")
        for x in up:
            gp = elo[x["team1_id"]] - elo[x["team2_id"]]
            pw = mw.predict_proba([[gp]])[0, 1]
            pe = 1 / (1 + 10 ** (-gp / 400))
            n1 = tn.get(int(x["team1_id"]), x["team1_id"])
            n2 = tn.get(int(x["team2_id"]), x["team2_id"])
            rows.append({"ts": x["start_date"][:16], "match_id": str(x["id"]), "team1": n1,
                         "team2": n2, "p_team1": pw, "p_elo": pe, "model": "v2-mov"})
            print(f"{x['start_date'][:16]}  {n1} vs {n2}  p({n1})={pw:.3f}  (elo {pe:.3f})  "
                  f"value: {n1} >= {min_odds(pw):.2f} / {n2} >= {min_odds(1-pw):.2f}")
        b = pd.DataFrame(rows)
        log = pd.concat([log[~log.match_id.isin(b.match_id)], b], ignore_index=True)
    else:
        print("no upcoming Bo3 tier-S/A matches in horizon")

    fin = m.assign(match_id=m.id.astype(str)).set_index("match_id")
    log["y"] = log.match_id.map((fin.winner_team_id == fin.team1_id).astype(float))
    log.loc[~log.match_id.isin(fin.index), "y"] = np.nan
    log.to_csv(pred_path, index=False)

    dec = log[log.y.notna()]
    if len(dec) >= 30:
        yw = dec.y.astype(int).values
        print(f"\nrolling grade (n={len(dec)}):")
        for lbl, p in (("winner head", dec.p_team1), ("elo floor  ", dec.p_elo)):
            pc = np.clip(p.values.astype(float), 1e-6, 1 - 1e-6)
            print(f"{lbl}: LL={log_loss(yw, pc):.4f}  Brier={brier_score_loss(yw, pc):.4f}  "
                  f"hit={np.mean((pc > .5) == (yw == 1)):.3f}  "
                  f"(base LL={log_loss(yw, np.full(len(yw), base_w)):.4f})")
        w = dec.tail(200)
        if len(w) >= 200:
            yy = w.y.astype(int).values
            pp = np.clip(w.p_team1.values.astype(float), 1e-6, 1 - 1e-6)
            edge = log_loss(yy, np.full(len(yy), base_w)) - log_loss(yy, pp)
            print(f"last-200 winner edge vs base: {edge:+.4f}  "
                  f"[{'OK' if edge > 0 else 'TRIPWIRE: bench review'}]")
            if edge <= 0:
                alerts.append("CS2 winner")
        dec.to_csv(os.path.join(LOGS, "graded.csv"), index=False)

    print("\nseries-length head: RETIRED 2026-07-26. Final 2026 holdout read was "
          "+0.0009 log loss over base rate (t=0.27, p=0.39, AUC 0.5241) on n=623, "
          "with a prediction range topping out below its own base rate.")

    tg, ch = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT")
    if tg and ch:
        push = {r["match_id"]: {"line": (f"{r['team1']} vs {r['team2']}: {r['p_team1']:.3f} | "
                                         f"{r['team1']} >= {min_odds(r['p_team1']):.2f} / "
                                         f"{r['team2']} >= {min_odds(1-r['p_team1']):.2f}"),
                                "min_odds": min_odds(r["p_team1"])} for r in rows} if up else {}
        mode, lines = tg_delta.gate(LOGS, dt.date.today().strftime("%Y%m%d"), push)
        if lines or alerts:
            txt = "\n".join(lines) if lines else "no price changes since the last push"
            if mode == "delta" and lines:
                txt = "changes since the last push:\n" + txt
            if alerts:
                txt += "\n\u26a0 TRIPWIRE: " + ", ".join(alerts) + " below base rate, stop betting this board"
            requests.post(f"https://api.telegram.org/bot{tg}/sendMessage",
                          data={"chat_id": ch, "text": f"MBM CS2\n{txt}"}, timeout=20)


if __name__ == "__main__":
    run()
