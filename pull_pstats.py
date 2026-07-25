import requests, time, json, os, sys, pandas as pd
from concurrent.futures import ThreadPoolExecutor
H={"User-Agent":"Mozilla/5.0"}
BUDGET = int(sys.argv[1]) if len(sys.argv)>1 else 420
G = pd.read_csv("cs2/games.csv")
done = set()
if os.path.exists("cs2/pstats.jsonl"):
    for line in open("cs2/pstats.jsonl"):
        try: done.add(json.loads(line)["gid"])
        except Exception: pass
todo = [int(g) for g in G.id if int(g) not in done]
print("remaining games:", len(todo))
t0 = time.time(); out = open("cs2/pstats.jsonl","a")
def pull(gid):
    try:
        r = requests.get(f"https://api.bo3.gg/api/v1/games/{gid}/players_stats", headers=H, timeout=30)
        return gid, r.json()
    except Exception:
        return gid, None
n=0
with ThreadPoolExecutor(max_workers=12) as ex:
    for gid, js in ex.map(pull, todo):
        if js is not None:
            keep = [{k: p.get(k) for k in ("steam_profile_id","clan_name","kills","death","adr","player_rating","win")} for p in js]
            out.write(json.dumps({"gid": gid, "p": keep})+"\n"); n+=1
        if time.time()-t0 > BUDGET:
            print("budget hit after", n, "games"); break
out.close(); print("chunk pulled:", n, "| elapsed:", round(time.time()-t0,1))
