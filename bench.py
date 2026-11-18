"""Benchmark RamseyZero on known Ramsey lower bounds.

  python bench.py                 # exact values and records, 3 seeds, 120 s cap
  python bench.py --frontier      # also attempt harder records (minutes each)
  python bench.py --cases 5,5:42 6,6:101 --seeds 5 --cap 300 --jobs 4

Writes results/benchmark.md and results/benchmark.json; every success is re-verified
by the independent pure-Python verifier inside solve().
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ramsey_zero import solve  # noqa: E402
from ramsey_zero import known  # noqa: E402

CASES = [  # (ks, n): a colouring of K_n certifies R(ks) >= n+1
    ((3, 3), 5), ((3, 4), 8), ((3, 5), 13), ((3, 6), 17), ((3, 7), 22), ((3, 8), 27), ((3, 9), 35),
    ((4, 4), 17), ((4, 5), 24), ((3, 3, 3), 16), ((3, 3, 4), 29),
    ((5, 5), 42), ((4, 7), 48), ((6, 6), 101), ((3, 3, 3, 3), 50), ((4, 4, 4), 127),
]
FRONTIER = [((3, 10), 39), ((4, 6), 35), ((4, 8), 58), ((5, 6), 58)]


def note(ks, n):
    lo, up = known.lower_bound(ks), known.upper_bound_known(ks)
    name = "R(" + ",".join(map(str, ks)) + ")"
    if lo is None:
        return ""
    if lo == up:
        return f"exact: {name} = {lo}"
    tag = "record" if n == lo - 1 else "known size"
    return f"{tag}: {lo} <= {name}" + (f" <= {up}" if up else "")


def parse_cases(items):
    out = []
    for it in items:
        ks, n = it.split(":")
        out.append((tuple(int(x) for x in ks.split(",")), int(n)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cap", type=float, default=120.0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--frontier", action="store_true")
    ap.add_argument("--frontier-cap", type=float, default=600.0)
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    cases = [(c, a.cap, a.seeds) for c in (parse_cases(a.cases) if a.cases else CASES)]
    if a.frontier:
        cases += [(c, a.frontier_cap, 1) for c in FRONTIER]
    rows = []
    for (ks, n), cap, seeds in cases:
        times, routes, ok = [], [], 0
        for s in range(seeds):
            res = solve(list(ks), n, time=cap, seed=s, verbose=False, jobs=a.jobs)
            if res.solved:
                ok += 1
                times.append(res.seconds)
                routes.append(res.route.replace(" (complete search)", ""))
        row = {"ks": ks, "n": n, "note": note(ks, n), "solved": f"{ok}/{seeds}", "cap_s": cap,
               "median_s": round(statistics.median(times), 2) if times else None,
               "max_s": round(max(times), 2) if times else None, "routes": sorted(set(routes))}
        rows.append(row)
        print(json.dumps(row), flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "benchmark.json"), "w") as f:
        json.dump(rows, f, indent=1)
    with open(os.path.join(a.out, "benchmark.md"), "w") as f:
        f.write("| colouring of | certifies | solved | median s | max s | winning routes | literature |\n"
                "|---|---|---|---|---|---|---|\n")
        for r in rows:
            ks = ",".join(map(str, r["ks"]))
            f.write(f"| K_{r['n']}, ({ks}) | R({ks}) >= {r['n'] + 1} | {r['solved']} | {r['median_s']} | "
                    f"{r['max_s']} | {', '.join(r['routes'])} | {r['note']} |\n")
        f.write(f"\nGenerated {time.strftime('%Y-%m-%d %H:%M')} on {os.cpu_count()} CPU(s), "
                f"{a.jobs} worker(s) per run.\n")


if __name__ == "__main__":
    main()
