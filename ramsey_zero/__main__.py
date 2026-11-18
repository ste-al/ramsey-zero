"""Command line interface.

  python -m ramsey_zero solve 5 5 --n 42 --time 60
  python -m ramsey_zero solve 3 3 3 3 --n 50 --jobs 4          (four colours, four workers)
  python -m ramsey_zero climb 4 6 --from 25 --to 36 --time-per-n 120
  python -m ramsey_zero verify certificates/R5-5_n42.txt
  python -m ramsey_zero compare results/R4-6_n35.txt r46_35some.g6   (is my graph a known one?)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from .solver import Config, solve
from .verify import graph_of, load_coloring, save_coloring, to_graph6, verify


def _name(ks, n):
    return f"R{'-'.join(map(str, ks))}_n{n}"


def _rname(ks):
    return "R(" + ",".join(map(str, ks)) + ")"


def _save(res, ks, n, outdir):
    path = os.path.join(outdir, _name(ks, n) + ".txt")
    meta = {"certifies": f"{_rname(ks)} >= {n + 1}", "route": res.route,
            "seconds": round(res.seconds, 3), "clique_numbers": res.verification["clique_numbers"],
            "found": time.strftime("%Y-%m-%d %H:%M:%S")}
    if not res.verification.get("complete", True):
        meta["verification"] = "full recount in C only; run the verify command for the independent check"
    if res.record:
        meta["note"] = "beats the best published lower bound known to this program"
    if len(ks) == 2:
        meta["graph6"] = to_graph6(graph_of(res.coloring, 0))
    save_coloring(path, res.coloring, ks, meta)
    return path


def _save_best(res, ks, n, outdir):
    if res.best_coloring is None:
        return None
    path = os.path.join(outdir, _name(ks, n) + "_best.txt")
    save_coloring(path, res.best_coloring, ks, {"status": "NOT a valid colouring: best near-solution",
                                                "weighted_violations": res.best_score})
    return path


def _check(a):
    if any(k < 2 for k in a.ks):
        sys.exit("every forbidden clique size must be at least 2")
    if len(a.ks) > 8:
        sys.exit("at most 8 colours are supported")


def _common(a, time_s, seed):
    kw = dict(time=time_s, seed=seed, jobs=a.jobs, exact=not a.no_exact, verbose=not a.quiet,
              known="warn" if a.ignore_known else "stop", learn=a.learn)
    if getattr(a, "routes", ""):
        kw["routes"] = tuple(a.routes.split(","))
    if getattr(a, "flat_weights", False):
        kw["weights"] = "flat"
    if getattr(a, "init", ""):
        kw["init"] = a.init
    if a.spaces:
        kw["spaces"] = a.spaces
    return kw


def cmd_solve(a):
    _check(a)
    if not 1 <= a.n <= 512:
        sys.exit("n must be between 1 and 512")
    res = solve(a.ks, a.n, **_common(a, a.time, a.seed))
    name = _rname(a.ks)
    if res.solved:
        path = _save(res, a.ks, a.n, a.out)
        print(f"FOUND: {name} > {a.n}  [{res.seconds:.2f}s via {res.route}]  -> {path}")
        if res.record:
            print(f"NEW LOWER BOUND? {name} >= {a.n + 1} beats the best published bound in this program's "
                  "table. Check the current literature before claiming it.")
        if len(a.ks) == 2:
            print("graph6:", to_graph6(graph_of(res.coloring, 0)))
        return 0
    if res.impossible:
        print(f"PROVEN: no colouring of K_{a.n} exists, so {name} <= {a.n}  [{res.seconds:.2f}s]")
        print(f"  proof: {res.proof}")
        return 0
    if res.known_impossible:
        print(f"KNOWN: {res.note} (not re-proven here; --ignore-known searches anyway)")
        return 0
    best = _save_best(res, a.ks, a.n, a.out)
    print(f"not found within {a.time:.0f}s (and not proven impossible); "
          f"best weighted violation score {res.best_score:g}")
    if best:
        print(f"best near-solution saved to {best} (continue from it with --init {best})")
    return 1


def cmd_climb(a):
    _check(a)
    prev = None
    best_n = None
    name = _rname(a.ks)
    for n in range(a.n_from, min(a.n_to, 512) + 1):
        res = solve(a.ks, n, valid_below=prev, **_common(a, a.time_per_n, a.seed + n))
        if res.impossible:
            print(f"n={n}: PROVEN impossible in {res.seconds:.2f}s -- {res.proof}")
            if best_n == n - 1:
                print(f"so {name} = {n} exactly: a colouring of K_{n - 1} was found and K_{n} has none")
            else:
                print(f"so {name} <= {n}")
            break
        if res.known_impossible:
            print(f"n={n}: stopped -- {res.note} (not re-proven here; --ignore-known searches anyway)")
            break
        if not res.solved:
            best = _save_best(res, a.ks, n, a.out)
            print(f"stopped: no colouring of K_{n} found in {a.time_per_n:.0f}s (not proven impossible either)"
                  + (f"; best near-solution saved to {best}" if best else ""))
            break
        path = _save(res, a.ks, n, a.out)
        flag = "  ** beats the best published lower bound in this program's table **" if res.record else ""
        print(f"n={n}: found in {res.seconds:.2f}s via {res.route} -> {path}{flag}", flush=True)
        prev, best_n = res.coloring, n
    if best_n:
        print(f"largest certified: {name} >= {best_n + 1}")
    return 0


def cmd_verify(a):
    col, header = load_coloring(a.file)
    ks = a.ks or header.get("ks")
    if not ks:
        print("need --ks (forbidden clique sizes)")
        return 2
    if col.shape[0] > 90:
        print("checking (pure Python; large instances can take minutes) ...", flush=True)
    res = verify(col, ks)
    print(json.dumps({k: v for k, v in res.items() if k != "witness_cliques"}))
    if res["valid"]:
        print(f"VALID: certifies {_rname(ks)} >= {col.shape[0] + 1}")
        return 0
    print("INVALID: monochromatic forbidden cliques:", res.get("witness_cliques"))
    return 1


def cmd_compare(a):
    """Is the graph of colour 0 in a certificate isomorphic to a graph in a graph6 file?"""
    try:
        import networkx as nx
    except ImportError:
        print("the compare command needs networkx: pip install networkx")
        return 2
    from .verify import from_graph6
    col, _ = load_coloring(a.file)
    mine = graph_of(col, a.colour)

    def nxg(adj):
        g = nx.from_numpy_array(np.asarray(adj))
        tri = nx.triangles(g)
        for v in g.nodes:                       # vertex invariants speed up the matching a lot
            g.nodes[v]["inv"] = (g.degree[v], tri[v])
        return g

    def key(g):
        return (g.number_of_nodes(), g.number_of_edges(), sorted(g.nodes[v]["inv"] for v in g.nodes))

    g0 = nxg(mine)
    k0 = key(g0)
    same = lambda x, y: x["inv"] == y["inv"]
    with open(a.known) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith(">>")]
    hits = []
    for i, ln in enumerate(lines):
        adj = from_graph6(ln)
        if adj.shape[0] != mine.shape[0]:
            continue
        for label, cand in (("graph", adj), ("complement", 1 - adj - np.eye(adj.shape[0], dtype=adj.dtype))):
            g1 = nxg(cand)
            if key(g1) == k0 and nx.is_isomorphic(g0, g1, node_match=same):
                hits.append((i + 1, label))
    n_same = sum(1 for ln in lines if from_graph6(ln).shape[0] == mine.shape[0])
    if hits:
        print(f"KNOWN: colour {a.colour} of {a.file} is isomorphic to "
              + ", ".join(f"the {lab} of graph {i}" for i, lab in hits) + f" in {a.known}")
        return 0
    print(f"NEW: colour {a.colour} of {a.file} ({mine.shape[0]} vertices, {int(mine.sum()) // 2} edges) is not "
          f"isomorphic to any of the {n_same} graphs on {mine.shape[0]} vertices in {a.known}, nor to their complements")
    return 0


def _shared_options(p):
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jobs", type=int, default=1, help="parallel workers (0 = one per CPU core)")
    p.add_argument("--no-exact", action="store_true", help="do not run the complete full-space search")
    p.add_argument("--ignore-known", action="store_true",
                   help="search even if published results say the target is impossible")
    p.add_argument("--learn", action="store_true", help="policy-guided tabu starts on symmetric spaces")
    p.add_argument("--spaces", default="", help="only symmetric spaces whose name matches, e.g. 'circ*' "
                                                "or 'circ*,D*' (names are printed by the solver)")
    p.add_argument("--out", default="results")
    p.add_argument("--quiet", action="store_true")


def main(argv=None):
    p = argparse.ArgumentParser(prog="ramsey_zero", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("solve", help="find a colouring of K_n avoiding K_{k_c} in colour c")
    s.add_argument("ks", type=int, nargs="+", help="forbidden clique sizes, one per colour")
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--time", type=float, default=60.0)
    s.add_argument("--flat-weights", action="store_true", help="unweighted violation count")
    s.add_argument("--routes", default="", help="subset of sym,weak,descend,ascend,full")
    s.add_argument("--init", default="", help="start from this file: a colouring on n vertices (local "
                                              "search starts there) or valid colourings on n-1 vertices "
                                              "(the ascend route extends them); certificate or graph6 (.g6)")
    _shared_options(s)
    s.set_defaults(fn=cmd_solve)

    c = sub.add_parser("climb", help="solve n = from..to, feeding each colouring to the next size")
    c.add_argument("ks", type=int, nargs="+")
    c.add_argument("--from", dest="n_from", type=int, required=True)
    c.add_argument("--to", dest="n_to", type=int, required=True)
    c.add_argument("--time-per-n", type=float, default=60.0)
    _shared_options(c)
    c.set_defaults(fn=cmd_climb)

    v = sub.add_parser("verify", help="independently check a saved colouring")
    v.add_argument("file")
    v.add_argument("--ks", type=int, nargs="+")
    v.set_defaults(fn=cmd_verify)

    q = sub.add_parser("compare", help="check whether a found graph is isomorphic to a graph in a graph6 "
                                       "file (needs networkx)")
    q.add_argument("file", help="certificate written by solve or climb")
    q.add_argument("known", help="graph6 file, e.g. r46_35some.g6 from McKay's Ramsey graph page")
    q.add_argument("--colour", type=int, default=0, help="which colour class to compare (default 0)")
    q.set_defaults(fn=cmd_compare)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
