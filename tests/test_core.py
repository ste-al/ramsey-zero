"""Correctness tests. Run:  python tests/test_core.py   (or  python -m pytest -q tests)"""
import itertools
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ramsey_zero import (Engine, ExactSearch, OrbitSearch, abelian_group, block_circulant_space,
                         cayley_space, circulant_space, count_monochromatic, cyclic_group,
                         extension_space, full_space, metacyclic_group, save_coloring, solve,
                         symmetric_catalogue, upper_bound, verify, vertex_counts)
from ramsey_zero import known
from ramsey_zero.learner import OrbitPolicy, color_permutations
from ramsey_zero.solver import Config, RamseyZero
from ramsey_zero.spaces import (_closure, _orbits_slow, cyclotomic_spaces, field_group,
                                groups_of_order, multiplier_candidates)

RNG = np.random.default_rng(12345)


def rand_col(n, r):
    col = RNG.integers(0, r, size=(n, n)).astype(np.int8)
    col = np.triu(col, 1)
    col = col + col.T
    np.fill_diagonal(col, -1)
    return col


def brute_counts(col, ks):
    n = col.shape[0]
    out = []
    for c, k in enumerate(ks):
        out.append(sum(all(col[a, b] == c for a, b in itertools.combinations(S, 2))
                       for S in itertools.combinations(range(n), k)))
    return out


def weak_space(n, m):
    f = n % m
    return cayley_space(cyclic_group(m), blocks=(n - f) // m, fixed=f, sym=False)


# ------------------------------------------------------------------ counting kernels
def test_counter_matches_brute_force():
    for _ in range(40):
        n = int(RNG.integers(4, 13))
        r = int(RNG.integers(2, 4))
        ks = [int(RNG.integers(2, 8)) for _ in range(r)]
        col = rand_col(n, r)
        if r == 2 and RNG.random() < 0.5:  # skewed colourings: dense colour, large cliques
            col = np.where(RNG.random((n, n)) < 0.8, 0, 1).astype(np.int8)
            col = np.triu(col, 1)
            col = col + col.T
            np.fill_diagonal(col, -1)
        assert count_monochromatic(col, ks).tolist() == brute_counts(col, ks)


def test_vertex_counts():
    for _ in range(10):
        n, ks = 11, [3, 4]
        col = rand_col(n, 2)
        vc = vertex_counts(col, ks)
        for v in range(n):
            keep = [u for u in range(n) if u != v]
            removed = count_monochromatic(col, ks) - count_monochromatic(col[np.ix_(keep, keep)], ks)
            assert vc[:, v].tolist() == removed.tolist()


# ------------------------------------------------------------------ groups and spaces
def test_groups_act_regularly():
    for m in (6, 8, 12, 16, 18, 20, 21, 27):
        for G in groups_of_order(m, limit=99):
            assert len(_closure(m, G.gens)) == m, G.name
    assert [g.name for g in groups_of_order(16, limit=99)][:5] == ["Z16", "Z8xZ2", "Z4xZ4", "Z4xZ2xZ2", "Z2xZ2xZ2xZ2"]
    assert len(_closure(27, field_group(27).gens)) == 27


def test_vectorised_orbits_equal_reference():
    for G, b, f in ((cyclic_group(17), 1, 0), (cyclic_group(7), 3, 1), (abelian_group([4, 4]), 3, 2),
                    (metacyclic_group(7, 3, 2), 2, 0), (cyclic_group(2), 9, 1)):
        sp = cayley_space(G, blocks=b, fixed=f)
        m, n = G.order, sp.n
        gens = []
        for g in G.gens:
            lifted = np.arange(n)
            for i in range(b):
                lifted[i * m:(i + 1) * m] = i * m + g
            gens.append(lifted)
        ref, reps, rsz, _ = _orbits_slow(n, gens)
        assert len(ref) == sp.V and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(sp.orbits, ref))
        assert list(sp.reps) == list(reps) and list(sp.rsz) == list(rsz)


def test_symmetric_formula_equals_full_count():
    spaces = [circulant_space(17), circulant_space(22), circulant_space(41, (10,)),
              block_circulant_space(7, 3), block_circulant_space(6, 2, fixed=1),
              block_circulant_space(5, 3, fixed=2, multipliers=(2,)),
              cayley_space(abelian_group([4, 4])), cayley_space(abelian_group([2, 2, 2, 2])),
              cayley_space(metacyclic_group(7, 3, 2)), cayley_space(metacyclic_group(9, 2, 8)),
              cayley_space(abelian_group([4, 4]), blocks=3, fixed=2),
              cayley_space(metacyclic_group(5, 2, 4), blocks=2, fixed=1),
              cayley_space(field_group(27))] + [s for s, _ in cyclotomic_spaces(16)]
    for sp in spaces:
        for ks in ([3, 3], [4, 4], [3, 5], [4, 5], [3, 3, 3], [3, 7], [5, 6]):
            e = Engine(sp.n, ks, seed=1)
            e.set_space(sp)
            for _ in range(3):
                col = sp.coloring_from_x(RNG.integers(0, len(ks), size=sp.V))
                e.set_coloring(col)
                assert e.counts().tolist() == e.full_counts().tolist() == count_monochromatic(col, ks).tolist()


def test_catalogue_contains_the_classic_spaces():
    names = [s.name for s in symmetric_catalogue(50)]
    assert "circ50" in names and "3xZ4xZ4+2" in names and "D25" in names
    assert "GF16/3" in [s.name for s in symmetric_catalogue(16)]
    assert any(cnt == 10 for _, cnt in multiplier_candidates(101, max_orbits=30, min_orbits=2))


# ------------------------------------------------------------------ local search
def test_local_search_bookkeeping_all_modes():
    cases = [([3, 5], 14, None), ([4, 4], 18, None), ([3, 3, 3], 17, None), ([4, 5], 22, None),
             ([2, 4], 9, None), ([3, 7], 21, None), ([6, 6], 30, None),
             ([4, 4], 19, 2), ([4, 5], 23, 3), ([3, 7], 22, 4), ([3, 3, 4], 25, 2)]
    for ks, n, m in cases:
        sp = full_space(n) if m is None else weak_space(n, m)
        for mode in (0, 1, 2, 3):
            e = Engine(n, ks, seed=7 + mode)
            e.set_space(sp)
            e.set_coloring(sp.coloring_from_x(RNG.integers(0, len(ks), sp.V)))
            r = e.search(maxit=2000, time_limit=2, mode=mode, nsample=(0 if mode == 0 else 40))
            assert e.check_bad() == 0
            assert e.counts().tolist() == count_monochromatic(e.coloring(), ks).tolist()
            assert sp.is_invariant(e.coloring()) and sp.is_invariant(e.best())
            assert abs(r.best - float(count_monochromatic(e.best(), ks).sum())) < 1e-9


def test_extension_space_keeps_base_frozen():
    base = solve([3, 5], 13, time=10, verbose=False).coloring
    sp = extension_space(base, 1)
    e = Engine(14, [3, 5], seed=3)
    e.set_space(sp)
    e.set_coloring(sp.coloring_from_x(RNG.integers(0, 2, sp.V)))
    e.search(maxit=500, time_limit=1)
    assert np.array_equal(e.best()[:13, :13], base)


def test_time_limit_zero_stops_immediately():
    e = Engine(30, [4, 5], seed=1)
    e.set_space(full_space(30))
    e.set_coloring(rand_col(30, 2))
    r = e.search(time_limit=0.0)
    assert r.iters <= 1 and r.seconds < 0.1


def test_interrupted_searches_keep_the_bookkeeping_exact():
    """With large k one clique count can outlast a time slice, so the kernels can be stopped
    in the middle of a step.  Whatever was interrupted, the counts, the list of bad edges and
    the best-so-far colouring must stay exact."""
    rng = np.random.default_rng(2024)
    cases = [(full_space(40), [3, 11]), (weak_space(41, 2), [3, 11]), (weak_space(40, 4), [4, 9]),
             (circulant_space(51), [3, 11]), (cayley_space(metacyclic_group(7, 3, 2), blocks=2), [3, 10])]
    idle = moved = 0
    for sp, ks in cases:
        e = Engine(sp.n, ks, seed=5)
        e.set_space(sp)
        for dense in (0.85, 0.8, 0.75, 0.7):    # mostly colour 1: many large cliques, slow counts
            if e.set_coloring(sp.coloring_from_x((rng.random(sp.V) < dense).astype(np.int8)), time_limit=0.3):
                break
        assert e.ready
        for t in range(40):
            limit = float(rng.choice([0.0, 1e-3, 1e-2, 5e-2, 0.15]))
            kw = {} if sp.sym else dict(mode=int(rng.integers(0, 4)), nsample=int(rng.choice([0, 40])))
            before = e.coloring()
            r = e.search(time_limit=limit, stall=5, kick=3, **kw)
            assert r.seconds < limit + 0.5
            idle += r.iters == 0
            moved += r.iters > 0
            if r.iters == 0:
                assert np.array_equal(before, e.coloring())
            assert sp.is_invariant(e.coloring()) and sp.is_invariant(e.best())
            truth = count_monochromatic(e.coloring(), ks, time_limit=10)
            assert truth is not None and e.counts().tolist() == truth.tolist()
            best = count_monochromatic(e.best(), ks, time_limit=10)
            assert best is not None and abs(r.best - float(best.sum())) < 1e-9
            if not sp.sym:
                assert e.check_bad() in (0, -1)     # -1: the list of bad edges is not complete yet
    assert idle > 20 and moved > 20, (idle, moved)


def test_counting_gives_up_in_time_on_hopeless_colourings():
    n, ks = 90, [3, 20]
    col = np.ones((n, n), dtype=np.int8)        # K_90 in colour 1: about 5e19 forbidden cliques
    np.fill_diagonal(col, -1)
    t0 = time.time()
    assert count_monochromatic(col, ks, time_limit=0.05) is None
    assert vertex_counts(col, ks, time_limit=0.05) is None
    e = Engine(n, ks)
    e.set_space(full_space(n))
    assert e.set_coloring(col, time_limit=0.05) is False
    assert time.time() - t0 < 3.0
    try:
        e.search(time_limit=0.1)
        raise AssertionError("searching without a loaded colouring must fail")
    except RuntimeError:
        pass
    assert e.set_coloring(rand_col(n, 2)) is True       # the engine is usable again
    r = e.search(time_limit=0.2)
    assert e.counts().tolist() == count_monochromatic(e.coloring(), ks).tolist() and r.iters > 0


# ------------------------------------------------------------------ complete orbit search
def _brute_sym_exists(sp, ks):
    for x in itertools.product(range(len(ks)), repeat=sp.V):
        if count_monochromatic(sp.coloring_from_x(np.array(x)), ks).sum() == 0:
            return True
    return False


def test_orbit_search_agrees_with_brute_force():
    """Every ordering option must give the brute-force verdict on symmetric spaces."""
    spaces = [circulant_space(n) for n in range(4, 17)] + [
        block_circulant_space(4, 2), block_circulant_space(3, 3, 1),
        cayley_space(abelian_group([2, 2, 2])), cayley_space(abelian_group([3, 3])),
        cayley_space(metacyclic_group(5, 2, 4)), cayley_space(abelian_group([2, 2]), blocks=2, fixed=1)]
    checked = empty = 0
    for sp in spaces:
        for ks in ([3, 3], [3, 4], [4, 4], [3, 5], [2, 4], [3, 3, 3], [3, 3, 4]):
            if len(ks) ** sp.V > 7000:
                continue
            truth = _brute_sym_exists(sp, ks)
            empty += not truth
            for caps, rnd, phase, act in itertools.product((False, True), repeat=4):
                ox = OrbitSearch(sp, ks, degree_caps=caps, randomise=rnd, phase=phase, activity=act, seed=5)
                st = ox.run(None)
                assert st in (OrbitSearch.FOUND, OrbitSearch.EXHAUSTED)
                assert (st == OrbitSearch.FOUND) == truth, (sp.name, ks, caps, rnd, phase, act)
                if truth:
                    col = ox.solution()
                    assert count_monochromatic(col, ks).sum() == 0 and sp.is_invariant(col)
                checked += 1
    assert checked > 1000 and empty > 20


def test_orbit_search_restart_keeps_completeness():
    sp = circulant_space(43)                       # no circulant (5,5) colouring on 43 vertices
    ox = OrbitSearch(sp, [5, 5], randomise=True, activity=True, seed=1)
    limit, restarts = 50, 0
    while ox.status == OrbitSearch.RUNNING:
        ox.run(None, max_nodes=max(1, limit - ox.run_nodes))
        if ox.status == OrbitSearch.RUNNING and ox.run_nodes >= limit:
            restarts += 1
            limit = limit * 2
            ox.restart(restarts)
    assert ox.status == OrbitSearch.EXHAUSTED and restarts >= 3


def test_orbit_search_reproduces_circulant_ramsey_numbers():
    """Published values RC(k,l): a circulant colouring exists on RC-1 vertices and on none of
    RC, RC+1, RC+2 (Harborth & Krause 2003; Coniglio et al. 2026)."""
    for (k, l), rc in known.CIRCULANT_RAMSEY.items():
        if rc > 66:
            continue                               # keep the test quick; larger ones pass too
        ox = OrbitSearch(circulant_space(rc - 1), [k, l])
        assert ox.run(60) == OrbitSearch.FOUND, (k, l, rc)
        assert verify(ox.solution(), [k, l])["valid"]
        for n in (rc, rc + 1, rc + 2):
            assert OrbitSearch(circulant_space(n), [k, l]).run(60) == OrbitSearch.EXHAUSTED, (k, l, n)
    # the irregular "first kind" values: no circulant on 15 / 25 / 37 vertices although larger ones exist
    for ks, n in (([3, 6], 15), ([3, 8], 25), ([3, 10], 37), ([4, 6], 32), ([5, 5], 39)):
        assert OrbitSearch(circulant_space(n), ks).run(60) == OrbitSearch.EXHAUSTED
        assert OrbitSearch(circulant_space(n + 1), ks).run(60) == OrbitSearch.FOUND


def test_orbit_search_extension_matches_brute_force():
    for ks, n0 in (([3, 4], 7), ([3, 5], 11), ([3, 5], 12), ([4, 4], 14), ([3, 3, 3], 10)):
        for seed in range(2):
            base = solve(ks, n0, time=20, verbose=False, seed=seed).coloring
            sp = extension_space(base, 1)
            truth = any(count_monochromatic(sp.coloring_from_x(np.array(x)), ks).sum() == 0
                        for x in itertools.product(range(len(ks)), repeat=n0))
            ox = OrbitSearch(sp, ks)
            st = ox.run(None)
            assert (st == OrbitSearch.FOUND) == truth
            if truth:
                assert np.array_equal(ox.solution()[:n0, :n0], base)
                assert count_monochromatic(ox.solution(), ks).sum() == 0
    bad = np.zeros((5, 5), np.int8)                # a base that already contains a triangle
    np.fill_diagonal(bad, -1)
    assert OrbitSearch(extension_space(bad, 1), [3, 3]).status == OrbitSearch.EXHAUSTED


def test_four_colour_record_space():
    """R(3,3,3,3) >= 51: the 3 x (Z4 x Z4) + 2 space holds a colouring; the Z16 version is empty."""
    sp = cayley_space(abelian_group([4, 4]), blocks=3, fixed=2)
    ox = OrbitSearch(sp, [3, 3, 3, 3], randomise=True, activity=True, seed=3)
    limit, restarts = 200, 0
    while ox.status == OrbitSearch.RUNNING and restarts < 60:
        ox.run(None, max_nodes=max(1, limit - ox.run_nodes))
        if ox.status == OrbitSearch.RUNNING:
            restarts += 1
            limit = int(limit * 1.5) + 1
            ox.restart(restarts)
    assert ox.status == OrbitSearch.FOUND and verify(ox.solution(), [3, 3, 3, 3])["valid"]
    assert OrbitSearch(cayley_space(cyclic_group(16), blocks=3, fixed=2), [3, 3, 3, 3]).run(30) == OrbitSearch.EXHAUSTED


# ------------------------------------------------------------------ proofs of impossibility
def test_recursive_upper_bounds():
    assert upper_bound([3, 3]) == 6 and upper_bound([3, 4]) == 9 and upper_bound([3, 5]) == 14
    assert upper_bound([4, 4]) == 18 and upper_bound([3, 3, 3]) == 17
    assert upper_bound([2, 7]) == 7 and upper_bound([5]) == 5
    # only an upper bound where the recursion is not tight (true values 18, 25, <=46)
    assert upper_bound([3, 6]) == 19 and upper_bound([4, 5]) == 31 and upper_bound([5, 5]) == 62


def _brute_exists(ks, n):
    edges = [(u, v) for v in range(1, n) for u in range(v)]
    col = np.full((n, n), -1, np.int8)
    for assign in itertools.product(range(len(ks)), repeat=len(edges)):
        for (u, v), c in zip(edges, assign):
            col[u, v] = col[v, u] = c
        if count_monochromatic(col, ks).sum() == 0:
            return True
    return False


def test_exact_search_agrees_with_brute_force():
    """Both complete searches, with every combination of pruning rules, against brute force."""
    for ks in ([3, 3], [2, 3], [3, 4], [2, 2, 3], [2, 4], [3, 3, 2], [2, 2, 2]):
        for n in range(1, 8):
            if len(ks) ** (n * (n - 1) // 2) > 300_000:
                continue
            truth = _brute_exists(ks, n)
            for flags in range(8):
                ex = ExactSearch(ks, n, flags=flags)
                st = ex.run(30)
                assert st in (ExactSearch.FOUND, ExactSearch.IMPOSSIBLE)
                assert (st == ExactSearch.FOUND) == truth, (ks, n, flags)
                if st == ExactSearch.FOUND:
                    assert count_monochromatic(ex.solution(), ks).sum() == 0
            if n >= 2:
                for caps in (False, True):
                    ox = OrbitSearch(full_space(n), ks, degree_caps=caps)
                    assert (ox.run(30) == OrbitSearch.FOUND) == truth, (ks, n, caps)


def test_exact_search_known_values():
    for ks, R in (([3, 3], 6), ([3, 4], 9), ([3, 5], 14)):
        ex = ExactSearch(ks, R - 1)
        assert ex.run(60) == ExactSearch.FOUND and verify(ex.solution(), ks)["valid"]
        assert ExactSearch(ks, R).run(60) == ExactSearch.IMPOSSIBLE
        # the same verdict from pure search, without the degree-bound shortcut
        assert ExactSearch(ks, R, flags=3).run(120) == ExactSearch.IMPOSSIBLE


# ------------------------------------------------------------------ literature table
def test_known_table_is_consistent():
    for ks, lo in known.LOWER.items():
        up = known.UPPER.get(ks)
        assert lo >= max(ks), ks
        if up is not None:
            assert lo <= up, ks
        assert lo <= upper_bound(ks), ks           # the recursion is a valid upper bound
    for ks, up in known.UPPER.items():
        assert up <= upper_bound(ks), ks           # published bounds are at least as strong
    for (k, l), rc in known.CIRCULANT_RAMSEY.items():
        assert rc <= known.LOWER[(k, l)], (k, l)   # a circulant on rc-1 vertices gives R >= rc
    assert known.describe([3, 6], 18)[0] == "impossible" and known.describe([6, 3], 17)[0] == "known"
    assert known.describe([5, 5], 42)[0] == "known" and known.describe([5, 5], 43)[0] == "record"
    assert known.describe([5, 5], 46)[0] == "impossible" and known.describe([7, 30], 100)[0] == "unknown"
    assert known.describe([5, 7], 80)[0] == "record" and known.describe([3, 17], 92)[0] == "known"
    assert known.describe([3, 5, 5], 121)[0] == "known" and known.describe([3, 4, 8], 193)[0] == "record"


# ------------------------------------------------------------------ the solver
def test_solver_proves_or_reports_impossible_targets():
    res = solve([3, 3], 6, time=99999999, verbose=False)   # used to run forever
    assert res.impossible and not res.solved and res.seconds < 5
    res = solve([3, 3, 3], 17, time=99999999, verbose=False)
    assert res.impossible and res.seconds < 5
    res = solve([3, 6], 18, time=99999999, verbose=False)  # not provable here, but published
    assert res.known_impossible and not res.impossible and res.seconds < 5 and res.status == "known impossible"
    res = solve([3, 6], 18, time=1.5, verbose=False, known="warn")
    assert not res.solved and not res.known_impossible and res.status == "unsolved"


def test_known_values_are_found_and_verified():
    for ks, n in (([3, 5], 13), ([3, 6], 17), ([3, 7], 22), ([4, 4], 17), ([3, 3, 3], 16), ([4, 5], 24),
                  ([3, 9], 35), ([3, 3, 4], 29), ([5, 5], 42), ([6, 6], 101)):
        res = solve(ks, n, time=180, verbose=False, seed=0)
        assert res.solved, (ks, n)
        v = verify(res.coloring, ks)
        assert v["valid"] and all(w < k for w, k in zip(v["clique_numbers"], ks))
        assert not res.record                      # all of these are known sizes


def test_parallel_workers():
    res = solve([5, 5], 42, time=120, verbose=False, jobs=2)
    assert res.solved and verify(res.coloring, [5, 5])["valid"]
    assert solve([3, 3], 6, time=60, verbose=False, jobs=2).impossible
    res = solve([4, 6], 36, time=3, verbose=False, jobs=2)      # open problem: must stop on time
    assert not res.solved and res.seconds < 15


def test_init_and_best_near_solution():
    res = solve([4, 6], 35, time=3, verbose=False, routes=("full",))
    assert not res.solved and res.best_coloring is not None and res.best_coloring.shape == (35, 35)
    good = solve([4, 5], 24, time=30, verbose=False).coloring
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c.txt")
        save_coloring(path, good, [4, 5])
        again = solve([4, 5], 24, time=30, verbose=False, init=path)
    assert again.solved and again.route == "init"


def test_graph6_and_smaller_starting_colourings():
    from ramsey_zero.verify import coloring_from_graph, from_graph6, graph_of, load_colorings, to_graph6
    for n in (1, 2, 7, 30, 63, 64, 100):
        adj = (RNG.random((n, n)) < 0.5).astype(np.int8)
        adj = np.triu(adj, 1)
        adj = adj + adj.T
        assert np.array_equal(from_graph6(to_graph6(adj)), adj)
    base = solve([4, 5], 23, time=30, verbose=False, seed=3).coloring
    with tempfile.TemporaryDirectory() as d:
        g6 = os.path.join(d, "r45_23.g6")
        with open(g6, "w") as f:                # a graph6 file with two graphs
            f.write(to_graph6(graph_of(base, 0)) + "\n" + to_graph6(graph_of(base, 0)) + "\n")
        cols = load_colorings(g6)
        assert len(cols) == 2 and np.array_equal(cols[0], coloring_from_graph(graph_of(base, 0)))
        assert verify(cols[0], [4, 5])["valid"]
        res = solve([4, 5], 24, time=60, verbose=False, init=g6, routes=("ascend",), exact=False)
    assert res.solved and res.route == "ascend" and verify(res.coloring, [4, 5])["valid"]


def test_compare_finds_relabelled_and_complemented_copies():
    try:
        import networkx  # noqa: F401
    except ImportError:
        return                                  # optional dependency
    from ramsey_zero.__main__ import main
    from ramsey_zero.verify import graph_of, to_graph6
    col = solve([4, 5], 24, time=30, verbose=False, seed=1).coloring
    A = graph_of(col, 0)
    p = RNG.permutation(24)
    with tempfile.TemporaryDirectory() as d:
        cert = os.path.join(d, "c.txt")
        save_coloring(cert, col, [4, 5])
        for name, g, word in (("same", A[np.ix_(p, p)], "KNOWN"),
                              ("comp", (1 - A - np.eye(24, dtype=A.dtype))[np.ix_(p, p)], "KNOWN"),
                              ("diff", np.where(np.arange(24)[:, None] + np.arange(24) == 1, 1 - A, A), "NEW")):
            g6 = os.path.join(d, name + ".g6")
            with open(g6, "w") as f:
                f.write(to_graph6(g) + "\n")
            import contextlib, io
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                main(["compare", cert, g6])
            assert buf.getvalue().startswith(word), (name, buf.getvalue())


def test_time_limit_is_kept_on_large_instances():
    # large k: single clique counts used to overrun the limit by tens of seconds
    for ks, n, limit in (([3, 19], 105, 4.0), ([6, 8], 120, 3.0)):
        t0 = time.time()
        res = solve(ks, n, time=limit, verbose=False, known="warn")
        assert time.time() - t0 < limit + 6.0, (ks, n, time.time() - t0)
        assert res.solved or res.best_score > 0


def test_overrunning_arms_pay_the_time_back():
    rz = RamseyZero([4, 5], 24, Config(time=1000, verbose=False))
    for i in range(len(rz.arms)):
        rz.bandit.update(i, 0.5)
    rz.t0 = time.time()
    rz.arms[0].debt = 2.0                       # ran 2 s over; slices are 0.1 s
    picks = []
    for _ in range(40 * len(rz.arms)):
        i = rz._pick_local(0.1)
        if i is not None:
            rz.bandit.update(i, 0.5)
            picks.append(i)
    first = picks.index(0)
    assert first >= 2 * (len(rz.arms) - 1)      # the others ran at least twice each meanwhile
    assert rz.arms[0].debt <= 0 and picks.count(0) < picks.count(1)
    rz.arms[1].last = 1e9                       # an arm that could not finish in the time left
    assert all(rz._pick_local(0.1) != 1 for _ in range(50))


def test_verifier_rejects_bad_colourings():
    col = rand_col(20, 2)  # random 2-colouring of K_20 certainly contains a mono K_4
    assert not verify(col, [4, 4])["valid"]


def test_policy_alignment_recovers_symmetric_copy():
    sp = circulant_space(31)
    pol = OrbitPolicy(sp.V, 2, rng=np.random.default_rng(0))
    target = np.random.default_rng(1).integers(0, 2, sp.V).astype(np.int8)
    pol.p = np.where(target[:, None] == np.arange(2)[None, :], 0.9, 0.1)
    perms = sp.orbit_perms()
    g = perms[5]
    image = np.empty_like(target)
    image[g] = target                 # an isomorphic copy in another "gauge"
    back = pol.align(image, perms, color_permutations([4, 4]))
    assert pol.loglik(back) >= pol.loglik(target) - 1e-9


if __name__ == "__main__":
    import time
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t = time.time()
            fn()
            print(f"ok {name}  ({time.time() - t:.1f}s)", flush=True)
