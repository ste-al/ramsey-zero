"""RamseyZero orchestrator: zero-knowledge search for Ramsey colourings.

Target: colour the edges of K_n with r colours so that colour c has no K_{k_c}.

Two tiers share the time budget.

  S  Complete search on symmetric spaces.  A catalogue of spaces (circulant, abelian and
     metacyclic Cayley, block Cayley with fixed points, cyclotomic) is generated for n.
     Each space is searched by OrbitSearch, which either finds a colouring or proves the
     space empty and retires it.  Open spaces are served least-served-first, with time in
     proportion to weight / (number of orbits).

  L  Local search, chosen by a UCB bandit that learns from search outcomes:
       weak      tabu on a space with a small symmetry group (one involution, Z_3, ...)
       full      tabu on the full edge space
       descend   near-solution on circulant n+d -> delete d vertices -> repair
       ascend    valid colouring on n-1 -> exact one-vertex extension, else add and repair

Alongside: impossibility proofs (a recursive bound, then a complete search for small n) and
a literature table used only for reporting.  Every reported colouring is recounted in full
and re-checked by an independent exact verifier.

Time limits are kept even where a single clique count is slow (large k): the C kernels can
be interrupted inside a step, and a local-search arm that still overruns its slice sits out
later turns until the excess is paid back.
"""
from __future__ import annotations

import fnmatch
import math
import os
import threading
import time
from dataclasses import dataclass, field, replace

import numpy as np

from . import known
from .core import Engine, count_monochromatic, vertex_counts
from .exact import ExactSearch, OrbitSearch, upper_bound
from .learner import UCB, OrbitPolicy, color_permutations
from .spaces import (_signature, cayley_space, circulant_space, cyclic_group, extension_space,
                     full_space, symmetric_catalogue)
from .verify import load_coloring, load_colorings, save_coloring, verify

# One clock for every duration and deadline here.  time.time() can tick as coarsely as
# 15.6 ms on Windows, which is the size of the shortest time slices.
now = time.perf_counter


def default_weights(ks, mode="auto"):
    if isinstance(mode, (list, tuple, np.ndarray)):
        return [float(x) for x in mode]
    if mode in (None, "flat", "none"):
        return [1.0] * len(ks)
    kmax = max(ks)
    return [(kmax / k) ** 1.5 for k in ks]  # Exoo-style: small forbidden cliques weigh more


def random_reference(n, ks, w):
    r = len(ks)
    return sum(wc * math.comb(n, k) * float(r) ** (-math.comb(k, 2)) for wc, k in zip(w, ks))


def random_coloring(n, r, rng):
    col = rng.integers(0, r, size=(n, n)).astype(np.int8)
    col = np.triu(col, 1)
    col = col + col.T
    np.fill_diagonal(col, -1)
    return col


def greedy_delete(col, ks, w, d, deadline=None):
    """Remove d vertices, each time the one in the most (weighted) violated cliques.
    Returns None if the counting is not done by `deadline` (a now() value)."""
    w = np.asarray(w)
    for _ in range(d):
        left = None if deadline is None else max(0.0, deadline - now())
        vc = vertex_counts(col, ks, time_limit=left)  # (r, n)
        if vc is None:
            return None
        v = int(np.argmax((w[:, None] * vc).sum(0)))
        keep = np.r_[0:v, v + 1:col.shape[0]]
        col = col[np.ix_(keep, keep)].copy()
    return col


WEAK_ORDERS = (2, 3, 4, 5, 7)   # cyclic groups Z_m used for the weak-symmetry arms


@dataclass
class Config:
    time: float = 60.0         # total wall-clock budget (s)
    seed: int = 0
    jobs: int = 1              # parallel workers (threads; the C kernels release the GIL)
    weights: object = "auto"
    routes: tuple = ("sym", "weak", "descend", "ascend", "full")
    exact: bool = True         # complete search alongside (n <= 64): can PROVE impossibility
    known: str = "stop"        # literature says impossible: "stop", "warn" (search anyway) or "ignore"
    learn: bool = False        # cross-entropy policy for tabu starts on symmetric spaces (measured: neutral)
    batch: int = 8             # tabu starts per generation on a symmetric space
    per_start: float = 0.25    # max seconds of tabu search per start (symmetric spaces)
    elite: float = 0.25
    lr: float = 0.25
    slice: float = 4.0         # max seconds per pull
    min_slice: float = 0.05    # first pull of every arm; doubles afterwards
    max_blocks: int = 6        # symmetric catalogue: at most this many blocks ...
    max_fixed: int = 2         # ... and fixed points
    nonabelian: bool = True
    max_spaces: int = 64
    spaces: str = ""           # only symmetric spaces whose name matches (shell patterns, comma separated)
    sym_share: float = 0.75    # largest share of the time for tier S (complete search on symmetric spaces)
    heartbeat: float = 10.0    # first status line after this many quiet seconds; interval doubles to 120 s
    init: object = None        # optional starting colouring (n x n matrix or file) for the local search
    verbose: bool = True


@dataclass
class Result:
    solved: bool
    coloring: np.ndarray = None
    route: str = ""
    seconds: float = 0.0
    best_score: float = math.inf
    verification: dict = None
    log: list = field(default_factory=list)
    impossible: bool = False        # True: PROVEN here that no valid colouring of K_n exists
    proof: str = ""
    known_impossible: bool = False  # True: published results rule the target out (not re-proven here)
    note: str = ""                  # what the literature says about this target
    record: bool = False            # solved AND better than the best published lower bound in known.py
    best_coloring: np.ndarray = None  # best near-solution when unsolved (resume with init=)

    @property
    def status(self):
        if self.solved:
            return "solved"
        if self.impossible:
            return "proven impossible"
        if self.known_impossible:
            return "known impossible"
        return "unsolved"


@dataclass
class Arm:
    kind: str
    name: str
    space: object = None
    d: int = 0
    eng: object = None
    live: bool = False
    kernel: dict = None         # tabu settings of the running trajectory
    debt: float = 0.0           # seconds this arm ran beyond its slices and has not paid back yet
    last: float = 0.0           # duration of its latest pull


class SymArm:
    """One symmetric space under complete search.

    Two OrbitSearch runs share each time slice.  For the first worker: a deterministic run in
    natural orbit order that is never restarted (best for proving the space empty) and a
    randomised run with conflict-driven ordering and geometric restarts (best for finding a
    colouring).  Other parallel workers use two randomised restart runs instead of repeating
    the deterministic one; the first of them also tries the colours in random order."""

    def __init__(self, spec):
        self.spec, self.name, self.est_v = spec, spec.name, spec.est_v
        self.cost = spec.est_v / spec.weight    # time share is proportional to 1 / cost
        self.runs = None          # [search, restart node limit or None]
        self.spent, self.dead, self.nodes, self.restarts = 0.0, False, 0, 0

    def pull(self, ks, seconds, seed, deterministic=True):
        """Returns (status, colouring or None); status is an OrbitSearch constant."""
        sp = self.spec.space()
        t0 = now()
        if self.runs is None:
            first = (OrbitSearch(sp, ks, randomise=False) if deterministic
                     else OrbitSearch(sp, ks, randomise=True, activity=True, values=None, seed=seed + 1))
            self.runs = [[first, None if deterministic else 200],
                         [OrbitSearch(sp, ks, randomise=True, activity=True, seed=seed), 200]]
        out = (OrbitSearch.RUNNING, None)
        # the first run proves, the second finds: equal shares at first, then 1 : 3
        shares = (0.5, 0.5) if self.spent < 2.0 else (0.25, 0.75)
        for run, share in zip(self.runs, shares):
            ox, limit = run
            end = now() + seconds * share
            while ox.status == OrbitSearch.RUNNING and now() < end:
                if limit is None:
                    ox.run(end - now())
                else:
                    ox.run(end - now(), max_nodes=max(1, limit - ox.run_nodes))
                    if ox.status == OrbitSearch.RUNNING and ox.run_nodes >= limit:
                        self.restarts += 1
                        limit = run[1] = int(limit * 1.5) + 1
                        ox.restart(seed + 7919 * self.restarts)
            if ox.status == OrbitSearch.FOUND:
                out = (OrbitSearch.FOUND, ox.solution())
                break
            if ox.status == OrbitSearch.EXHAUSTED:
                self.dead = True
                out = (OrbitSearch.EXHAUSTED, None)
                break
        self.nodes = sum(r[0].nodes for r in self.runs)
        self.spent += now() - t0
        return out


class Shared:
    """State shared by parallel workers."""

    def __init__(self):
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.dead = {}          # symmetric spaces proven empty: name -> number of orbits


class RamseyZero:
    def __init__(self, ks, n, cfg: Config | None = None, worker=0, shared=None):
        self.ks, self.n, self.r = [int(k) for k in ks], int(n), len(ks)
        self.cfg = cfg or Config()
        self.worker, self.shared = worker, shared or Shared()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.w = default_weights(self.ks, self.cfg.weights)
        self.ref = max(random_reference(n, self.ks, self.w), 1.0)
        self.cperms = color_permutations(self.ks)
        self.engines, self.policies, self._spi, self._circ = {}, {}, {}, {}
        self.valid = {}   # m -> list of valid colourings on m vertices
        self.near = {}    # m -> list of (score, colouring) near-solutions
        self._ext_tested = set()
        self.sym_arms, self.arms = self._build_arms()
        self._sigs = set()
        self._sym_summary = False
        self.bandit = UCB(max(1, len(self.arms)), rng=self.rng)
        self.exact = None
        self.t0 = now()
        self.log = []
        self.best_overall = math.inf
        self.l_moves = 0          # local-search moves made so far (all local arms)

    # ---------------------------------------------------------------- setup
    def _build_arms(self):
        n, routes = self.n, self.cfg.routes
        sym, arms = [], []
        if "sym" in routes and n >= 4:
            cap = self.cfg.max_spaces if n <= 128 else (24 if n <= 256 else 8)
            specs = symmetric_catalogue(n, self.cfg.max_blocks, self.cfg.max_fixed, self.cfg.nonabelian,
                                        max_spaces=cap if not self.cfg.spaces else 10 ** 6)
            if self.cfg.spaces:
                pats = [p.strip() for p in self.cfg.spaces.split(",") if p.strip()]
                specs = [s for s in specs if any(fnmatch.fnmatchcase(s.name, p) for p in pats)][:cap]
            sym = [SymArm(s) for s in specs]
        if "weak" in routes and n >= 8:
            seen = set()
            for m in WEAK_ORDERS:
                for f in ((n % m, n % m + m) if m == 2 else (n % m,)):
                    if (n - f) // m < 2 or (m, f) in seen:
                        continue
                    seen.add((m, f))
                    arms.append(Arm("weak", f"weak:Z{m}+{f}", d=m * 1000 + f))
        if "descend" in routes:
            for d in (0, 1, 2):
                arms.append(Arm("descend", f"descend+{d}", d=d))
        if "ascend" in routes and n > 4:
            arms.append(Arm("ascend", "ascend"))
        if "full" in routes:
            arms.append(Arm("full", "full"))
        return sym, arms

    def _circulant(self, m):
        if m not in self._circ:
            self._circ[m] = circulant_space(m)
        return self._circ[m]

    def _engine(self, space):
        key = (space.n, space.name)
        if key not in self.engines:
            e = Engine(space.n, self.ks, self.w, seed=int(self.rng.integers(1 << 62)))
            e.set_space(space)
            self.engines[key] = e
        e = self.engines[key]
        e.seed(int(self.rng.integers(1 << 62)))
        return e

    def _full_engine(self, m):
        key = (m, f"full{m}")
        if key not in self.engines:
            return self._engine(full_space(m))
        e = self.engines[key]
        e.seed(int(self.rng.integers(1 << 62)))
        return e

    def _policy(self, space):
        key = (space.n, space.name)
        if key not in self.policies:
            self.policies[key] = OrbitPolicy(space.V, self.r, lr=self.cfg.lr, rng=self.rng)
        return self.policies[key]

    def _remaining(self):
        if self.shared.stop.is_set():
            return 0.0
        return self.cfg.time - (now() - self.t0)

    def _say(self, msg):
        tag = f"w{self.worker} " if self.cfg.jobs > 1 else ""
        line = f"[{now() - self.t0:7.1f}s] {tag}{msg}"
        self.log.append(line)
        if self.cfg.verbose:
            with self.shared.lock:
                print(line, flush=True)

    def _note_near(self, m, score, col, keep=6):
        lst = self.near.setdefault(m, [])
        if any(abs(s - score) < 1e-9 and np.array_equal(c, col) for s, c in lst):
            return
        lst.append((score, col.copy()))
        lst.sort(key=lambda t: t[0])
        del lst[keep:]

    def _reward(self, best):
        if best <= 0:
            return 1.0
        return float(np.clip(1.0 - math.log1p(best) / math.log1p(self.ref), 0.0, 1.0))

    def _ks_str(self):
        return ",".join(map(str, self.ks))

    # ---------------------------------------------------------------- primitives
    def sym_generation(self, space, deadline, learn=False):
        """Tabu from a batch of starts on a symmetric space (optionally policy-guided).
        Returns (best score, colouring, solved)."""
        eng = self._engine(space)
        pol = self._policy(space) if learn else None
        X = pol.sample(self.cfg.batch) if learn else self.rng.integers(0, self.r, (self.cfg.batch, space.V))
        res_x, res_s = [], []
        best, best_col = math.inf, None
        per = min(self.cfg.per_start, max(0.02, (deadline - now()) / 4))
        key = (space.n, space.name)
        spi = self._spi.get(key)        # measured seconds per tabu iteration
        if spi is not None:             # every start gets at least one descent pass, <= 5 s
            per = max(per, min(5.0, (space.V + 10) * spi))
        for i, x in enumerate(X):
            if i > 0 and now() >= deadline:
                break
            load = min(max(per, deadline - now()), max(0.0, self._remaining()))
            if not eng.set_coloring(space.coloring_from_x(x), time_limit=load):
                continue                # too many cliques to count in time: next start
            if spi is None:             # calibrate the cost of one iteration on the first start
                cal = eng.search(maxit=5, time_limit=min(5.0, max(0.0, self._remaining())))
                self.l_moves += cal.iters
                spi = self._spi[key] = cal.seconds / max(1, cal.iters)
                per = max(per, min(5.0, (space.V + 10) * spi))
                if cal.solved:
                    return 0.0, eng.best(), True
            res = eng.search(time_limit=min(per, max(0.0, self._remaining())))
            self.l_moves += res.iters
            col = eng.best()
            res_x.append(space.x_from_coloring(col))
            res_s.append(res.best)
            if res.best < best:
                best, best_col = res.best, col
            if res.solved:
                return 0.0, col, True
        if learn and len(res_s) >= 2:
            order = np.argsort(res_s)
            k = max(1, int(math.ceil(self.cfg.elite * len(res_s))))
            perms = space.orbit_perms()
            pol.update([pol.align(res_x[i], perms, self.cperms) for i in order[:k]])
        return best, best_col, False

    def polish(self, col, deadline):
        """Focused tabu in the full edge space of col's size."""
        m = col.shape[0]
        eng = self._full_engine(m)
        if not eng.set_coloring(col, time_limit=max(0.0, deadline - now())):
            return math.inf, col, False     # no time left to count its cliques
        res = eng.search(time_limit=max(0.0, deadline - now()), **self._kernel())
        self.l_moves += res.iters
        out = eng.best()
        if not res.solved:
            self._note_near(m, res.best, out)
        return res.best, out, res.solved

    def _kernel(self):
        """Mixed tabu settings, varied between runs: share of focused steps and number of
        sampled candidates (fewer = faster steps, more = steeper descent)."""
        return dict(mode=3, temp=float(self.rng.choice((0.3, 0.5))), nsample=int(self.rng.choice((60, 150))))

    def extend_exact(self, base, seconds):
        """Complete one-vertex extension of a valid colouring.  Returns a colouring or None."""
        key = hash(base.tobytes())
        if key in self._ext_tested:
            return None
        self._ext_tested.add(key)
        ox = OrbitSearch(extension_space(base, 1), self.ks)
        if ox.run(seconds) == OrbitSearch.FOUND:
            return ox.solution()
        return None

    def extend_one(self, col, deadline):
        """Add one vertex: tabu on its row (old graph frozen), then full repair."""
        m = col.shape[0]
        sp = extension_space(col, 1)
        eng = Engine(m + 1, self.ks, self.w, seed=int(self.rng.integers(1 << 62)))
        eng.set_space(sp)
        if self.rng.random() < 0.5:  # clone a random vertex's row and perturb it
            u = int(self.rng.integers(m))
            x = col[u].copy()
            x[u] = self.rng.integers(self.r)
            flip = self.rng.random(m) < 0.1
            x[flip] = self.rng.integers(0, self.r, flip.sum())
        else:
            x = self.rng.integers(0, self.r, m).astype(np.int8)
        left = deadline - now()
        if not eng.set_coloring(sp.coloring_from_x(x), time_limit=max(0.0, left)):
            return math.inf, None, False
        res = eng.search(time_limit=max(0.0, min(0.3 * left, 2.0)), mode=1, nsample=0)
        self.l_moves += res.iters
        out = eng.best()
        if res.solved:
            return 0.0, out, True
        return self.polish(out, deadline)

    def _add_valid(self, m, col, cap=24):
        lst = self.valid.setdefault(m, [])
        if not any(np.array_equal(col, c) for c in lst):
            lst.append(col)
            del lst[:-cap]

    def get_valid(self, m, deadline, depth=0):
        """A valid colouring on m vertices: from the pool (kept diverse), an exact circulant,
        a repaired circulant near-solution, or grown from m-1."""
        pool = self.valid.get(m)
        if pool and (len(pool) >= 24 or self.rng.random() < 0.5):
            return pool[int(self.rng.integers(len(pool)))]
        sp = self._circulant(m)
        stop = deadline if not pool else min(deadline, now() + 0.25 * (deadline - now()))
        if not pool and ("circ", m) not in self._ext_tested:     # exact circulant, once per size
            self._ext_tested.add(("circ", m))
            ox = OrbitSearch(sp, self.ks)
            if ox.run(min(0.5, max(0.0, stop - now()))) == OrbitSearch.FOUND:
                self._add_valid(m, ox.solution())
                return ox.solution()
        if now() < stop:
            best, col, ok = self.sym_generation(sp, stop, learn=self.cfg.learn)
            if ok:
                self._add_valid(m, col)
                return col
            if col is not None:
                b2, c2, ok2 = self.polish(col, min(stop, now() + 1.0))
                if ok2:
                    self._add_valid(m, c2)
                    return c2
        if pool:
            return pool[int(self.rng.integers(len(pool)))]
        if depth < 3 and m > 6 and now() < deadline:
            base = self.get_valid(m - 1, now() + 0.5 * (deadline - now()), depth + 1)
            if base is not None:
                ext = self.extend_exact(base, min(0.5, max(0.0, deadline - now())))
                if ext is not None:
                    self._add_valid(m, ext)
                    return ext
                b, col, ok = self.extend_one(base, deadline)
                if ok:
                    self._add_valid(m, col)
                    return col
        return None

    # ---------------------------------------------------------------- local arms
    def _resume_or_restart(self, arm, space, deadline, near_ok):
        """Persistent tabu engine: mostly resume the running trajectory, sometimes restart."""
        if arm.eng is None:
            arm.eng = Engine(self.n, self.ks, self.w, seed=int(self.rng.integers(1 << 62)))
            arm.eng.set_space(space)
            arm.live = False
        eng = arm.eng
        if not (arm.live and self.rng.random() < 0.75):
            near = self.near.get(self.n) if near_ok else None
            if near and self.rng.random() < 0.5:
                start = near[int(self.rng.integers(len(near)))][1]
            elif space.singleton:
                start = random_coloring(self.n, self.r, self.rng)
            else:
                start = space.coloring_from_x(self.rng.integers(0, self.r, space.V))
            arm.live = False
            if not eng.set_coloring(start, time_limit=max(0.0, deadline - now())):
                return math.inf, None, False    # too many cliques to count in this slice
            arm.live = True
            arm.kernel = self._kernel()
        res = eng.search(time_limit=max(0.0, deadline - now()), **arm.kernel)
        self.l_moves += res.iters
        out = eng.best()
        if not res.solved:
            self._note_near(self.n, res.best, out)
        return res.best, out, res.solved

    def pull(self, arm, slice_s):
        deadline = now() + slice_s
        n = self.n
        if arm.kind == "weak":
            if arm.space is None:
                m, f = divmod(arm.d, 1000)
                arm.space = cayley_space(cyclic_group(m), blocks=(n - f) // m, fixed=f, sym=False)
                arm.space.name = arm.name
            return self._resume_or_restart(arm, arm.space, deadline, near_ok=False)
        if arm.kind == "full":
            if arm.space is None:
                arm.space = full_space(n)
            return self._resume_or_restart(arm, arm.space, deadline, near_ok=True)
        if arm.kind == "descend":
            sp = self._circulant(n + arm.d)
            mid = now() + 0.5 * slice_s
            best_big, col_big = math.inf, None
            while now() < mid and self._remaining() > 0:
                b, c, ok = self.sym_generation(sp, mid, learn=self.cfg.learn)
                if c is not None and b < best_big:
                    best_big, col_big = b, c
                if ok:
                    break
            if col_big is None:
                return math.inf, None, False
            if best_big == 0:           # valid on n + d vertices: any n of them are valid too
                return 0.0, np.ascontiguousarray(col_big[:n, :n]), True
            col = greedy_delete(col_big, self.ks, self.w, arm.d, deadline)
            if col is None:
                return math.inf, None, False
            return self.polish(col, deadline)
        if arm.kind == "ascend":
            base = self.get_valid(n - 1, now() + 0.5 * slice_s)
            if base is None:
                return math.inf, None, False
            ext = self.extend_exact(base, min(1.0, 0.25 * slice_s))
            if ext is not None:
                return 0.0, ext, True
            return self.extend_one(base, deadline)
        raise ValueError(arm.kind)

    # ---------------------------------------------------------------- results
    def _result(self, **kw):
        near = self.near.get(self.n)
        base = dict(solved=False, seconds=now() - self.t0, best_score=self.best_overall,
                    log=self.log, note=self._note, best_coloring=near[0][1] if near else None)
        base.update(kw)
        return Result(**base)

    def _proven(self, proof):
        self._say(f"PROVEN IMPOSSIBLE: no colouring of K_{self.n} avoids ({self._ks_str()}), "
                  f"so R({self._ks_str()}) <= {self.n}.")
        self._say(f"  proof: {proof}")
        return self._result(route="proof", impossible=True, proof=proof, best_score=math.inf)

    def _accept(self, col, route):
        """Independent verification of a claimed colouring; returns a Result or None."""
        col = np.ascontiguousarray(col, dtype=np.int8)
        recount = count_monochromatic(col, self.ks)     # full recount: no symmetry, no incremental state
        if recount.sum() != 0:
            self._say(f"WARNING: {route} claimed a solution but a full recount finds violations {recount.tolist()}")
            return None
        budget = max(20.0, 0.25 * (now() - self.t0))
        v = verify(col, self.ks, time_limit=budget)     # separate pure-Python implementation
        if v["valid"] is False:
            self._say(f"WARNING: {route} claimed a solution that failed verification: {v}")
            return None
        if v["valid"]:
            self._say(f"SOLVED via {route}; independent check: clique numbers {v['clique_numbers']} < {self.ks}")
        else:
            self._say(f"SOLVED via {route}; a full recount in C confirms no forbidden clique. The slower "
                      f"pure-Python check did not finish in {budget:.0f}s: complete it with the verify command.")
        record = self._lit == "record"
        if record:
            self._say(f"*** This beats the best published lower bound known to this program: "
                      f"R({self._ks_str()}) >= {self.n + 1}. Check the current literature, then report it. ***")
        return self._result(solved=True, coloring=col, route=route, best_score=0.0, verification=v,
                            record=record, best_coloring=None)

    def _exact_step(self, seconds):
        """Give the complete full-space search a slice. Returns a final Result or None."""
        ex = self.exact
        if ex is None or ex.status != ExactSearch.RUNNING:
            return None
        st = ex.run(max(0.01, min(seconds, self._remaining())))
        if st == ExactSearch.FOUND:
            res = self._accept(ex.solution(), "exact")
            if res is not None:
                return res
            self.exact = None  # cannot happen if the search is correct; never trust it again
        elif st == ExactSearch.IMPOSSIBLE:
            return self._proven(ex.proof())
        return None

    def _sym_share(self, n_open):
        """Share of time for tier S: one share per open symmetric space against one per kind
        of local search (weak, descend, ascend, full), capped at sym_share.  With one open
        space the complete search gets a fifth of the time, with a dozen it gets the cap.
        (Measured: the wins that take longest come from tier S, local search wins early.)"""
        kinds = len({a.kind for a in self.arms})
        return min(self.cfg.sym_share, n_open / (n_open + kinds + 1e-9))

    def _pick_local(self, slice_s, forgive=False):
        """Bandit choice among the local-search arms; None if no arm may run now.

        Some steps cannot be interrupted (a single clique count on a large instance can take
        a while), so a pull can overrun its slice.  The excess is booked as debt and the arm
        sits out later turns until it is paid back, which keeps the time shares of the arms
        as the bandit intended.  Near the end, an arm whose latest pull took longer than the
        time left is not started.  forgive: nothing else could use the time, so debts are
        cancelled instead of waited out."""
        left = self._remaining()
        if forgive:
            for a in self.arms:
                a.debt = 0.0
        for _ in range(4 * len(self.arms)):
            i = self.bandit.select()
            arm = self.arms[i]
            if arm.debt > 0:
                arm.debt -= slice_s
            elif arm.last <= max(1.0, 4 * slice_s) or arm.last <= left:
                return i
            self.bandit.skip(i)
        return None

    # ---------------------------------------------------------------- main loop
    def run(self) -> Result:
        cfg, n = self.cfg, self.n
        self._lit, self._note = known.describe(self.ks, n)
        if cfg.known == "ignore":
            self._lit, self._note = "unknown", ""
        self._say(f"target: colour K_{n} avoiding ({self._ks_str()}), i.e. R({self._ks_str()}) > {n}; "
                  f"{len(self.sym_arms)} symmetric spaces, {len(self.arms)} local-search arms"
                  + (f", {cfg.jobs} workers" if cfg.jobs > 1 and self.worker == 0 else ""))
        if self._note:
            self._say(f"literature: {self._note}")
        # 0) proofs of impossibility that cost nothing
        ub = upper_bound(self.ks)
        if n >= ub:
            return self._proven(f"the Greenwood-Gleason recursion gives R({self._ks_str()}) <= {ub} <= {n}")
        if self._lit == "impossible" and cfg.known == "stop":
            self._say("stopping: this target is known to be impossible (not re-proven here; "
                      "pass known='warn' / --ignore-known to search anyway)")
            return self._result(route="literature", known_impossible=True, best_score=math.inf)
        if cfg.exact and n <= 64 and self.worker == 0:
            self.exact = ExactSearch(self.ks, n)
            if self.exact.status == ExactSearch.IMPOSSIBLE:
                return self._proven(self.exact.proof())
        if cfg.init is not None:
            starts = (load_colorings(cfg.init) if isinstance(cfg.init, (str, os.PathLike))
                      else [np.asarray(c) for c in (cfg.init if isinstance(cfg.init, (list, tuple)) else [cfg.init])])
            smaller = [c.astype(np.int8) for c in starts if c.shape == (n - 1, n - 1)]
            if smaller:                 # valid colourings on n-1 vertices: the ascend route extends them
                good = [c for c in smaller
                        if (lambda cnt: cnt is not None and cnt.sum() == 0)(
                            count_monochromatic(c, self.ks, time_limit=max(1.0, 0.1 * cfg.time)))]
                for c in good:
                    self._add_valid(n - 1, c, cap=max(24, len(good)))
                self._say(f"{len(good)} valid starting colouring(s) on {n - 1} vertices for the ascend route"
                          + (f" ({len(smaller) - len(good)} invalid ones ignored)" if len(good) < len(smaller) else ""))
            same = [c for c in starts if c.shape == (n, n)]
            col = same[0] if same else None
            if col is None and not smaller:
                self._say(f"starting colouring ignored: it has {starts[0].shape[0] if starts else 0} vertices, "
                          f"expected {n} or {n - 1}")
            elif col is not None:
                col = col.astype(np.int8)
                cnt = count_monochromatic(col, self.ks, time_limit=max(1.0, 0.1 * cfg.time))
                if cnt is None:
                    self._say("starting colouring ignored: too many forbidden cliques to count")
                elif cnt.min() >= 0:
                    score = float(np.dot(self.w, cnt))
                    self._say(f"starting colouring loaded: score {score:g}")
                    if score == 0:
                        res = self._accept(col, "init")
                        if res is not None:
                            return res
                    self._note_near(n, score, col)
        hb, next_hb = cfg.heartbeat, now() + cfg.heartbeat
        self.s_time = self.l_time = 0.0
        pulls = 0
        while self._remaining() > 0:
            for a in self.sym_arms:               # spaces another worker proved empty
                if not a.dead and a.name in self.shared.dead:
                    a.dead = True
            live_sym = [a for a in self.sym_arms if not a.dead]
            exact_live = self.exact is not None and self.exact.status == ExactSearch.RUNNING
            if not live_sym and not self.arms and not exact_live:
                self._say("every search space exhausted without a colouring (not a proof of impossibility)")
                break
            i = None
            do_sym = live_sym and (not self.arms or
                                   self.s_time <= self._sym_share(len(live_sym)) * (self.s_time + self.l_time + 1e-9))
            if not do_sym and self.arms:
                rnd = min(pulls // len(self.arms), 12)   # doubling schedule, capped
                slice_s = min(cfg.slice, cfg.min_slice * 2 ** rnd, max(0.01, self._remaining()))
                i = self._pick_local(slice_s, forgive=not live_sym and not exact_live)
                if i is None and not live_sym and not exact_live:
                    break                       # no local arm could finish in the time left
            if live_sym and (do_sym or i is None):
                # least-served first, with time in proportion to weight / (number of orbits):
                # small spaces are cheap to decide, so they stay ahead of large ones
                arm = min(live_sym, key=lambda a: (a.spent * a.cost, a.cost))
                # each pull adds a quarter to the arm's time: the arms stay close to their shares
                slice_s = min(cfg.slice, max(cfg.min_slice, 0.25 * arm.spent), max(0.01, self._remaining()))
                t_pull = now()
                st, col = self._pull_sym(arm, slice_s)
                self.s_time += now() - t_pull
                if st == OrbitSearch.FOUND:
                    res = self._accept(col, f"{arm.name} (complete search)")
                    if res is not None:
                        return res
                    arm.dead = True
            elif i is not None:
                arm = self.arms[i]
                pulls += 1
                t_pull = now()
                best, col, ok = self.pull(arm, slice_s)
                spent = now() - t_pull
                self.l_time += spent
                arm.last, arm.debt = spent, max(0.0, spent - 1.5 * slice_s - 0.05)
                self.bandit.update(i, self._reward(best))
                if best < self.best_overall:
                    self.best_overall = best
                    self._say(f"{arm.name:>14}: best score {best:g}")
                if ok and col is not None and col.shape[0] == n:
                    res = self._accept(col, arm.name)
                    if res is not None:
                        return res
                if exact_live:  # small n: as much time as the pull; otherwise one share
                    res = self._exact_step(spent * (1.0 if n <= 24 else 1.0 / len(self.arms)))
                    if res is not None:
                        return res
            elif exact_live:
                res = self._exact_step(min(cfg.slice, 1.0))
                if res is not None:
                    return res
            if now() >= next_hb and self.worker == 0:
                dead = sum(a.dead for a in self.sym_arms)
                msg = (f"still searching: best score {self.best_overall:g}; symmetric spaces: {dead} proven "
                       f"empty, {len(self.sym_arms) - dead} open; {pulls} local-search pulls")
                if self.exact is not None and self.exact.status == ExactSearch.RUNNING:
                    msg += f"; complete search {self.exact.progress()}"
                self._say(msg)
                hb = min(120.0, 2 * hb)
                next_hb = now() + hb
        return self._result()

    def _pull_sym(self, arm, slice_s):
        if arm.runs is None:                        # drop a space that duplicates an earlier one
            sp = arm.spec.space()
            sig = (sp.V, _signature(sp))
            if sig in self._sigs:
                arm.dead = True
                return OrbitSearch.RUNNING, None
            self._sigs.add(sig)
        st, col = arm.pull(self.ks, slice_s, int(self.rng.integers(1 << 62)), deterministic=self.worker == 0)
        if st == OrbitSearch.EXHAUSTED:
            self.shared.dead[arm.name] = arm.spec.space().V
            if arm.spent >= 0.5:        # quick ones are only counted (see the summary line)
                self._say(f"{arm.name:>14}: no colouring with this symmetry ({arm.spec.space().V} orbits, "
                          f"{arm.nodes:,} nodes, {arm.spent:.1f}s)")
        if not self._sym_summary and all(a.dead or a.runs is not None for a in self.sym_arms):
            self._sym_summary = True    # every space has had its first look
            dead = [a for a in self.sym_arms if a.dead]
            if self.worker == 0 and self.sym_arms:
                self._say(f"symmetric spaces: {len(dead)} of {len(self.sym_arms)} proven empty at first look"
                          + (f" (e.g. {', '.join(a.name for a in dead[:4])})" if dead else "")
                          + f"; {len(self.sym_arms) - len(dead)} still open")
        return st, col

    def bandit_summary(self):
        means = self.bandit.means()
        out = {a.name: (int(self.bandit.n[i]), None if np.isnan(means[i]) else round(float(means[i]), 3))
               for i, a in enumerate(self.arms)}
        out["symmetric spaces"] = {"proven empty": sum(a.dead for a in self.sym_arms),
                                   "open": sum(not a.dead for a in self.sym_arms)}
        return out


def solve(ks, n, valid_below=None, **kw) -> Result:
    """Search for a colouring of K_n with no K_{ks[c]} in colour c.

    valid_below: optional valid colouring on n-1 vertices (seeds the 'ascend' route)."""
    cfg = Config(**kw)
    if cfg.jobs == 0:
        cfg = replace(cfg, jobs=os.cpu_count() or 1)
    if cfg.jobs <= 1:
        rz = RamseyZero(ks, n, cfg)
        if valid_below is not None:
            rz.valid[n - 1] = [np.asarray(valid_below, dtype=np.int8)]
        return rz.run()
    shared = Shared()
    results, errors = [None] * cfg.jobs, []
    t0 = now()

    def work(i):
        try:
            c = replace(cfg, seed=cfg.seed + 100003 * i)
            rz = RamseyZero(ks, n, c, worker=i, shared=shared)
            rz.t0 = t0
            if valid_below is not None:
                rz.valid[n - 1] = [np.asarray(valid_below, dtype=np.int8)]
            results[i] = rz.run()
        except BaseException as e:      # surface a worker's failure instead of hiding it
            errors.append(e)
        finally:
            r = results[i]
            if r is None or r.solved or r.impossible or r.known_impossible:
                shared.stop.set()

    threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(cfg.jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    done = [r for r in results if r is not None]
    for pick in (lambda r: r.solved, lambda r: r.impossible, lambda r: r.known_impossible):
        hit = [r for r in done if pick(r)]
        if hit:
            return hit[0]
    return min(done, key=lambda r: r.best_score)


def climb(ks, n_from, n_to, time_per_n=60.0, **kw):
    """Curriculum: solve n_from, n_from+1, ... feeding each colouring to the next size.
    Returns {n: Result} for every n tried; stops at the first n that is not solved (its
    Result says whether it was proven or known to be impossible)."""
    results = {}
    prev = None
    for n in range(n_from, n_to + 1):
        res = solve(ks, n, valid_below=prev, time=time_per_n, **kw)
        results[n] = res
        if not res.solved:
            break
        prev = res.coloring
    return results
