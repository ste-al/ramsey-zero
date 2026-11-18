"""Proving that a colouring CANNOT exist, and complete search in general.

Local search can only ever find colourings.  The tools here can also prove that none
exists, and all of them are rigorous.

upper_bound(ks)
    The Greenwood-Gleason recursion (1955), a theorem:
        R(k_1..k_r) <= 2 - r + sum_i R(.., k_i - 1, ..)
    and one less when that sum is even and some term is even: every vertex would need
    exactly U_i - 1 neighbours in each colour i, and a graph with an odd number of
    vertices cannot be regular of odd degree.  Base cases: R(k) = k for one colour, and
    a colour with k = 2 can never be used.  The bounds of the terms may themselves be
    upper bounds; the argument still holds.  The recursion is exact for R(3,3)=6,
    R(3,4)=9, R(3,5)=14, R(4,4)=18 and R(3,3,3)=17, and only an upper bound elsewhere.

ExactSearch
    Complete search of the FULL space (rx_* in _core.c): fixed edge order, clique checks,
    lex-leader symmetry breaking over vertex and colour relabellings, degree bounds.  Best
    at proving impossibility; practical only for small n.

OrbitSearch
    Complete search over the orbit variables of ANY space (ox_* in _core.c): forward
    checking, smallest-domain-first, optional conflict-driven ordering and restarts.  It
    decides symmetric spaces (finds a colouring or proves there is none with that
    symmetry) and one-vertex extensions, and is a strong randomised search as well.
"""
from __future__ import annotations

import ctypes as C
from functools import lru_cache

import numpy as np

from .core import LIB, _ptr


@lru_cache(maxsize=None)
def _ub(ks: tuple) -> int:
    if any(k <= 1 for k in ks):
        return 1
    ks = tuple(k for k in ks if k != 2)   # a colour that may not contain an edge is unusable
    if not ks:
        return 2                          # K_2's edge needs a colour
    if len(ks) == 1:
        return ks[0]
    r = len(ks)
    terms = [_ub(tuple(sorted(ks[:i] + (ks[i] - 1,) + ks[i + 1:]))) for i in range(r)]
    total = 2 - r + sum(terms)
    if total % 2 == 0 and any(t % 2 == 0 for t in terms):
        total -= 1
    return total


def upper_bound(ks) -> int:
    """A proven upper bound on R(ks): no valid colouring of K_n exists for n >= this."""
    return _ub(tuple(sorted(int(k) for k in ks)))


def degree_bounds(ks):
    """dmax[c]: most colour-c edges any vertex can have (its colour-c neighbourhood is a
    valid colouring for ks with k_c lowered by one)."""
    ks = list(ks)
    return [upper_bound(ks[:c] + [ks[c] - 1] + ks[c + 1:]) - 1 for c in range(len(ks))]


class ExactSearch:
    """Resumable complete search for a valid colouring of K_n (n <= 64)."""

    RUNNING, FOUND, IMPOSSIBLE = 0, 1, 2

    def __init__(self, ks, n, flags=7):
        if n > 64:
            raise ValueError("exact search supports n <= 64")
        self.ks, self.n, self.r = [int(k) for k in ks], int(n), len(ks)
        self.dmax = degree_bounds(self.ks)
        self._k = np.ascontiguousarray(self.ks, dtype=np.int32)
        self._d = np.ascontiguousarray(self.dmax, dtype=np.int32)
        self.h = LIB.rx_create(self.n, self.r, _ptr(self._k, C.c_int), _ptr(self._d, C.c_int), int(flags))
        if not self.h:
            raise ValueError("rx_create failed")
        self.stats = np.zeros(6)
        self.status = self.run(0.0, max_nodes=1)

    def __del__(self):
        h = getattr(self, "h", None)
        if h:
            LIB.rx_free(h)
            self.h = None

    def run(self, time_limit=None, max_nodes=0) -> int:
        """Continue the search. Returns RUNNING (budget used up), FOUND or IMPOSSIBLE."""
        tl = -1.0 if time_limit is None else max(0.0, float(time_limit))
        self.status = LIB.rx_run(self.h, int(max_nodes), tl, _ptr(self.stats, C.c_double))
        return self.status

    def solution(self) -> np.ndarray:
        out = np.empty((self.n, self.n), dtype=np.int8)
        LIB.rx_get(self.h, _ptr(out, C.c_int8))
        return out

    @property
    def nodes(self) -> int:
        return int(self.stats[0])

    def progress(self) -> str:
        return (f"{self.nodes:,} nodes, edge {int(self.stats[1])}/{int(self.stats[3])} "
                f"(deepest {int(self.stats[2])})")

    def proof(self) -> str:
        reason = int(self.stats[5])
        ks = ",".join(map(str, self.ks))
        if reason == 1:
            return (f"degree count: each vertex of K_{self.n} has {self.n - 1} edges, but at most "
                    f"{' + '.join(map(str, self.dmax))} = {sum(self.dmax)} can be coloured "
                    f"(colour-c neighbourhoods are valid colourings for ({ks}) with k_c lowered)")
        if reason == 2:
            return ("degree parity: some colour class would have to be regular of odd degree "
                    f"on {self.n} (odd) vertices")
        return (f"exhaustive search over all colourings of K_{self.n} up to symmetry "
                f"({self.nodes:,} nodes): none avoids ({ks})")


class OrbitSearch:
    """Complete search over the orbit variables of any Space (ox_* in _core.c).

    FOUND      a valid colouring of the space (get it with solution())
    EXHAUSTED  the space provably contains no valid colouring.  For the full space that is a
               proof of impossibility; for a symmetric or extension space it only rules out
               colourings with that symmetry / extending that base graph.
    Restarts (with a new random order) make it a good randomised search as well; a run that
    is not interrupted by a restart is complete.

    values: order in which colours are tried.  "small" (default) = smallest forbidden clique
    first, "large" = largest first, None = colour order (shuffled when randomise is set).
    The order never affects completeness, only how soon a colouring turns up.
    """

    RUNNING, FOUND, EXHAUSTED = 0, 1, 2

    def __init__(self, space, ks, degree_caps=True, randomise=False, seed=0, phase=False, activity=False,
                 values="small"):
        self.space, self.ks, self.r, self.n = space, [int(k) for k in ks], len(ks), space.n
        self._k = np.ascontiguousarray(self.ks, dtype=np.int32)
        dm = degree_bounds(self.ks) if degree_caps else [self.n] * self.r
        self._d = np.ascontiguousarray(dm, dtype=np.int32)
        base = None
        if space.base is not None:
            self._base = np.ascontiguousarray(space.base, dtype=np.int8)
            base = _ptr(self._base, C.c_int8)
        self.h = LIB.ox_create(self.n, self.r, _ptr(self._k, C.c_int), space.V,
                               _ptr(space.optr, C.c_int), _ptr(space.oedges, C.c_int), base,
                               _ptr(self._d, C.c_int),
                               (1 if randomise else 0) | (2 if phase else 0) | (4 if activity else 0)
                               | {None: 0, "small": 8, "large": 16}[values],
                               C.c_ulonglong(seed & 0xFFFFFFFFFFFFFFFF))
        if not self.h:
            raise ValueError("ox_create failed (frozen edges need a base colouring)")
        self.stats = np.zeros(6)
        self.status = self.run(0.0, max_nodes=1) if space.V else self.run(None)
        self.restarts = 0

    def __del__(self):
        h = getattr(self, "h", None)
        if h:
            LIB.ox_free(h)
            self.h = None

    def run(self, time_limit=None, max_nodes=0) -> int:
        tl = -1.0 if time_limit is None else max(0.0, float(time_limit))
        self.status = LIB.ox_run(self.h, int(max_nodes), tl, _ptr(self.stats, C.c_double))
        return self.status

    def restart(self, seed):
        LIB.ox_restart(self.h, C.c_ulonglong(seed & 0xFFFFFFFFFFFFFFFF))
        self.restarts += 1
        self.status = self.RUNNING if self.status != self.EXHAUSTED else self.EXHAUSTED

    def solution(self) -> np.ndarray:
        out = np.empty((self.n, self.n), dtype=np.int8)
        LIB.ox_get(self.h, _ptr(out, C.c_int8))
        return out

    @property
    def nodes(self) -> int:
        return int(self.stats[0])

    @property
    def run_nodes(self) -> int:
        return int(self.stats[1])

    def progress(self) -> str:
        return f"{self.nodes:,} nodes, depth {int(self.stats[2])}/{int(self.stats[4])} (deepest {int(self.stats[3])})"
