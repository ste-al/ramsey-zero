"""What the literature says about a target -- used for REPORTING ONLY, never for searching.

The search itself stays zero-knowledge.  This table lets the program tell you three things
the search cannot know by itself:

  * a target is already known to be impossible (so it does not search forever),
  * a colouring it found matches the best published lower bound, or
  * a colouring it found would BEAT the best published lower bound (worth double-checking
    against the current literature and reporting).

State of 18 November 2026.  Sources:
  S. Radziszowski, "Small Ramsey Numbers", Electronic J. Combinatorics, Dynamic Survey DS1,
    revision 18 (24 April 2026) -- the base for everything not listed below;
  A. Nagda, P. Raghavan, A. Thakurta, arXiv:2603.09172 (AlphaEvolve, March-April 2026):
    R(3,13) >= 61, R(3,18) >= 100, R(4,13..16), R(4,18..20);
  Y. Wang, ScaleAutoResearch-Ramsey (May 2026): R(3,17) >= 93, R(4,15) >= 160;
  witness file from Google's AlphaEvolve repository, verified by the ramsey-witnesses
    project (Nov 2026): R(4,15) >= 163, R(4,21) >= 244;
  C. Yu, arXiv:2608.18169 (Aug 2026): R(4,20) >= 252 (hence R(4,21) >= 252);
  S. Coniglio et al., arXiv:2608.18769 (Aug 2026): R(3,n) for 24 <= n <= 49 (circulants);
  F. Cremer, arXiv:2610.12122 (Oct 2026): R(6,8) >= 135, R(8,10) >= 345;
  8rulerstar/ramsey-lower-bounds (Nov 2026): R(7,12) >= 420;
  L. Dauth, ramsey-neighbourhood-counting (Oct 2026): R(3,3,k) for k = 9, 11..15;
  L. Boza, arXiv:2603.10851 (March 2026): R(4,4,4) <= 229, R(3,4,5) <= 157, R(3,3,6) <= 91;
  V. Angeltveit, Electron. J. Combin. 32(4) P4.30 (2025): R(3,10) <= 41.
Unrefereed upper bounds (for example Dauth's flag-algebra bounds R(3,4,4) <= 73,
R(4,4,4) <= 198) are NOT used: an upper bound here makes the program refuse to search.
Where sources disagree, the larger lower bound is used, so the program never calls a
colouring a new record when someone already has a better one.  A missing entry means "no
data here", not "nothing is known".

LOWER[ks] = L  means R(ks) >= L: colourings on L-1 vertices are known.
UPPER[ks] = U  means R(ks) <= U: no colouring on U or more vertices exists (proven).
Keys are sorted tuples of the forbidden clique sizes.
"""
from __future__ import annotations

SURVEY = "Radziszowski's survey DS1.18 (April 2026) and results posted up to 18 November 2026"

# exact values: lower == upper
_EXACT = {
    (3, 3): 6, (3, 4): 9, (3, 5): 14, (3, 6): 18, (3, 7): 23, (3, 8): 28, (3, 9): 36,
    (4, 4): 18, (4, 5): 25,
    (3, 3, 3): 17, (3, 3, 4): 30,
}

LOWER = {
    # R(3,k)
    (3, 10): 40, (3, 11): 47, (3, 12): 53, (3, 13): 61, (3, 14): 67, (3, 15): 74,
    (3, 16): 82, (3, 17): 93, (3, 18): 100, (3, 19): 106, (3, 20): 111, (3, 21): 122,
    (3, 22): 131, (3, 23): 139,
    # R(4,k)
    (4, 6): 36, (4, 7): 49, (4, 8): 59, (4, 9): 73, (4, 10): 92, (4, 11): 102, (4, 12): 128,
    (4, 13): 139, (4, 14): 148, (4, 15): 163, (4, 16): 174, (4, 17): 200, (4, 18): 209,
    (4, 19): 219, (4, 20): 252, (4, 21): 252, (4, 22): 314,
    # R(5,k)
    (5, 5): 43, (5, 6): 59, (5, 7): 80, (5, 8): 101, (5, 9): 133, (5, 10): 149, (5, 11): 183,
    (5, 12): 203, (5, 13): 233, (5, 14): 267, (5, 15): 275, (5, 17): 388, (5, 18): 396,
    (5, 19): 411, (5, 20): 424,
    # R(6,k), R(7,k), R(8,k)
    (6, 6): 102, (6, 7): 115, (6, 8): 135, (6, 9): 183, (6, 10): 204, (6, 11): 262, (6, 12): 294,
    (6, 13): 347, (6, 14): 401,
    (7, 7): 205, (7, 8): 219, (7, 9): 252, (7, 10): 292, (7, 11): 405, (7, 12): 420, (7, 13): 513,
    (8, 8): 282, (8, 9): 329, (8, 10): 345, (8, 11): 457,
    # three colours
    (3, 3, 5): 45, (3, 3, 6): 61, (3, 3, 7): 85, (3, 3, 8): 103, (3, 3, 9): 133, (3, 3, 10): 150,
    (3, 3, 11): 183, (3, 3, 12): 203, (3, 3, 13): 232, (3, 3, 14): 267, (3, 3, 15): 278,
    (3, 3, 16): 291,
    (3, 4, 4): 55, (3, 4, 5): 89, (3, 4, 6): 117, (3, 4, 7): 152, (3, 4, 8): 193, (3, 4, 9): 242,
    (3, 5, 5): 139, (3, 5, 6): 181, (3, 5, 7): 241, (3, 6, 6): 338,
    (4, 4, 4): 128, (5, 5, 5): 454,
    # four and five colours
    (3, 3, 3, 3): 51, (3, 3, 3, 4): 97, (3, 3, 3, 5): 162, (3, 3, 4, 4): 174, (3, 4, 4, 4): 381,
    (3, 3, 3, 10): 513, (4, 4, 4, 4): 634,
    (3, 3, 3, 3, 3): 162,
}
# R(3,n) >= value for 24 <= n <= 49 (circulant certificates, arXiv:2608.18769; n = 27 from DS1.18)
LOWER.update({(3, n): v for n, v in {
    24: 151, 25: 160, 26: 163, 27: 172, 28: 181, 29: 194, 30: 199, 31: 210, 32: 219, 33: 230,
    34: 239, 35: 253, 36: 261, 37: 272, 38: 281, 39: 294, 40: 305, 41: 317, 42: 327, 43: 340,
    44: 348, 45: 364, 46: 371, 47: 385, 48: 391, 49: 411}.items()})

UPPER = {
    (3, 10): 41, (3, 11): 50, (3, 12): 59, (3, 13): 68, (3, 14): 77, (3, 15): 87, (3, 16): 97,
    (3, 17): 109, (3, 18): 120, (3, 19): 132, (3, 20): 145, (3, 21): 157, (3, 22): 171,
    (3, 23): 185,
    (4, 6): 40, (4, 7): 58, (4, 8): 79, (4, 9): 105, (4, 10): 135, (4, 11): 170, (4, 12): 210,
    (4, 13): 256, (4, 14): 307, (4, 15): 364,
    (5, 5): 46, (5, 6): 85, (5, 7): 133, (5, 8): 193, (5, 9): 282, (5, 10): 381,
    (6, 6): 160, (6, 7): 270, (6, 8): 423, (7, 7): 492,
    (3, 3, 5): 57, (3, 3, 6): 91, (3, 4, 4): 77, (3, 4, 5): 157, (4, 4, 4): 229,
    (3, 3, 3, 3): 62, (3, 3, 3, 4): 149, (3, 3, 4, 4): 450, (3, 3, 3, 3, 3): 307,
}
LOWER.update(_EXACT)
UPPER.update(_EXACT)

# Circulant Ramsey numbers RC(k,l): a circulant (cyclic) colouring exists on RC-1 vertices and
# on no larger order.  From Harborth & Krause (2003) and arXiv:2608.18769.  Used by the
# test-suite to validate the complete orbit search, never by the search itself.
CIRCULANT_RAMSEY = {
    (3, 3): 6, (3, 4): 9, (3, 5): 14, (3, 6): 17, (3, 7): 22, (3, 8): 27, (3, 9): 36, (3, 10): 39,
    (3, 11): 46, (3, 12): 49, (3, 13): 58, (3, 14): 64, (3, 15): 73, (3, 16): 79, (3, 17): 92,
    (3, 18): 98, (3, 19): 106, (3, 20): 111,
    (4, 4): 18, (4, 5): 25, (4, 6): 34, (4, 7): 47, (4, 8): 52, (4, 9): 69,
    (5, 5): 42, (5, 6): 57,
}


def _key(ks):
    return tuple(sorted(int(k) for k in ks))


def lower_bound(ks):
    """Best published lower bound L (R >= L), or None if this table has no entry."""
    return LOWER.get(_key(ks))


def upper_bound_known(ks):
    """Best published upper bound U (R <= U), or None."""
    return UPPER.get(_key(ks))


def describe(ks, n):
    """Classify the target 'colour K_n avoiding ks' against the literature.

    Returns (status, text):
      'impossible'  published results rule it out (n >= upper bound)
      'known'       colourings of this size are known (n < lower bound)
      'record'      would improve the best published lower bound (lower bound <= n < upper bound)
      'unknown'     no data in this table
    """
    key = _key(ks)
    name = "R(" + ",".join(map(str, ks)) + ")"
    lo, up = LOWER.get(key), UPPER.get(key)
    if up is not None and n >= up:
        exact = f"{name} = {up}" if lo == up else f"{name} <= {up}"
        return "impossible", f"published results give {exact}, so no colouring of K_{n} exists"
    if lo is not None and n < lo:
        tail = f"{name} = {lo}" if lo == up else f"{name} >= {lo}"
        extra = " (this size matches the record)" if n == lo - 1 and lo != up else ""
        return "known", f"colourings of this size are known: {tail}{extra}"
    if lo is not None:
        rng = f"{lo} <= {name}" + (f" <= {up}" if up is not None else "")
        return "record", (f"OPEN: the best published bounds are {rng}; a colouring of K_{n} would prove "
                          f"{name} >= {n + 1}, a new lower bound")
    return "unknown", f"no published bound for {name} in this program's table"
