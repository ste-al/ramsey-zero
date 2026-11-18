"""Independent exact verification (pure Python, shares no code with the C search core).

For each colour c we compute the clique number of the colour-c graph with a
branch-and-bound maximum-clique algorithm (greedy-colouring bound, Tomita MCQ
style) on Python big-int bitsets.  A colouring certifies R(k_0..k_{r-1}) > n
iff omega(colour c) < k_c for every c.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np


def _bits(row_mask):
    while row_mask:
        low = row_mask & -row_mask
        yield low.bit_length() - 1
        row_mask ^= low


class VerificationTimeout(Exception):
    pass


def _degeneracy_order(nbr, n):
    """Vertices in reverse smallest-last order (densest core first): tightens the colouring bound."""
    deg = [bin(x).count("1") for x in nbr]
    alive, seq = (1 << n) - 1, []
    for _ in range(n):
        v = min((u for u in range(n) if alive >> u & 1), key=deg.__getitem__)
        seq.append(v)
        alive &= ~(1 << v)
        m = nbr[v] & alive
        while m:
            low = m & -m
            deg[low.bit_length() - 1] -= 1
            m ^= low
    return seq[::-1]


def _max_clique(nbr, n, stop_at=None, deadline=None):
    """Return a maximum clique (list of vertices). Stops early once a clique of size stop_at
    exists.  Raises VerificationTimeout if `deadline` (time.time() value) passes."""
    perm = _degeneracy_order(nbr, n)
    pos = [0] * n
    for i, v in enumerate(perm):
        pos[v] = i
    nb = []
    for v in perm:                       # relabelled adjacency
        m, x = nbr[v], 0
        while m:
            low = m & -m
            x |= 1 << pos[low.bit_length() - 1]
            m ^= low
        nb.append(x)
    best = []
    calls = [0]

    def colour_sort(P):
        # greedy colouring of P: returns vertices in order with their colour bound
        order, bounds = [], []
        uncoloured, colour = P, 0
        while uncoloured:
            colour += 1
            Q = uncoloured
            while Q:
                low = Q & -Q
                v = low.bit_length() - 1
                Q &= ~low
                Q &= ~nb[v]
                uncoloured &= ~low
                order.append(v)
                bounds.append(colour)
        return order, bounds

    def expand(R, P):
        nonlocal best
        calls[0] += 1
        if deadline is not None and calls[0] % 2000 == 0 and time.time() > deadline:
            raise VerificationTimeout()
        order, bounds = colour_sort(P)
        for i in range(len(order) - 1, -1, -1):
            if len(R) + bounds[i] <= len(best):
                return
            v = order[i]
            R2 = R + [v]
            P2 = P & nb[v]
            if P2:
                expand(R2, P2)
            elif len(R2) > len(best):
                best = R2
            if stop_at is not None and len(best) >= stop_at:
                return
            P &= ~(1 << v)

    expand([], (1 << n) - 1)
    return [perm[i] for i in best]


def clique_number(col, c, stop_at=None, deadline=None):
    col = np.asarray(col)
    n = col.shape[0]
    nbr = []
    for u in range(n):
        m = 0
        for v in np.nonzero(col[u] == c)[0].tolist():
            if v != u:
                m |= 1 << v
        nbr.append(m)
    return _max_clique(nbr, n, stop_at, deadline)


def verify(col, ks, verbose=False, time_limit=None):
    """Exact certificate check. Returns a dict with per-colour clique numbers and 'valid'.

    time_limit (seconds): give up after that long; 'valid' is then None and 'complete' False
    (the colours checked so far are reported).  Without a limit the check always completes."""
    col = np.asarray(col)
    n = col.shape[0]
    assert col.shape == (n, n)
    off = ~np.eye(n, dtype=bool)
    if not (col == col.T).all():
        return {"valid": False, "complete": True, "reason": "matrix not symmetric"}
    vals = col[off]
    if vals.size and (vals.min() < 0 or vals.max() >= len(ks)):
        return {"valid": False, "complete": True, "reason": "colour out of range"}
    t0 = time.time()
    deadline = None if time_limit is None else t0 + time_limit
    omegas, witnesses, complete = [], [], True
    # sparse colours first: they are quick and most likely to expose a mistake
    order = sorted(range(len(ks)), key=lambda c: int((col == c).sum()))
    done = {}
    for c in order:
        try:
            done[c] = clique_number(col, c, stop_at=ks[c], deadline=deadline)  # stops at a forbidden clique
        except VerificationTimeout:
            complete = False
            break
        if len(done[c]) >= ks[c]:
            break
    for c in range(len(ks)):
        omegas.append(len(done[c]) if c in done else None)
        witnesses.append(sorted(done[c]) if c in done else None)
    bad = any(w is not None and w >= k for w, k in zip(omegas, ks))
    valid = False if bad else (True if complete and all(w is not None for w in omegas) else None)
    res = {"valid": valid, "complete": valid is not None, "n": n, "ks": list(ks), "clique_numbers": omegas,
           "witness_cliques": witnesses, "seconds": round(time.time() - t0, 3)}
    if verbose:
        print(json.dumps(res))
    return res


def to_graph6(adj):
    """graph6 string of a 0/1 adjacency matrix (readable by nauty, networkx, SageMath)."""
    adj = np.asarray(adj)
    n = adj.shape[0]
    if n <= 62:
        out = [chr(n + 63)]
    else:
        out = [chr(126)] + [chr(((n >> s) & 63) + 63) for s in (12, 6, 0)]
    bits = [int(adj[i, j] != 0) for j in range(1, n) for i in range(j)]
    bits += [0] * (-len(bits) % 6)
    for t in range(0, len(bits), 6):
        v = 0
        for b in bits[t:t + 6]:
            v = (v << 1) | b
        out.append(chr(v + 63))
    return "".join(out)


def from_graph6(line):
    """0/1 adjacency matrix from a graph6 string (the format of McKay's and Exoo's data)."""
    data = [ord(c) - 63 for c in line.strip()]
    if not data or min(data) < 0 or max(data) > 63:
        raise ValueError("not a graph6 string")
    if data[0] < 63:
        n, pos = data[0], 1
    elif len(data) > 3 and data[1] < 63:
        n, pos = (data[1] << 12) | (data[2] << 6) | data[3], 4
    else:
        raise ValueError("graph6 with more than 258047 vertices is not supported")
    bits = []
    for v in data[pos:]:
        bits.extend((v >> s) & 1 for s in (5, 4, 3, 2, 1, 0))
    need = n * (n - 1) // 2
    if len(bits) < need:
        raise ValueError("graph6 string too short")
    adj = np.zeros((n, n), dtype=np.int8)
    t = 0
    for j in range(1, n):
        for i in range(j):
            if bits[t]:
                adj[i, j] = adj[j, i] = 1
            t += 1
    return adj


def coloring_from_graph(adj):
    """Two-colour colouring of a graph: edges get colour 0, non-edges colour 1.  So a
    (k,l)-graph (no K_k, no independent l-set) is a colouring for ks = (k, l)."""
    adj = np.asarray(adj)
    col = np.where(adj != 0, 0, 1).astype(np.int8)
    np.fill_diagonal(col, -1)
    return col


def graph_of(col, colour=0):
    """0/1 adjacency of the colour class (2 colours: G = colour 0, so omega(G)<k0, alpha(G)<k1)."""
    col = np.asarray(col)
    A = (col == colour).astype(np.int8)
    np.fill_diagonal(A, 0)
    return A


def save_coloring(path, col, ks, meta=None):
    """Write a plain-text certificate: one row per vertex, colour digits; JSON header comment."""
    col = np.asarray(col)
    n = col.shape[0]
    header = {"n": n, "ks": list(ks), "format": "row i, column j = colour of edge ij ('-' on diagonal)"}
    if meta:
        header.update(meta)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("# " + json.dumps(header) + "\n")
        for i in range(n):
            f.write("".join("-" if j == i else str(int(col[i, j])) for j in range(n)) + "\n")


def load_colorings(path):
    """All colourings in a file: a certificate written by save_coloring (one colouring), or
    a graph6 file with one graph per line (two colours: edges = colour 0)."""
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if lines and (str(path).lower().endswith(".g6") or not any(ch in "-." for ch in lines[0])
                  and any(ch not in "0123456789" for ch in lines[0])):
        return [coloring_from_graph(from_graph6(ln)) for ln in lines if not ln.startswith(">>")]
    return [load_coloring(path)[0]]


def load_coloring(path):
    """(colouring, header) from a certificate; a graph6 file gives its first graph."""
    if str(path).lower().endswith(".g6"):
        return load_colorings(path)[0], {"format": "graph6"}
    rows, header = [], {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                try:
                    header = json.loads(line[1:].strip())
                except json.JSONDecodeError:
                    pass
                continue
            rows.append(line)
    n = len(rows)
    col = np.full((n, n), -1, dtype=np.int8)
    for i, row in enumerate(rows):
        for j, ch in enumerate(row):
            if ch not in "-.":
                col[i, j] = int(ch)
    return col, header
