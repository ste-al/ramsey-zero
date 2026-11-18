# RamseyZero

A zero-knowledge search system for Ramsey colourings. Give it forbidden clique sizes and a
number of vertices, for example `(5,5)` and `n = 42`, and it looks for an `n x n` colour
matrix with no forbidden monochromatic clique. Such a matrix proves `R(5,5) > 42`.

It starts with no stored graphs and no hand-picked constructions. Every colouring it reports
is recounted in full and then re-checked by an independent exact verifier. When a target is
impossible it says so, with a proof where it can find one.

```bash
git clone https://github.com/ste-al/ramsey-zero && cd ramsey-zero
pip install numpy                                  # plus a C compiler, see below

python -m ramsey_zero solve 5 5 --n 42             # R(5,5) >= 43, about 1 s
python -m ramsey_zero solve 3 3 3 3 --n 50         # four colours: R(3,3,3,3) >= 51
python -m ramsey_zero solve 6 6 --n 101            # R(6,6) >= 102
python -m ramsey_zero solve 3 20 --n 110 --time 300 --jobs 4
python -m ramsey_zero climb 3 3 --from 3 --to 10   # finds K_3..K_5, proves K_6 impossible: R(3,3) = 6
python -m ramsey_zero climb 4 6 --from 30 --to 36 --time-per-n 600 --jobs 0   # 0 = all cores
python -m ramsey_zero verify certificates/R5-5_n42.txt
python -m ramsey_zero compare results/R4-6_n35.txt r46_35some.g6   # is my graph a known one?
python tests/test_core.py && python bench.py
```

Requirements: Python 3.9+, NumPy and a C compiler (gcc or clang). The C core compiles itself
on first import. On Windows use a 64-bit MinGW-w64 gcc (WinLibs or MSYS2 UCRT64) on PATH, or
set `CC` to its full path; the DLL it builds needs only KERNEL32 and the C runtime.

Layout: `ramsey_zero/` the package, `tests/` the test suite, `certificates/` colourings found
while testing (each re-verified), `benchmark/` the reference benchmark. Your own runs write
to `results/`, which git ignores. MIT licence, see `LICENSE`.

## What it finds

All times are wall-clock seconds on one core of a 2-core cloud VM, starting from nothing.

**Benchmark** (3 seeds each, all 48 runs solved; `python bench.py`, reference output in
`benchmark/`). It covers every exact value up to R(3,9), R(4,5) and R(3,3,4), and five
current records.

| certifies | vertices | median s | max s | found by |
|---|---|---|---|---|
| R(3,3) >= 6 ... R(3,7) >= 23 (five targets) | 5 to 22 | 0.00 to 0.01 | 0.02 | circulant, weak symmetry |
| R(3,8) >= 28 | 27 | 0.83 | 0.91 | local search (ascend, full, weak) |
| R(3,9) >= 36 | 35 | 0.07 | 0.07 | circulant with multiplier 11 |
| R(4,4) >= 18, R(4,5) >= 25 | 17, 24 | 0.00 | 0.1 | circulant, weak symmetry |
| R(3,3,3) >= 17, R(3,3,4) >= 30 | 16, 29 | 0.00, 0.07 | 0.07 | GF(16) cubic classes, circulant |
| R(5,5) >= 43 (record) | 42 | 1.5 | 1.5 | circulant on 43, delete a vertex, repair |
| R(4,7) >= 49 (record) | 48 | 3.5 | 5.6 | circulant near-solution, repair |
| R(6,6) >= 102 (record) | 101 | 0.15 | 0.16 | circulant with multiplier 4 (Paley-like) |
| R(3,3,3,3) >= 51 (record) | 50 | 3.9 | 23.3 | 3 blocks of Z4 x Z4 plus 2 fixed points |
| R(4,4,4) >= 128 (record) | 127 | 0.16 | 0.16 | circulant with multiplier 5 |

**More published record sizes matched** (one run each, 120 s limit). The certificates are
in `certificates/`, each re-checked by the independent verifier.

| certifies | vertices | seconds | found in |
|---|---|---|---|
| R(3,16) >= 82 | 81 | 1.4 | Z27 x Z3 |
| R(3,17) >= 92 (the record until May 2026, now 93) | 91 | 11.8 | circulant |
| R(3,19) >= 106 | 105 | 27.8 with two workers | circulant |
| R(3,20) >= 111 | 110 | 18.2 | Z11:Z10 (a Frobenius group) |
| R(4,9) >= 73 | 72 | 29.9 | two blocks of Z3:Z12 |
| R(4,10) >= 92 | 91 | 39.0 | circulant |
| R(4,12) >= 128 | 127 | 0.5 | circulant with multiplier 5 |
| R(3,3,5) >= 45 | 44 | 0.1 | circulant |
| R(3,4,4) >= 55 | 54 | 1.4 | circulant |

Together with the five in the benchmark, that is 13 current record sizes. On one core,
R(3,19) took 127 s in one run and was not found within 120 s in another. Multi-hour `climb`
runs on a desktop have also matched R(3,10) >= 40 (39 vertices) and R(4,6) >= 36 (35
vertices), two of the hardest small records.

**Not matched within 120 s** (best weighted number of violated cliques in brackets): R(4,6)
with 35 vertices (7) and R(3,10) with 39 (8), both matched later by multi-hour runs; further
off on R(3,11) with 46 (27), R(3,3,6) with 60 (34), R(3,12) with 52 (41), R(5,6) with 58
(89), R(3,4,5) with 88 (115), R(3,3,3,4) with 96 (123), R(4,8) with 58 (252) and R(3,3,7)
with 84 (292); far off on R(3,13), R(3,14), R(3,15), R(3,18), R(4,11), R(5,8), R(6,7) and
R(6,8). No colouring beyond a published bound has been found.

**Impossible targets** end quickly instead of running to the time limit. `climb 3 3 --from 3
--to 10` finds K_3 to K_5, proves K_6 impossible and prints `R(3,3) = 6 exactly`. A target
ruled out only by the literature, such as K_18 for (3,6), is reported as known impossible.

## How it works

Two tiers share the clock, and proofs run alongside.

**Tier S: complete search on symmetric spaces.** For the given `n` the program builds a
catalogue of up to 64 search spaces. Each one is the set of colourings invariant under a
group: circulants, circulants with multipliers (Paley-like), cyclotomic classes of GF(q),
Cayley colourings of abelian and metacyclic groups (dihedral, Frobenius), and the same
groups acting on 2 to 6 blocks with up to 2 fixed points. The variables are the edge orbits,
typically 5 to 250 of them instead of thousands of edges.

Each space is handed to one algorithm, `OrbitSearch`: depth-first search with forward
checking, smallest-domain-first, conflict-driven ordering and restarts. It ends in one of
two ways. Either it finds a colouring, or it exhausts the space, which proves that no
colouring with that symmetry exists; the space is then retired. Most spaces are decided in
milliseconds, so the time goes to the few that stay open.

**Tier L: local search, chosen by a bandit.** Tabu search with exact incremental
bookkeeping, on four kinds of arm:

* `weak`: colourings with a small symmetry group (one involution, Z_3, Z_4, Z_5 or Z_7),
  moved a whole orbit at a time;
* `descend+d`: take a near-solution on a circulant with `n+d` vertices, delete the `d`
  worst vertices, repair;
* `ascend`: take a colouring on `n-1` vertices, try every one-vertex extension exactly,
  otherwise add a vertex and repair;
* `full`: the plain edge space.

A UCB bandit learns from the outcomes which arm deserves the next time slice.

**Proofs and reporting.** Before searching, the Greenwood-Gleason recursion is checked; it
settles targets such as K_6 for (3,3) at once. For `n <= 64` a complete search of the full
space with lex-leader symmetry breaking runs alongside and can prove small cases impossible.
A table of published bounds is used for reporting only: it stops a search for a target the
literature rules out (`--ignore-known` searches anyway), and it flags a colouring that would
beat a published lower bound.

**Verification.** A claimed colouring is recounted from scratch in C with no symmetry and no
incremental state. Then `verify.py`, a pure-Python maximum-clique search that shares no code
with the search core, computes the clique number of every colour.

**Scheduling.** Tier S gets one share per open space against one per kind of local search,
at most 75%. Inside tier S the least-served space goes next, with time in proportion to
1 / (number of orbits). Several workers (`--jobs`) run the same portfolio with different
random seeds as threads; the C core releases the GIL.

## How this relates to AlphaGo Zero

The request behind this project was an AlphaGo-Zero-style learner. The honest summary after
two rounds of experiments: the learning that pays is small, and exact search does the work.

* What is learned from experience, starting from nothing: which local-search arm to pull
  (bandit), and which variables cause conflicts (the complete search reorders itself across
  restarts).
* An AlphaZero-style policy is included (`--learn`): a policy proposes an orbit colouring,
  tabu search improves it, the policy imitates the best results, with symmetric copies
  aligned first. Measured with it on and off on the same spaces and budgets, it changed
  nothing beyond seed noise, so it is off by default.
* Why Go and Ramsey differ: Go has two players, a cheap state and a learnable value. Here
  there is one player, the reward is zero until the last move, and an exact evaluation of a
  move costs microseconds, which no network can beat. Published work points the same way:
  the 2026 AlphaEvolve bounds were found by evolved local-search programs, and AlphaZero was
  matched by tabu search on a related extremal-graph problem (Mehrabian et al., 2024).

## What changed in version 2

This version came out of a full review of version 1. New ideas were tried against the old
code on the same instances, and only measured wins were kept.

**Kept**

* **Complete search on symmetric spaces** (`OrbitSearch`) replaces tabu search as the main
  tool. Version 1 could only give up on a space; this one finds a colouring or proves there
  is none. It reproduces the published circulant Ramsey numbers and finds the four-colour
  record colouring in well under a second of its own search time.
* **A general catalogue of symmetric spaces** replaces the hand-written list of circulant
  variants. Winners outside the circulants so far: Z27 x Z3 for (3,16) on 81 vertices,
  Z11:Z10 for (3,20) on 110, and three blocks of Z4 x Z4 plus two fixed points for
  (3,3,3,3) on 50.
* **Smallest clique first.** In the complete search, trying the colour with the smallest
  forbidden clique first finds colourings 1.3 to 4 times sooner than a random order on the
  record circulants for (3,17), (3,19) and (4,10); largest-first is 3 to 8 times slower
  (8 seeds each).
* **Weak-symmetry local search**: whole-orbit moves under a small group. About 50 times
  faster than the full edge space on (4,5) with 24 vertices.
* **A mixed tabu kernel**: steps through a violated clique mixed with sampled best-of-150
  steps. It reaches lower plateaus than either kind alone and repairs the circulant
  near-solution of (4,7) on 48 vertices in about 0.01 s.
* **Interruptible clique counting**, so that time limits hold when a single count is slow.
* **Threads for `--jobs`** and a table of published bounds for reporting.

**Tried and dropped**

* The AlphaZero-style policy for tabu starts: no effect beyond seed noise, now off by default.
* Phase saving in the complete search: slower (median 0.9 s against 0.5 s on the
  four-colour record space).
* Restart schedules (geometric with factor 1.1, 1.2 and 1.5, Luby, inner-outer): the
  differences stayed inside the seed noise over 8 seeds on four record spaces. Geometric
  with factor 1.5 stays.
* Search depth as a predictor of which symmetric space holds a colouring: the winning space
  ranked first or second, but by a margin too small to act on (0.96 against 0.94).
* A time split driven by local-search speed: move rates did not separate useful local
  search from useless.
* Pure focused tabu and pure sampled tabu: the first stalls on plateaus, the second fails to
  close the last violations.
* Random-start local search on frontier instances: it stalls above zero on (4,6) with 35
  vertices (best 7 violations), (3,10) with 39 (best 3) and (5,6) with 58, in full and
  weak-symmetry spaces alike, within 150 s.

**Flaws found and fixed**

* **Crash in long runs.** A time slice was computed as `min_slice * 2 ** round`, which
  overflows after about 12,000 pulls, roughly half a day of running. Capped.
* **Time limits were not kept with large k.** One clique count cannot be split, and while a
  colouring is still bad the number of cliques can be astronomical. A 60 s run on (3,19)
  with 105 vertices took 83 s, with single steps of 20 to 48 s. Counting can now be
  interrupted, a step that runs out of time takes the best move seen so far, and an arm that
  still overruns its slice sits out later turns. The same run now ends at 60.0 s.
* **Complete search was starved in long runs.** The share of tier S decayed to 15% after
  the first minute, although every win that takes longer than a few seconds comes from it.
  The share is now fixed.
* **Slow, unbounded verification.** The pure-Python check of a 110-vertex colouring took
  30 s inside the time budget. It is now about three times faster (degeneracy ordering) and
  time-limited inside `solve`; the full recount in C always runs, and `verify` on the
  command line always completes.
* **Known-impossible targets ran until the time limit.** K_18 for (3,6) is ruled out by
  R(3,6) = 18, which the program cannot prove itself. It now stops and says so.
* A colouring that was not invariant under the space's group was accepted by the engine in
  local-search mode. It is rejected now.
* The space catalogue crashed for prime-power orders such as 27.
* Windows: `time.time()` can tick as coarsely as 15.6 ms, the size of the shortest slices; a
  high-resolution clock is used now. Old compiled cores no longer pile up in the package
  folder.

## Validation

`python tests/test_core.py` runs 30 tests in about a minute. The main ones:

* The C clique counter against brute force over all subsets; the symmetric counting formula
  against full counts; the incremental bookkeeping of every local-search mode against
  rebuilds, including searches interrupted in the middle of a step.
* `OrbitSearch` against brute force on more than a thousand small spaces, with every
  combination of its options.
* `OrbitSearch` against the literature: it reproduces 19 published circulant Ramsey numbers
  RC(k,l) (Harborth and Krause 2003; Coniglio et al. 2026). For each it finds a circulant
  colouring on RC-1 vertices and proves that none exists on RC, RC+1 and RC+2 vertices.
* Both complete searches against brute force, and the known values R(3,3)=6, R(3,4)=9,
  R(3,5)=14, R(4,4)=18, R(3,3,3)=17 proved from scratch.
* The independent verifier rejects corrupted colourings.

## Usage

```python
from ramsey_zero import solve, verify, to_graph6, graph_of
res = solve([4, 5], 24, time=30)          # colour 0 avoids K4, colour 1 avoids K5
res.status                                # "solved", "proven impossible", "known impossible" or "unsolved"
G = graph_of(res.coloring, 0)             # adjacency of colour 0: omega(G) < 4, alpha(G) < 5
print(to_graph6(G), verify(res.coloring, [4, 5]))
```

* `solve(ks, n, time=60, seed=0, jobs=1, ...)` takes up to 8 colours and `n` up to 512.
  The result has `solved`, `coloring`, `route`, `impossible` and `proof`, `known_impossible`,
  `record` (beats the published bound in the built-in table) and, when unsolved,
  `best_coloring` and `best_score`.
* `climb(ks, n_from, n_to, time_per_n)` solves one size after another and feeds each
  colouring to the next size.
* Useful options, with the same names on the command line: `jobs` (parallel workers, 0 = all
  cores), `routes` (any of `sym,weak,descend,ascend,full`), `spaces` (only symmetric spaces
  whose name matches, e.g. `"circ*"`), `init` (a file to start from, see below),
  `known="warn"` / `--ignore-known`, `exact=False` / `--no-exact`.
* `--init FILE`: a colouring on `n` vertices starts the local search there; valid colourings
  on `n-1` vertices are handed to the ascend route, which extends them by one vertex and
  repairs. The file can be a certificate or a graph6 file with one graph per line (edges are
  colour 0), so published graphs such as McKay's `r46_35some.g6` can be extended directly.
* `compare FILE KNOWN.g6` reports whether the graph of colour 0 (or its complement) is
  isomorphic to a graph in a graph6 file. It needs `pip install networkx`.
* Your runs write certificates to `results/R<ks>_n<n>.txt` (the ones found while testing
  this program are in `certificates/`): one row per vertex, one colour digit
  per edge, with a JSON header (route, time, clique numbers, graph6 for two colours). An
  unsolved run saves its best near-solution as `..._best.txt`; continue from it with `--init`.
* Building blocks: `OrbitSearch(space, ks)`, `ExactSearch(ks, n)`, `upper_bound(ks)`,
  `Engine`, and the space constructors `circulant_space`, `cayley_space`, `abelian_group`,
  `metacyclic_group`, `symmetric_catalogue`, `extension_space`, `full_space`.

## Status of the problem (18 November 2026)

From Radziszowski's dynamic survey DS1.18 (24 April 2026) and everything posted since that
could be traced to a paper or a public witness file. `ramsey_zero/known.py` holds the same
values with sources; the program uses them only for reporting.

**Two colours**

| | lower | upper | note |
|---|---|---|---|
| R(3,10) | 40 | 41 | upper bound published Nov 2025; a colouring of K_40 would settle R(3,10) = 41 |
| R(3,11), R(3,12) | 47, 53 | 50, 59 | |
| R(3,13) | 61 | 68 | lower bound: AlphaEvolve, 2026 |
| R(3,14), R(3,15), R(3,16) | 67, 74, 82 | 77, 87, 97 | |
| R(3,17) | 93 | 109 | lower bound: Wang, May 2026 (92 had stood since 1994) |
| R(3,18) | 100 | 120 | lower bound: AlphaEvolve, 2026 |
| R(3,19), R(3,20) | 106, 111 | 132, 145 | |
| R(3,21), R(3,22), R(3,23) | 122, 131, 139 | 157, 171, 185 | |
| R(3,24) ... R(3,49) | 151 ... 411 | | circulants, Coniglio et al., Aug 2026 |
| R(4,6) | 36 | 40 | 37 graphs on 35 vertices known (Exoo 2012), "there could be more" |
| R(4,7), R(4,8), R(4,9) | 49, 59, 73 | 58, 79, 105 | |
| R(4,10), R(4,11), R(4,12) | 92, 102, 128 | 135, 170, 210 | |
| R(4,13) ... R(4,16) | 139, 148, 163, 174 | | AlphaEvolve and Wang, 2026 |
| R(4,20) | 252 | | Yu, Aug 2026 |
| R(5,5) | 43 | 46 | conjectured to be 43 |
| R(5,6), R(5,7), R(5,8) | 59, 80, 101 | 85, 133, 193 | |
| R(6,6), R(6,7) | 102, 115 | 160, 270 | |
| R(6,8) | 135 | 423 | lower bound: Cremer, Oct 2026 |

**Three or more colours**

| | lower | upper | note |
|---|---|---|---|
| R(3,3,5) | 45 | 57 | |
| R(3,3,6) | 61 | 91 | upper bound: Boza, Mar 2026 |
| R(3,3,7), R(3,3,8) | 85, 103 | | Exoo's linear colourings |
| R(3,3,9) ... R(3,3,15) | 133, 150, 183, 203, 232, 267, 278 | | all but k = 10 raised by circulants (Dauth, Oct 2026) |
| R(3,4,4), R(3,4,5) | 55, 89 | 77, 157 | upper bound for R(3,4,5): Boza 2026 |
| R(3,4,6) ... R(3,4,9) | 117, 152, 193, 242 | | 193 is just the formula 4 R(4,7) - 3 |
| R(3,5,5), R(3,5,6), R(3,5,7) | 139, 181, 241 | | 241 is just the formula 4 R(3,3,6) - 3 |
| R(4,4,4) | 128 | 229 | lower bound from 1982 (cubic residues mod 127); upper: Boza 2026 |
| R(3,3,3,3) | 51 | 62 | lower bound from 1973 |
| R(3,3,3,4), R(3,3,3,5) | 97, 162 | 149, - | the published construction for 162 cannot be reproduced from public data |
| R(3,3,4,4), R(3,3,3,3,3) | 174, 162 | 450, 307 | |

A 2026 preprint claims much lower upper bounds for several three-colour numbers with flag
algebras (Dauth: R(4,4,4) <= 198, R(3,4,4) <= 73, R(3,3,6) <= 86). The program does not use
unrefereed upper bounds, because an upper bound makes it refuse to search.

**What happened in 2026.** Most new lower bounds came from AI-assisted search: AlphaEvolve
(R(3,13), R(3,18), R(4,13) to R(4,20)), coding agents (R(3,17) >= 93, R(6,8) >= 135,
R(8,10) >= 345, R(7,12) >= 420), and exact searches over circulant colourings (R(3,24) to
R(3,49); R(3,3,9) to R(3,3,15)). The recipe behind most two-colour records: take the
published record graph, add one vertex, repair by local search. The `ramsey-witnesses`
project (github.com/chrispbsmolen/ramsey-witnesses) re-checks a public witness graph for
every bound in the survey.

## What to try next

Results here come in two kinds. Instant ones come from a symmetric space that contains a
colouring, which the complete search finds in seconds. Results that take hours come from
local search, or from a symmetric space so large that the complete search needs hours to
decide it. The targets below are where one of the two has a real chance and the literature
has a gap. The numbers come from 2 to 10 minute runs on a 2-core VM; a desktop with
`--jobs 0` and hours per target goes much further.

**Contributions available now**

* **R(3,23) >= 139 now has a public witness.** The survey lists this bound, but no graph for
  it was public (ramsey-witnesses, 16 Nov 2026: "still no file"). This program finds one in
  about 2 minutes with two workers: a circulant on 138 vertices,
  `certificates/R3-23_n138.txt` (graph6: `certificates/R3-23_n138.g6`), checked by the C
  recount, the independent verifier and networkx.
* **Is your (4,6,35) graph new?** Only 37 are known (Exoo 2012), and McKay's Ramsey graph
  page says "there could be more". Download `r46_35some.g6` there and run
  `python -m ramsey_zero compare results/R4-6_n35.txt r46_35some.g6`.

**Best chances of a new lower bound**

| target | why | command |
|---|---|---|
| R(3,23) on 139, R(3,22) on 131, R(3,21) on 122 | the circulant on 138 works for (3,23); the 2026 circulant study covered R(3,24) to R(3,49) only. In 10 minutes the circulants on 139 and 131 were neither found nor ruled out. Each space is a finite job: a find is a new bound, exhaustion rules that symmetry out | `solve 3 23 --n 139 --routes sym --time 86400 --jobs 0` |
| R(3,17) on 92, then 93 | the record 93 rests on one graph, found by AI agents in May 2026. The circulant on 91 takes seconds; `climb` then extends it by a vertex and repairs, the recipe behind most 2026 records | `climb 3 17 --from 91 --to 93 --time-per-n 86400 --jobs 0` |
| R(4,6) on 36 from every known 35-vertex graph | the same recipe, started from all 37 published graphs plus yours | `solve 4 6 --n 36 --init r46_35some.g6 --time 86400 --jobs 0` |
| R(3,3,6) on 60, then 61 | reaches 59 in seconds. The record 61 comes from a 2015 paper whose three-colour bounds were mostly derived from two-colour graphs; a direct circulant search beat that paper's R(3,3,k) bounds for k = 11 to 15 in October 2026 | `climb 3 3 6 --from 59 --to 62 --time-per-n 86400 --jobs 0` |

**Famous open cases (long shots).** R(3,10) on 40 would settle R(3,10) = 41; also R(4,6)
on 36, R(3,3,3,3) on 51, R(3,3,5) on 45 and R(3,4,4) on 55. In 2-minute runs most of their
symmetric spaces were proven empty within seconds (22 of 30 for R(3,10), 15 of 16 for
R(3,3,3,3)), and local search stayed 12 to 31 violations short. Decades of work have gone
into each, so leave `climb` running past your current results, but expect nothing.
**Skip R(5,5) on 43**: local search gets within 2 violations in 2 minutes, a well-known
trap. McKay and Radziszowski's evidence points to R(5,5) = 43.

**Records not matched yet (benchmarks, hours each).** R(3,11) on 46, R(3,12) on 52, R(4,8)
and R(5,6) on 58, R(5,7) on 79, and the multicolour records the solver approaches in seconds:
R(3,3,7) on 84 (reaches 80), R(3,3,8) on 102 (98), R(3,4,5) on 88 (81), R(3,4,6) on 116
(104), R(3,3,3,4) on 96 (86) and R(3,5,5) on 138 (121).

**Tested, not promising here.** Multicolour bounds that come only from a formula (R(3,4,8)
>= 193, R(3,5,7) >= 241) or from 2015 two-colour constructions (R(3,4,7) >= 152, R(3,4,9)
>= 242, R(3,3,10) >= 150, R(3,3,16) >= 291): the solver stalls 20 to 35 vertices below
them. Exoo-style linear colourings (colour depends only on j - i): a quick prototype stalled
below the circulants, so it is not included.

## Limits

* **No new bound.** Everything found matches the literature. The one new contribution is a
  public witness for R(3,23) >= 139. The unmatched records above have no colouring in any
  of the symmetric spaces the catalogue could decide, and local search stalls a few
  violations short on the closest ones.
* **Breadth costs time.** A winning space usually needs about a second of search of its
  own, but it shares the clock with 10 to 30 other spaces and the local-search arms, so the
  wall-clock time is 10 to 30 times that. `--jobs` adds workers; `--spaces "circ*"` narrows
  the symmetric tier when you already suspect a family.
* **Parallel speed-up is measured on 2 cores only**: R(3,3,3,3) on 50 vertices had a median
  of 7.4 s with two workers against 8.5 s with one (5 seeds), R(4,7) on 48 had 2.2 s
  against 2.7 s. More cores should help more, since the workers use different random
  seeds, but that is untested here.
* **Large k is hard for local search.** For targets like (3,19) a colouring that is still
  bad can hold billions of forbidden cliques, so exact counting is slow and the local tier
  contributes little; the complete search carries those targets.
* **Proofs stay small.** The full-space complete search settles cases up to about 17
  vertices. R(3,6) = 18 and beyond are taken from the literature table, not re-proved.
* Sizes: up to 8 colours and 512 vertices.

The program prints `NEW LOWER BOUND?` if it ever beats its table; check such a result
against the current literature before claiming it. "What to try next" above lists targets.
