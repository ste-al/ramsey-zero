/*
 * _core.c -- exact search kernels for Ramsey colourings (RamseyZero).
 *
 * Problem: colour the edges of K_n with r colours so that colour c contains no
 * K_{k_c}.  Objective = sum_c w_c * (#monochromatic K_{k_c} in colour c).
 * A colouring with objective 0 certifies R(k_0,...,k_{r-1}) > n.
 *
 * Variables are edge ORBITS of a permutation group (a single edge when there is no
 * symmetry); a colouring always gives every edge of an orbit the same colour.
 *
 *  rz_search_edge   local search with exact single-edge bookkeeping, for orbits of any
 *                   size and for spaces with frozen edges.  Per-edge delta:
 *                     delta(uv: a->b) = -w_a * #K_{k_a-2}[N_a(u)&N_a(v)]
 *                                       +w_b * #K_{k_b-2}[N_b(u)&N_b(v)]
 *                   bad[e] (#violated cliques through e) is maintained incrementally.
 *                   Modes: focused tabu, sampled tabu, annealing, mixed tabu (default).
 *  rz_search_sym    tabu over orbits of a large group.  Clique counts are exact from
 *                   vertex-orbit representatives:
 *                     #K_k(colour c) = (1/k) * sum_i |orbit_i| * L_c(rep_i),
 *                     L_c(v) = #K_{k-1} inside N_c(v).
 *  ox_*             COMPLETE search over orbit variables with forward checking: finds a
 *                   colouring of the space or proves that it has none.
 *  rx_*             COMPLETE search of the full space with lex-leader symmetry breaking:
 *                   can prove that no colouring of K_n exists at all (small n).
 *
 * Bitsets of up to MAXW 64-bit words: n <= 64*MAXW.  No global state: every function
 * works on its own handle, so separate handles can be used from separate threads.
 * Build (Linux/macOS): gcc -O3 -march=native -shared -fPIC -o _core.so _core.c -lm
 * Build (Windows, MinGW-w64): gcc -O3 -march=native -shared -static-libgcc -static -o _core.dll _core.c
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#define RZ_API __declspec(dllexport)
#else
#include <time.h>
#define RZ_API __attribute__((visibility("default")))
#endif

#define MAXW 8
#define MAXR 8
#define MAXK 64

typedef uint64_t u64;
typedef long long ll;

static inline int popc(u64 x) { return __builtin_popcountll(x); }
static inline int ctz64(u64 x) { return __builtin_ctzll(x); }

/* ---------------- RNG: xoshiro256** seeded by splitmix64 ---------------- */
typedef struct { u64 s[4]; } rng_t;
static inline u64 rotl(u64 x, int k) { return (x << k) | (x >> (64 - k)); }
static u64 rng_next(rng_t *r) {
    u64 *s = r->s;
    u64 res = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
    return res;
}
static void rng_seed(rng_t *r, u64 seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9E3779B97F4A7C15ULL;
        u64 z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        r->s[i] = z ^ (z >> 31);
    }
}
static inline double rng_unif(rng_t *r) { return (rng_next(r) >> 11) * (1.0 / 9007199254740992.0); }
static inline u64 rng_below(rng_t *r, u64 n) { return n ? rng_next(r) % n : 0; }

#ifdef _WIN32
/* QueryPerformanceCounter is in KERNEL32.  clock_gettime would come from libwinpthread
   under MinGW-w64, a runtime DLL that Python >= 3.8 does not look for on PATH. */
static double now_sec(void) {
    static double inv = 0.0;
    LARGE_INTEGER c;
    if (inv == 0.0) {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        inv = 1.0 / (double)f.QuadPart;
    }
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * inv;
}
#else
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

/* ---------------- clique counting / enumeration on bitsets ---------------- */

/* For j >= CB_MIN_J the counters use a greedy-colouring bound (as in max-clique solvers):
   colour S greedily with classes 1..c (each class an independent set), list vertices by
   class, and count every j-clique at its last vertex v in that order.  The clique's other
   vertices precede v and all clique vertices have distinct classes <= class(v), so
   class(v) < j means no j-clique can end at v or anywhere earlier -> stop.  Exact, and it
   prunes the huge trees of smaller cliques that dense colour classes produce (large k). */
#ifndef CB_MIN_J
#define CB_MIN_J 5
#endif

/* Time budget.  Counting or listing cliques can take arbitrarily long: while a colouring
   is still bad the number of cliques can be astronomical (large k).  A BG makes the work
   interruptible: once its deadline has passed, `out` is set, every counter returns at once
   and the results of the interrupted call are void.  bg == NULL: never interrupt. */
typedef struct { double deadline; unsigned tick; int out; } BG;
static void bg_init(BG *bg, double t0, double tlim) {
    bg->deadline = tlim >= 0 ? t0 + tlim : 1e300;
    bg->tick = 0; bg->out = 0;
}
static inline int bg_out(BG *bg) {
    if (!bg) return 0;
    if (!bg->out && (++bg->tick & 255) == 0 && now_sec() > bg->deadline) bg->out = 1;
    return bg->out;
}

static int greedy_colour(const u64 *adj, int W, const u64 *S, int *order, int *cls, int *ncls) {
    u64 U[MAXW];
    memcpy(U, S, sizeof(u64) * W);
    int m = 0, c = 0;
    for (;;) {
        int any = 0;
        for (int w = 0; w < W; w++) any |= (U[w] != 0);
        if (!any) break;
        c++;
        u64 Q[MAXW];
        memcpy(Q, U, sizeof(u64) * W);
        for (int w = 0; w < W; w++) {
            while (Q[w]) {
                int v = w * 64 + ctz64(Q[w]);
                Q[w] &= Q[w] - 1;
                U[w] &= ~(1ULL << (v & 63));
                const u64 *av = adj + (size_t)v * W;
                for (int x = w; x < W; x++) Q[x] &= ~av[x];
                order[m] = v; cls[m] = c; m++;
            }
        }
    }
    *ncls = c;
    return m;
}

static ll count_cliques(const u64 *adj, int W, const u64 *S, int j, BG *bg);

static ll count_cliques_cb(const u64 *adj, int W, const u64 *S, int j, BG *bg) {
    if (bg_out(bg)) return 0;
    int order[64 * MAXW], cls[64 * MAXW], nc;
    int m = greedy_colour(adj, W, S, order, cls, &nc);
    if (nc < j) return 0;
    u64 R[MAXW];
    memcpy(R, S, sizeof(u64) * W);
    ll tot = 0;
    for (int i = m - 1; i >= 0; i--) {
        if (cls[i] < j) break;
        int v = order[i];
        R[v >> 6] &= ~(1ULL << (v & 63));
        const u64 *av = adj + (size_t)v * W;
        u64 S2[MAXW];
        int s2 = 0;
        for (int w = 0; w < W; w++) { S2[w] = R[w] & av[w]; s2 += popc(S2[w]); }
        if (s2 >= j - 1) tot += count_cliques(adj, W, S2, j - 1, bg);
    }
    return tot;
}

/* number of j-cliques contained in vertex set S of graph `adj` (rows of W words) */
static ll count_cliques(const u64 *adj, int W, const u64 *S, int j, BG *bg) {
    if (j <= 0) return 1;
    int sz = 0;
    for (int i = 0; i < W; i++) sz += popc(S[i]);
    if (sz < j) return 0;
    if (j == 1) return sz;
    if (j >= CB_MIN_J) return count_cliques_cb(adj, W, S, j, bg);
    ll tot = 0;
    u64 T[MAXW];
    memcpy(T, S, sizeof(u64) * W);
    int rem = sz;
    for (int i = 0; i < W; i++) {
        while (T[i]) {
            if (rem < j) return tot;
            int v = i * 64 + ctz64(T[i]);
            T[i] &= T[i] - 1;
            rem--;
            const u64 *av = adj + (size_t)v * W;
            u64 S2[MAXW];
            int s2 = 0;
            for (int w = 0; w < i; w++) S2[w] = 0;
            for (int w = i; w < W; w++) { S2[w] = T[w] & av[w]; s2 += popc(S2[w]); }
            if (j == 2) tot += s2;
            else if (s2 >= j - 1) tot += count_cliques(adj, W, S2, j - 1, bg);
        }
    }
    return tot;
}

typedef void (*clq_cb)(void *ctx, const int *verts, int m);

static void enum_cliques(const u64 *adj, int W, const u64 *S, int j, int *stack, int depth,
                         clq_cb cb, void *ctx, BG *bg);

static void enum_cliques_cb(const u64 *adj, int W, const u64 *S, int j, int *stack, int depth,
                            clq_cb cb, void *ctx, BG *bg) {
    if (bg_out(bg)) return;
    int order[64 * MAXW], cls[64 * MAXW], nc;
    int m = greedy_colour(adj, W, S, order, cls, &nc);
    if (nc < j) return;
    u64 R[MAXW];
    memcpy(R, S, sizeof(u64) * W);
    for (int i = m - 1; i >= 0; i--) {
        if (cls[i] < j) break;
        int v = order[i];
        R[v >> 6] &= ~(1ULL << (v & 63));
        stack[depth] = v;
        const u64 *av = adj + (size_t)v * W;
        u64 S2[MAXW];
        int s2 = 0;
        for (int w = 0; w < W; w++) { S2[w] = R[w] & av[w]; s2 += popc(S2[w]); }
        if (s2 >= j - 1) enum_cliques(adj, W, S2, j - 1, stack, depth + 1, cb, ctx, bg);
    }
}

/* enumerate all j-cliques inside S; callback gets the chosen vertices */
static void enum_cliques(const u64 *adj, int W, const u64 *S, int j, int *stack, int depth,
                         clq_cb cb, void *ctx, BG *bg) {
    if (j <= 0) { cb(ctx, stack, depth); return; }
    int sz = 0;
    for (int i = 0; i < W; i++) sz += popc(S[i]);
    if (sz < j) return;
    if (j >= CB_MIN_J) { enum_cliques_cb(adj, W, S, j, stack, depth, cb, ctx, bg); return; }
    u64 T[MAXW];
    memcpy(T, S, sizeof(u64) * W);
    int rem = sz;
    for (int i = 0; i < W; i++) {
        while (T[i]) {
            if (rem < j) return;
            int v = i * 64 + ctz64(T[i]);
            T[i] &= T[i] - 1;
            rem--;
            stack[depth] = v;
            if (j == 1) { cb(ctx, stack, depth + 1); continue; }
            const u64 *av = adj + (size_t)v * W;
            u64 S2[MAXW];
            int s2 = 0;
            for (int w = 0; w < i; w++) S2[w] = 0;
            for (int w = i; w < W; w++) { S2[w] = T[w] & av[w]; s2 += popc(S2[w]); }
            if (s2 >= j - 1) enum_cliques(adj, W, S2, j - 1, stack, depth + 1, cb, ctx, bg);
        }
    }
}

/* ---------------- main state ---------------- */
typedef struct {
    int n, W, r;
    int k[MAXR];
    double wt[MAXR];
    u64 *adj;        /* r * n * W */
    int8_t *col;     /* n * n, colour of edge (diag = -1) */
    ll cnt[MAXR];    /* current number of monochromatic K_{k_c} per colour */
    /* variables (edge orbits); edges not in any orbit are frozen */
    int V;
    int *optr, *oe;  /* orbit edges, encoded u*n+v with u<v */
    int *eorb;       /* n*n -> orbit id or -1 */
    int8_t *ocol;    /* colour of each orbit (sym mode) */
    /* sym-mode data */
    int sym, R;
    int *reps, *rsz;
    ll *Lrep;        /* r * R */
    /* edge-mode data */
    int *bad, *badpos, *badlist, nbad, bad_ok;
    int bad_cur;     /* next edge (u*n+v) of an unfinished build of bad[]; 0 = none */
    /* best-so-far of the last search */
    int8_t *bestcol;
    double bestscore;
    ll *tabu;        /* per variable */
    int *mark; int stamp;
    int ready;       /* 0: no colouring loaded, or loading it ran out of time */
    rng_t rng;
} RZ;

static inline u64 *ROW(RZ *z, int c, int v) { return z->adj + ((size_t)c * z->n + v) * z->W; }
static inline const u64 *ADJC(RZ *z, int c) { return z->adj + (size_t)c * z->n * z->W; }
static inline void setb(u64 *row, int v) { row[v >> 6] |= (1ULL << (v & 63)); }
static inline void clrb(u64 *row, int v) { row[v >> 6] &= ~(1ULL << (v & 63)); }
static inline int eid(RZ *z, int u, int v) { return u < v ? u * z->n + v : v * z->n + u; }

static inline void recolor(RZ *z, int u, int v, int b) {
    int n = z->n, a = z->col[u * n + v];
    if (a == b) return;
    clrb(ROW(z, a, u), v); clrb(ROW(z, a, v), u);
    setb(ROW(z, b, u), v); setb(ROW(z, b, v), u);
    z->col[u * n + v] = z->col[v * n + u] = (int8_t)b;
}

static inline void common_nb(RZ *z, int c, int u, int v, u64 *S) {
    const u64 *au = ROW(z, c, u), *av = ROW(z, c, v);
    for (int w = 0; w < z->W; w++) S[w] = au[w] & av[w];
}

/* #cliques of colour c that would contain edge uv if uv had colour c */
static inline ll edge_cnt(RZ *z, int u, int v, int c, BG *bg) {
    u64 S[MAXW];
    common_nb(z, c, u, v, S);
    return count_cliques(ADJC(z, c), z->W, S, z->k[c] - 2, bg);
}

static double score_of(RZ *z) {
    double s = 0;
    for (int c = 0; c < z->r; c++) s += z->wt[c] * (double)z->cnt[c];
    return s;
}
static int solved(RZ *z) {
    for (int c = 0; c < z->r; c++) if (z->cnt[c] != 0) return 0;
    return 1;
}

static void full_counts(RZ *z, ll *out, BG *bg) {
    u64 ALL[MAXW];
    memset(ALL, 0, sizeof(ALL));
    for (int v = 0; v < z->n; v++) setb(ALL, v);
    for (int c = 0; c < z->r; c++) out[c] = count_cliques(ADJC(z, c), z->W, ALL, z->k[c], bg);
}

/* ---------------- edge-mode bookkeeping ---------------- */
static void bad_add(RZ *z, int e, int d) {
    int old = z->bad[e], nw = old + d;
    z->bad[e] = nw;
    if (old == 0 && nw > 0) { z->badpos[e] = z->nbad; z->badlist[z->nbad++] = e; }
    else if (old > 0 && nw == 0) {
        int p = z->badpos[e], last = z->badlist[--z->nbad];
        z->badlist[p] = last; z->badpos[last] = p;
    }
}
typedef struct { RZ *z; int u, v, d; ll cnt; } badctx;
static void bad_cb(void *vc, const int *st, int m) {
    badctx *c = (badctx *)vc;
    RZ *z = c->z;
    c->cnt++;
    if (!c->d) return;
    int vs[MAXK + 2], M = m + 2;
    vs[0] = c->u; vs[1] = c->v;
    for (int i = 0; i < m; i++) vs[i + 2] = st[i];
    for (int a = 0; a < M; a++)
        for (int b = a + 1; b < M; b++) bad_add(z, eid(z, vs[a], vs[b]), c->d);
}
/* Build bad[] from the current colouring.  Returns 1 when complete, 0 if the budget ran
   out.  The work is resumable: bad_cur remembers the next edge, so repeated short calls
   finish the job (the first count of every call is never interrupted). */
static int build_bad(RZ *z, BG *bg) {
    int n = z->n, first = 1;
    int u0 = z->bad_cur / n, v0 = z->bad_cur % n;
    if (z->bad_cur == 0) {
        memset(z->bad, 0, sizeof(int) * n * n);
        z->nbad = 0;
    }
    z->bad_ok = 0;
    for (int u = u0; u < n - 1; u++) {
        int vs = (u == u0 && v0 > u) ? v0 : u + 1;
        if (!first && bg && (bg->out || now_sec() > bg->deadline)) { z->bad_cur = u * n + vs; return 0; }
        for (int v = vs; v < n; v++) {
            ll x = edge_cnt(z, u, v, z->col[u * n + v], first ? NULL : bg);
            if (!first && bg && bg->out) { z->bad_cur = u * n + v; return 0; }   /* x is void */
            first = 0;
            if (x > 0) {
                int e = u * n + v;
                z->bad[e] = (int)x;
                z->badpos[e] = z->nbad;
                z->badlist[z->nbad++] = e;
            }
        }
    }
    z->bad_cur = 0;
    z->bad_ok = 1;
    return 1;
}

/* recolour edge uv to b, updating counts (and bad[] if maintained).  Never interrupted: a
   half-applied move would leave the bookkeeping wrong. */
static void edge_apply(RZ *z, int u, int v, int b) {
    int a = z->col[u * z->n + v];
    if (a == b) return;
    int st[MAXK];
    u64 S[MAXW];
    badctx c1 = {z, u, v, z->bad_ok ? -1 : 0, 0};
    common_nb(z, a, u, v, S);
    if (z->bad_ok) enum_cliques(ADJC(z, a), z->W, S, z->k[a] - 2, st, 0, bad_cb, &c1, NULL);
    else c1.cnt = count_cliques(ADJC(z, a), z->W, S, z->k[a] - 2, NULL);
    recolor(z, u, v, b);
    badctx c2 = {z, u, v, z->bad_ok ? +1 : 0, 0};
    common_nb(z, b, u, v, S);
    if (z->bad_ok) enum_cliques(ADJC(z, b), z->W, S, z->k[b] - 2, st, 0, bad_cb, &c2, NULL);
    else c2.cnt = count_cliques(ADJC(z, b), z->W, S, z->k[b] - 2, NULL);
    z->cnt[a] -= c1.cnt;
    z->cnt[b] += c2.cnt;
}

typedef struct { rng_t *rng; ll seen; int chosen[MAXK]; int m; } pickctx;
static void pick_cb(void *vc, const int *st, int m) {
    pickctx *p = (pickctx *)vc;
    p->seen++;
    if (rng_below(p->rng, p->seen) == 0) { memcpy(p->chosen, st, sizeof(int) * m); p->m = m; }
}

/* random violated clique through a random bad edge; returns size, 0 if none (also when the
   budget ran out: check bg->out) */
static int edge_pick_violation(RZ *z, int *verts, BG *bg) {
    if (z->nbad == 0) return 0;
    int n = z->n, e = z->badlist[rng_below(&z->rng, z->nbad)];
    int u = e / n, v = e % n, c = z->col[u * n + v];
    u64 S[MAXW];
    int st[MAXK];
    common_nb(z, c, u, v, S);
    pickctx p;
    p.rng = &z->rng; p.seen = 0; p.m = 0;
    enum_cliques(ADJC(z, c), z->W, S, z->k[c] - 2, st, 0, pick_cb, &p, bg);
    if (bg && bg->out) return 0;
    verts[0] = u; verts[1] = v;
    for (int i = 0; i < p.m; i++) verts[2 + i] = p.chosen[i];
    return p.m + 2;
}

/* ---------------- sym-mode helpers ---------------- */
static ll sym_count_color(RZ *z, int c, ll *Lout, BG *bg) {
    ll tot = 0;
    const u64 *A = ADJC(z, c);
    for (int i = 0; i < z->R; i++) {
        int v = z->reps[i];
        ll L = count_cliques(A, z->W, ROW(z, c, v), z->k[c] - 1, bg);
        if (Lout) Lout[i] = L;
        tot += (ll)z->rsz[i] * L;
    }
    return tot / z->k[c];
}
static void sym_all_counts(RZ *z, BG *bg) {
    for (int c = 0; c < z->r; c++) z->cnt[c] = sym_count_color(z, c, z->Lrep + (size_t)c * z->R, bg);
}
static void orbit_recolor(RZ *z, int o, int b) {
    int n = z->n;
    for (int t = z->optr[o]; t < z->optr[o + 1]; t++) {
        int e = z->oe[t];
        recolor(z, e / n, e % n, b);
    }
    z->ocol[o] = (int8_t)b;
}

/* random violated clique in a G-invariant colouring (through a representative); 0 if none
   (also when the budget ran out: check bg->out) */
static int sym_pick_violation(RZ *z, int *verts, BG *bg) {
    double tot = 0;
    for (int c = 0; c < z->r; c++)
        for (int i = 0; i < z->R; i++) tot += (double)z->rsz[i] * z->Lrep[(size_t)c * z->R + i];
    if (tot <= 0) return 0;
    double x = rng_unif(&z->rng) * tot;
    int cc = -1, ii = -1;
    for (int c = 0; c < z->r && cc < 0; c++)
        for (int i = 0; i < z->R; i++) {
            x -= (double)z->rsz[i] * z->Lrep[(size_t)c * z->R + i];
            if (x <= 0 && z->Lrep[(size_t)c * z->R + i] > 0) { cc = c; ii = i; break; }
        }
    if (cc < 0) { /* numeric fallthrough: take the last positive entry */
        for (int c = 0; c < z->r; c++)
            for (int i = 0; i < z->R; i++)
                if (z->Lrep[(size_t)c * z->R + i] > 0) { cc = c; ii = i; }
    }
    int v = z->reps[ii], st[MAXK];
    pickctx p;
    p.rng = &z->rng; p.seen = 0; p.m = 0;
    enum_cliques(ADJC(z, cc), z->W, ROW(z, cc, v), z->k[cc] - 1, st, 0, pick_cb, &p, bg);
    if (bg && bg->out) return 0;
    verts[0] = v;
    for (int i = 0; i < p.m; i++) verts[1 + i] = p.chosen[i];
    return p.m + 1;
}

/* ---------------- public API ---------------- */
RZ_API void *rz_create(int n, int r, const int *k, const double *wt, unsigned long long seed) {
    if (n < 1 || n > 64 * MAXW || r < 1 || r > MAXR) return NULL;
    RZ *z = (RZ *)calloc(1, sizeof(RZ));
    z->n = n; z->r = r; z->W = (n + 63) / 64;
    for (int c = 0; c < r; c++) { z->k[c] = k[c]; z->wt[c] = wt ? wt[c] : 1.0; }
    z->adj = (u64 *)calloc((size_t)r * n * z->W, sizeof(u64));
    z->col = (int8_t *)malloc((size_t)n * n);
    z->bestcol = (int8_t *)malloc((size_t)n * n);
    z->eorb = (int *)malloc(sizeof(int) * n * n);
    z->bad = (int *)calloc((size_t)n * n, sizeof(int));
    z->badpos = (int *)calloc((size_t)n * n, sizeof(int));
    z->badlist = (int *)calloc((size_t)n * n, sizeof(int));
    for (int i = 0; i < n * n; i++) { z->col[i] = 0; z->eorb[i] = -1; }
    for (int i = 0; i < n; i++) z->col[i * n + i] = -1;
    rng_seed(&z->rng, seed);
    return z;
}

RZ_API void rz_free(void *h) {
    RZ *z = (RZ *)h;
    if (!z) return;
    free(z->adj); free(z->col); free(z->bestcol); free(z->eorb);
    free(z->bad); free(z->badpos); free(z->badlist);
    free(z->optr); free(z->oe); free(z->ocol); free(z->tabu); free(z->mark);
    free(z->reps); free(z->rsz); free(z->Lrep);
    free(z);
}

RZ_API void rz_set_weights(void *h, const double *wt) {
    RZ *z = (RZ *)h;
    for (int c = 0; c < z->r; c++) z->wt[c] = wt[c];
}

RZ_API void rz_seed(void *h, unsigned long long seed) { rng_seed(&((RZ *)h)->rng, seed); }

/* oedges: flattened pairs (u,v) of all orbits; optr: V+1 offsets (in edges).
   sym=1: R representatives with orbit sizes rsz (sum must be n). */
RZ_API int rz_set_orbits(void *h, int V, const int *optr, const int *oedges, int sym, int R,
                  const int *reps, const int *rsz) {
    RZ *z = (RZ *)h;
    int n = z->n;
    free(z->optr); free(z->oe); free(z->ocol); free(z->tabu); free(z->mark);
    free(z->reps); free(z->rsz); free(z->Lrep);
    z->V = V;
    z->optr = (int *)malloc(sizeof(int) * (V + 1));
    memcpy(z->optr, optr, sizeof(int) * (V + 1));
    int E = optr[V];
    z->oe = (int *)malloc(sizeof(int) * (E > 0 ? E : 1));
    z->ocol = (int8_t *)calloc(V > 0 ? V : 1, 1);
    z->tabu = (ll *)calloc(V > 0 ? V : 1, sizeof(ll));
    z->mark = (int *)calloc(V > 0 ? V : 1, sizeof(int));
    z->stamp = 0;
    for (int i = 0; i < n * n; i++) z->eorb[i] = -1;
    for (int o = 0; o < V; o++)
        for (int t = optr[o]; t < optr[o + 1]; t++) {
            int u = oedges[2 * t], v = oedges[2 * t + 1];
            if (u < 0 || v < 0 || u >= n || v >= n || u == v) return -1;
            if (z->eorb[u * n + v] != -1) return -2; /* edge in two orbits */
            z->eorb[u * n + v] = z->eorb[v * n + u] = o;
            z->oe[t] = u < v ? u * n + v : v * n + u;
        }
    z->sym = sym;
    z->R = sym ? R : 0;
    z->reps = NULL; z->rsz = NULL; z->Lrep = NULL;
    if (sym) {
        int tot = 0;
        z->reps = (int *)malloc(sizeof(int) * R);
        z->rsz = (int *)malloc(sizeof(int) * R);
        z->Lrep = (ll *)calloc((size_t)z->r * R, sizeof(ll));
        for (int i = 0; i < R; i++) { z->reps[i] = reps[i]; z->rsz[i] = rsz[i]; tot += rsz[i]; }
        if (tot != n) return -3;
        if (E != n * (n - 1) / 2) return -4; /* sym mode requires all edges free */
    }
    z->bad_ok = 0; z->bad_cur = 0;
    return 0;
}

/* Load a full colouring (n*n, symmetric) and count its cliques.  Returns 0 on success, -1
   bad colours, -2 not constant on an orbit, -3 the count did not finish within tlim seconds
   (tlim < 0: no limit).  After a failure the handle holds no colouring: load one before
   searching. */
RZ_API int rz_set_coloring(void *h, const int8_t *col, double tlim) {
    RZ *z = (RZ *)h;
    int n = z->n, W = z->W;
    BG bg;
    bg_init(&bg, now_sec(), tlim);
    z->ready = 0;
    memset(z->adj, 0, sizeof(u64) * (size_t)z->r * n * W);
    for (int u = 0; u < n; u++) {
        z->col[u * n + u] = -1;
        for (int v = 0; v < n; v++) {
            if (u == v) continue;
            int c = col[u * n + v];
            if (c < 0 || c >= z->r || col[v * n + u] != c) return -1;
            z->col[u * n + v] = (int8_t)c;
            setb(ROW(z, c, u), v);
        }
    }
    for (int o = 0; o < z->V; o++) {
        int e0 = z->oe[z->optr[o]];
        int c0 = z->col[e0];
        z->ocol[o] = (int8_t)c0;
        for (int t = z->optr[o] + 1; t < z->optr[o + 1]; t++)
            if (z->col[z->oe[t]] != c0) return -2; /* not invariant */
        z->tabu[o] = 0;
    }
    z->bad_ok = 0; z->bad_cur = 0;
    if (z->sym) sym_all_counts(z, &bg);
    else full_counts(z, z->cnt, &bg);
    if (bg.out) return -3;
    memcpy(z->bestcol, z->col, (size_t)n * n);
    z->bestscore = score_of(z);
    z->ready = 1;
    return 0;
}

RZ_API void rz_get_coloring(void *h, int8_t *out) { RZ *z = (RZ *)h; memcpy(out, z->col, (size_t)z->n * z->n); }
RZ_API void rz_get_best(void *h, int8_t *out) { RZ *z = (RZ *)h; memcpy(out, z->bestcol, (size_t)z->n * z->n); }
RZ_API void rz_get_counts(void *h, ll *out) { RZ *z = (RZ *)h; for (int c = 0; c < z->r; c++) out[c] = z->cnt[c]; }
RZ_API double rz_score(void *h) { return score_of((RZ *)h); }

/* exact clique counts by brute bitset recursion over the whole graph (self-check) */
RZ_API void rz_full_counts(void *h, ll *out) { full_counts((RZ *)h, out, NULL); }

/* standalone counter: col n*n, k[r] -> out[r].  Returns 0 ok, 1 not finished within tlim
   seconds (tlim < 0: no limit; out is void then), -1 bad input (out = -1). */
RZ_API int rz_count_coloring(int n, int r, const int8_t *col, const int *k, double tlim, ll *out) {
    double wt[MAXR] = {0};
    for (int c = 0; c < r && c < MAXR; c++) wt[c] = 1.0;
    RZ *z = (RZ *)rz_create(n, r, k, wt, 1);
    if (!z) { for (int c = 0; c < r && c < MAXR; c++) out[c] = -1; return -1; }
    int V = 0, optr = 0;
    rz_set_orbits(z, V, &optr, NULL, 0, 0, NULL, NULL);
    int rc = rz_set_coloring(z, col, tlim);
    for (int c = 0; c < r; c++) out[c] = rc == 0 ? z->cnt[c] : -1;
    rz_free(z);
    return rc == 0 ? 0 : (rc == -3 ? 1 : -1);
}

/* self-check: compare incrementally maintained bad[] with a rebuild; returns #mismatches */
RZ_API int rz_check_bad(void *h) {
    RZ *z = (RZ *)h;
    int n = z->n, mism = 0;
    if (!z->bad_ok) return -1;
    int nb = 0;
    for (int u = 0; u < n; u++)
        for (int v = u + 1; v < n; v++) {
            int e = u * n + v;
            ll x = edge_cnt(z, u, v, z->col[e], NULL);
            if (x != z->bad[e]) mism++;
            if (z->bad[e] > 0) {
                nb++;
                if (z->badpos[e] < 0 || z->badpos[e] >= z->nbad || z->badlist[z->badpos[e]] != e) mism++;
            }
        }
    if (nb != z->nbad) mism++;
    return mism;
}

/* per-vertex counts: out[c*n+v] = #K_{k_c} of colour c that contain v.  Returns 0 ok, 1 not
   finished within tlim seconds (tlim < 0: no limit; out is void then), -1 bad input. */
RZ_API int rz_vertex_counts(int n, int r, const int8_t *col, const int *k, double tlim, ll *out) {
    double wt[MAXR] = {0};
    for (int c = 0; c < r && c < MAXR; c++) wt[c] = 1.0;
    RZ *z = (RZ *)rz_create(n, r, k, wt, 1);
    if (!z) { for (int i = 0; i < r * n && r <= MAXR; i++) out[i] = -1; return -1; }
    int optr = 0;
    BG bg;
    bg_init(&bg, now_sec(), tlim);
    rz_set_orbits(z, 0, &optr, NULL, 0, 0, NULL, NULL);
    /* load the graph only: the per-vertex counts below are all that is needed */
    int ok = 1;
    for (int u = 0; u < n && ok; u++)
        for (int v = 0; v < n; v++) {
            if (u == v) continue;
            int c = col[u * n + v];
            if (c < 0 || c >= r || col[v * n + u] != c) { ok = 0; break; }
            setb(ROW(z, c, u), v);
        }
    if (!ok) { for (int i = 0; i < r * n; i++) out[i] = -1; rz_free(z); return -1; }
    for (int c = 0; c < r && !bg.out; c++)
        for (int v = 0; v < n && !bg.out; v++)
            out[c * n + v] = count_cliques(ADJC(z, c), z->W, ROW(z, c, v), z->k[c] - 1, &bg);
    int out_of_time = bg.out;
    rz_free(z);
    return out_of_time ? 1 : 0;
}

/* ---------------- EDGE-mode focused tabu search ---------------- */
/* recolour every edge of orbit o to b, one edge at a time (keeps cnt and bad[] exact) */
static void orbit_apply(RZ *z, int o, int b) {
    int n = z->n;
    for (int t = z->optr[o]; t < z->optr[o + 1]; t++) {
        int e = z->oe[t];
        edge_apply(z, e / n, e % n, b);
    }
    z->ocol[o] = (int8_t)b;
}

/* exact score change of recolouring orbit o (all edges colour a) to b.  Edges are flipped
   one at a time: the cliques lost at edge i are those of colour a through it that survived
   the earlier flips, the cliques gained are those of colour b through it including the
   earlier flips.  The colouring is restored before returning (also when the budget runs
   out; the value is void then). */
static double orbit_delta(RZ *z, int o, int b, BG *bg) {
    int n = z->n, a = z->ocol[o], t0 = z->optr[o], t1 = z->optr[o + 1];
    double d = 0;
    for (int t = t0; t < t1; t++) {
        int e = z->oe[t], u = e / n, v = e % n;
        d -= z->wt[a] * (double)edge_cnt(z, u, v, a, bg);
        recolor(z, u, v, b);
        d += z->wt[b] * (double)edge_cnt(z, u, v, b, bg);
    }
    for (int t = t0; t < t1; t++) { int e = z->oe[t]; recolor(z, e / n, e % n, a); }
    return d;
}

/* random recolourings.  Each is costed first (interruptibly): applying a move takes about
   as long as evaluating it, and a random move can create a great many cliques. */
static void edge_kick(RZ *z, int kick, BG *bg) {
    for (int t = 0; t < kick && z->V > 0; t++) {
        int o = (int)rng_below(&z->rng, z->V);
        int a = z->ocol[o], b = (a + 1 + (int)rng_below(&z->rng, z->r - 1)) % z->r;
        if (bg) { orbit_delta(z, o, b, bg); if (bg->out) return; }
        orbit_apply(z, o, b);
    }
}

/* Local search on orbit variables with exact single-edge bookkeeping.  Orbits may have any
   size (singletons = plain edge search); edges outside every orbit are frozen.
   Only orbits that contain an edge of a violated clique are ever recoloured: recolouring any
   other orbit cannot remove a violation.
     mode 0  focused tabu: pick a random violated clique, recolour the best non-tabu orbit
             among its edges (+ nsample random orbits)
     mode 1  sampled tabu: best non-tabu move among nsample random "bad" orbits (all of them
             if nsample <= 0: steepest descent over the whole violated neighbourhood)
     mode 2  annealing: propose a random bad orbit and colour, accept by the Metropolis rule
             at temperature temp, cooled geometrically to temp/50 over `stall` iterations and
             then reheated
     mode 3  mixed tabu: each step is focused (mode 0) with probability temp, else sampled
             (mode 1).  Sampled steps reach lower plateaus, focused steps close the last few
             violations; the mix does both.
   stats out: [0]=best score, [1]=iterations, [2]=seconds, [3]=solved(0/1) */
RZ_API int rz_search_edge(void *h, ll maxit, double tlim, int tenure, int trand, double noise,
                   int stall, int kick, int nsample, int mode, double temp, double *stats) {
    RZ *z = (RZ *)h;
    int n = z->n, r = z->r, V = z->V;
    if (z->sym) return -1;
    if (!z->ready) return -5;
    double t0 = now_sec();
    BG bg;
    bg_init(&bg, t0, tlim);
    double sc = score_of(z);
    z->bestscore = sc;
    memcpy(z->bestcol, z->col, (size_t)n * n);
    if (!z->bad_ok && !build_bad(z, &bg)) {
        /* out of time before the first move: the colouring is unchanged */
        stats[0] = sc; stats[1] = 0; stats[2] = now_sec() - t0; stats[3] = solved(z) ? 1 : 0;
        return 0;
    }
    for (int o = 0; o < V; o++) z->tabu[o] = 0;
    ll it = 0, last_imp = 0;
    int fails = 0;
    int verts[MAXK + 2];
    int ncap = MAXK * MAXK + (nsample > 0 ? nsample : 0) + 8;
    if (mode != 0 && nsample <= 0) ncap = V + MAXK * MAXK + 8;
    int *cand = (int *)malloc(sizeof(int) * ncap);
    double T = temp, cool = (stall > 0) ? pow(0.02, 1.0 / (double)stall) : 1.0;
    while (it < maxit && !solved(z)) {
        if (tlim >= 0 && (mode != 2 || (it & 15) == 0) && now_sec() - t0 > tlim) break;  /* tlim < 0: unlimited */
        it++;
        if (z->nbad == 0) break;
        if (mode == 2) {                       /* ---- annealing on bad orbits ---- */
            int e = z->badlist[rng_below(&z->rng, z->nbad)];
            int o = z->eorb[e];
            if (o < 0) { if (++fails > 100000) break; continue; }
            int a = z->ocol[o], b = (a + 1 + (int)rng_below(&z->rng, r - 1)) % r;
            double d;
            if (z->optr[o + 1] - z->optr[o] == 1) {
                int u = e / n, v = e % n;
                d = -z->wt[a] * (double)edge_cnt(z, u, v, a, &bg) + z->wt[b] * (double)edge_cnt(z, u, v, b, &bg);
            } else d = orbit_delta(z, o, b, &bg);
            if (bg.out) { it--; break; }
            if (d <= 0 || rng_unif(&z->rng) < exp(-d / T)) {
                orbit_apply(z, o, b);
                sc = score_of(z);
                if (sc < z->bestscore - 1e-9) {
                    z->bestscore = sc;
                    memcpy(z->bestcol, z->col, (size_t)n * n);
                }
            }
            T *= cool;
            if (T < 0.02 * temp) T = temp;     /* reheat */
            continue;
        }
        int nc = 0;
        z->stamp++;
        int step = mode;
        if (mode == 3) step = (rng_unif(&z->rng) < temp) ? 0 : 1;
        if (step == 1) {                       /* ---- candidates from the bad list ---- */
            if (nsample <= 0 || nsample >= z->nbad) {
                for (int q = 0; q < z->nbad; q++) {
                    int o = z->eorb[z->badlist[q]];
                    if (o >= 0 && z->mark[o] != z->stamp) { z->mark[o] = z->stamp; cand[nc++] = o; }
                }
            } else {
                for (int s = 0; s < nsample; s++) {
                    int o = z->eorb[z->badlist[rng_below(&z->rng, z->nbad)]];
                    if (o >= 0 && z->mark[o] != z->stamp) { z->mark[o] = z->stamp; cand[nc++] = o; }
                }
            }
        } else {                               /* ---- orbits of one violated clique ---- */
            int m = edge_pick_violation(z, verts, &bg);
            if (bg.out) { it--; break; }
            if (m == 0) break;
            for (int a = 0; a < m; a++)
                for (int b = a + 1; b < m; b++) {
                    int o = z->eorb[verts[a] * n + verts[b]];
                    if (o >= 0 && z->mark[o] != z->stamp) { z->mark[o] = z->stamp; cand[nc++] = o; }
                }
            for (int s = 0; s < nsample && V > 0; s++) {
                int o = (int)rng_below(&z->rng, V);
                if (z->mark[o] != z->stamp) { z->mark[o] = z->stamp; cand[nc++] = o; }
            }
        }
        if (nc == 0) { if (++fails > 1000) break; continue; }
        int bo = -1, bb = -1, to = -1, tb = -1, nties = 0, tties = 0, late = 0;
        double bd = 1e300, td = 1e300;   /* late: out of time mid-step -> take the best move seen so far */
        if (noise > 0 && rng_unif(&z->rng) < noise) {
            bo = cand[rng_below(&z->rng, nc)];
            bb = (z->ocol[bo] + 1 + (int)rng_below(&z->rng, r - 1)) % r;
        } else {
            for (int q = 0; q < nc; q++) {
                /* one evaluation can take long (large k, many cliques): watch the clock here too */
                if (bg.out || (tlim >= 0 && (q & 15) == 15 && now_sec() - t0 > tlim)) { late = 1; break; }
                int o = cand[q], a = z->ocol[o];
                int single = (z->optr[o + 1] - z->optr[o] == 1);
                int e = z->oe[z->optr[o]], u = e / n, v = e % n;
                double base = single ? -z->wt[a] * (double)edge_cnt(z, u, v, a, &bg) : 0.0;
                for (int b = 0; b < r; b++) {
                    if (b == a) continue;
                    double d = single ? base + z->wt[b] * (double)edge_cnt(z, u, v, b, &bg) : orbit_delta(z, o, b, &bg);
                    if (bg.out) break;              /* d is void */
                    int is_tabu = z->tabu[o] > it && !(sc + d < z->bestscore - 1e-9);
                    if (!is_tabu) {
                        if (d < bd - 1e-12) { bd = d; bo = o; bb = b; nties = 1; }
                        else if (fabs(d - bd) <= 1e-12 && rng_below(&z->rng, ++nties) == 0) { bo = o; bb = b; }
                    } else {
                        if (d < td - 1e-12) { td = d; to = o; tb = b; tties = 1; }
                        else if (fabs(d - td) <= 1e-12 && rng_below(&z->rng, ++tties) == 0) { to = o; tb = b; }
                    }
                }
            }
            if (bg.out) late = 1;
            if (bo < 0) { bo = to; bb = tb; }
        }
        if (bo < 0) { if (late) { it--; break; } continue; }
        orbit_apply(z, bo, bb);
        z->tabu[bo] = it + tenure + (trand > 0 ? (ll)rng_below(&z->rng, trand + 1) : 0);
        sc = score_of(z);
        if (sc < z->bestscore - 1e-9) {
            z->bestscore = sc;
            memcpy(z->bestcol, z->col, (size_t)n * n);
            last_imp = it;
        } else if (!late && stall > 0 && it - last_imp > stall) {
            edge_kick(z, kick, &bg);
            sc = score_of(z);
            last_imp = it;
        }
        if (late) break;
    }
    free(cand);
    if (solved(z)) { z->bestscore = 0; memcpy(z->bestcol, z->col, (size_t)n * n); }
    stats[0] = z->bestscore; stats[1] = (double)it; stats[2] = now_sec() - t0; stats[3] = solved(z) ? 1 : 0;
    return 0;
}

/* ---------------- SYM-mode tabu search over orbits ---------------- */
RZ_API int rz_search_sym(void *h, ll maxit, double tlim, int tenure, int trand, double noise,
                  int stall, int kick, int nsample, double *stats) {
    RZ *z = (RZ *)h;
    if (!z->sym) return -1;
    if (!z->ready) return -5;
    int n = z->n, r = z->r, V = z->V;
    double t0 = now_sec();
    BG bg;
    bg_init(&bg, t0, tlim);
    double sc = score_of(z);
    z->bestscore = sc;
    memcpy(z->bestcol, z->col, (size_t)n * n);
    for (int o = 0; o < V; o++) z->tabu[o] = 0;
    ll it = 0, last_imp = 0;
    int *cand = (int *)malloc(sizeof(int) * (V + MAXK * MAXK + 8));
    int *kick_o = (int *)malloc(sizeof(int) * (kick > 0 ? kick : 1));
    int8_t *kick_c = (int8_t *)malloc(kick > 0 ? kick : 1);
    ll *save = (ll *)malloc(sizeof(ll) * ((size_t)r * z->R + r));
    int verts[MAXK + 2];
    while (it < maxit && !solved(z)) {
        if (tlim >= 0 && now_sec() - t0 > tlim) break;  /* tlim < 0: unlimited */
        it++;
        int nc = 0;
        z->stamp++;
        if (nsample <= 0 || nsample >= V) {
            for (int o = 0; o < V; o++) cand[nc++] = o;
        } else {
            int m = sym_pick_violation(z, verts, &bg);
            if (bg.out) { it--; break; }
            for (int a = 0; a < m; a++)
                for (int b = a + 1; b < m; b++) {
                    int o = z->eorb[verts[a] * n + verts[b]];
                    if (o >= 0 && z->mark[o] != z->stamp) { z->mark[o] = z->stamp; cand[nc++] = o; }
                }
            for (int s = 0; s < nsample; s++) {
                int o = (int)rng_below(&z->rng, V);
                if (z->mark[o] != z->stamp) { z->mark[o] = z->stamp; cand[nc++] = o; }
            }
        }
        int bo = -1, bb = -1, to = -1, tb = -1, nt = 0, tt = 0, late = 0;
        double bs = 1e300, ts = 1e300;   /* late: out of time mid-step -> take the best move seen so far */
        if (noise > 0 && rng_unif(&z->rng) < noise) {
            bo = cand[rng_below(&z->rng, nc)];
            bb = (z->ocol[bo] + 1 + (int)rng_below(&z->rng, r - 1)) % r;
        } else {
            for (int q = 0; q < nc; q++) {
                /* every evaluation recounts two colours: watch the clock here too */
                if (bg.out || (tlim >= 0 && (q & 3) == 3 && now_sec() - t0 > tlim)) { late = 1; break; }
                int o = cand[q], a = z->ocol[o];
                for (int b = 0; b < r; b++) {
                    if (b == a) continue;
                    orbit_recolor(z, o, b);
                    ll na = sym_count_color(z, a, NULL, &bg), nb = sym_count_color(z, b, NULL, &bg);
                    orbit_recolor(z, o, a);
                    if (bg.out) break;              /* na, nb are void */
                    double ns = sc - z->wt[a] * (double)(z->cnt[a] - na) - z->wt[b] * (double)(z->cnt[b] - nb);
                    int is_tabu = z->tabu[o] > it && !(ns < z->bestscore - 1e-9);
                    if (!is_tabu) {
                        if (ns < bs - 1e-12) { bs = ns; bo = o; bb = b; nt = 1; }
                        else if (fabs(ns - bs) <= 1e-12 && rng_below(&z->rng, ++nt) == 0) { bo = o; bb = b; }
                    } else {
                        if (ns < ts - 1e-12) { ts = ns; to = o; tb = b; tt = 1; }
                        else if (fabs(ns - ts) <= 1e-12 && rng_below(&z->rng, ++tt) == 0) { to = o; tb = b; }
                    }
                }
            }
            if (bg.out) late = 1;
            if (bo < 0) { bo = to; bb = tb; }
        }
        if (bo < 0) { if (late) { it--; break; } continue; }
        int a = z->ocol[bo];
        orbit_recolor(z, bo, bb);                   /* never interrupted: costs what its evaluation did */
        z->cnt[a] = sym_count_color(z, a, z->Lrep + (size_t)a * z->R, NULL);
        z->cnt[bb] = sym_count_color(z, bb, z->Lrep + (size_t)bb * z->R, NULL);
        z->tabu[bo] = it + tenure + (trand > 0 ? (ll)rng_below(&z->rng, trand + 1) : 0);
        sc = score_of(z);
        if (sc < z->bestscore - 1e-9) {
            z->bestscore = sc;
            memcpy(z->bestcol, z->col, (size_t)n * n);
            last_imp = it;
        } else if (!late && stall > 0 && it - last_imp > stall) {
            /* kick: random recolourings, then a recount that may be interrupted (a random
               colouring can hold a great many cliques); if it is, the kick is taken back */
            memcpy(save, z->Lrep, sizeof(ll) * (size_t)r * z->R);
            for (int c = 0; c < r; c++) save[(size_t)r * z->R + c] = z->cnt[c];
            int nk = 0;
            for (int t = 0; t < kick; t++) {
                int o = (int)rng_below(&z->rng, V);
                kick_o[nk] = o; kick_c[nk] = z->ocol[o]; nk++;
                orbit_recolor(z, o, (z->ocol[o] + 1 + (int)rng_below(&z->rng, r - 1)) % r);
            }
            sym_all_counts(z, &bg);
            if (bg.out) {
                while (nk > 0) { nk--; orbit_recolor(z, kick_o[nk], kick_c[nk]); }
                memcpy(z->Lrep, save, sizeof(ll) * (size_t)r * z->R);
                for (int c = 0; c < r; c++) z->cnt[c] = save[(size_t)r * z->R + c];
                break;
            }
            sc = score_of(z);
            last_imp = it;
        }
        if (late) break;
    }
    free(cand); free(kick_o); free(kick_c); free(save);
    if (solved(z)) { z->bestscore = 0; memcpy(z->bestcol, z->col, (size_t)n * n); }
    stats[0] = z->bestscore; stats[1] = (double)it; stats[2] = now_sec() - t0; stats[3] = solved(z) ? 1 : 0;
    return 0;
}

/* =====================================================================================
 * EXACT (complete) search: finds a valid colouring or PROVES that none exists (n <= 64).
 *
 * Edges are coloured in a fixed order: block w = 1..n-1, entries u = 0..w-1 (edge uw), so
 * a colouring is the string S = block_1 block_2 ... block_{n-1}, block_w = (c(0,w),...,c(w-1,w)).
 * Each rule only discards colourings that are invalid, or that are not the lexicographic
 * minimum of S over all vertex relabellings and all permutations among colours with equal k.
 * That minimum is valid whenever the original colouring is, and it passes every rule, so
 * exhausting the tree proves that no valid colouring exists.
 *  (1) clique: colouring uw with c must not close a K_{k_c} among already coloured edges.
 *  (2) vertices: for 1 <= v < w, the first v entries of block w are >=lex block v.  If not,
 *      swapping vertices v and w keeps blocks 1..v-1 and makes block v smaller.
 *  (3) colours: among colours with equal k, first appearances in S are in increasing order.
 *      If not, swapping the two colours makes S smaller at the first appearance.
 *      (2) and (3) hold together at the global minimum over both kinds of relabelling.
 *  (4) degrees: d_c(v) <= dmax[c] = U(k with k_c - 1) - 1 for a proven upper bound U, hence
 *      d_c(v) >= dmin[c] = (n-1) - sum_{j != c} dmax[j]; if dmin = dmax = d with n*d odd, no
 *      colouring exists (handshake lemma).
 * flags: 1 = rule (2), 2 = rule (3), 4 = rule (4).  Rule (1) is always on.
 * ===================================================================================== */
#define RX_MAXN 64
#define RX_MAXE (RX_MAXN * (RX_MAXN - 1) / 2)

typedef struct {
    int n, r, E, flags;
    int k[MAXR], dmax[MAXR], dmin[MAXR], prevsame[MAXR];
    u64 adj[MAXR][RX_MAXN];
    int deg[MAXR][RX_MAXN];
    int cnt[MAXR];
    int8_t col[RX_MAXN * RX_MAXN];
    short pu[RX_MAXE], pw[RX_MAXE];
    int8_t cur[RX_MAXE + 1];
    u64 tight[RX_MAXE + 1];
    int p, maxp, status, reason;   /* status: 0 running, 1 found, 2 proven impossible */
    ll nodes;                      /* reason (status 2): 0 search, 1 degree bound, 2 parity */
} RX;

static inline u64 ge_mask(int a) { return a >= 64 ? 0 : (~0ULL << a); }

static int rx_has_clique(const u64 *adj, u64 S, int j) {
    if (j <= 0) return 1;
    if (popc(S) < j) return 0;
    if (j == 1) return S != 0;
    while (S) {
        if (popc(S) < j) return 0;
        int v = ctz64(S);
        S &= S - 1;
        if (rx_has_clique(adj, S & adj[v], j - 1)) return 1;
    }
    return 0;
}

static inline void rx_assign(RX *x, int u, int w, int c) {
    x->col[u * x->n + w] = x->col[w * x->n + u] = (int8_t)c;
    x->adj[c][u] |= 1ULL << w;
    x->adj[c][w] |= 1ULL << u;
    x->deg[c][u]++; x->deg[c][w]++; x->cnt[c]++;
}

static inline void rx_unassign(RX *x, int u, int w, int c) {
    x->col[u * x->n + w] = x->col[w * x->n + u] = -1;
    x->adj[c][u] &= ~(1ULL << w);
    x->adj[c][w] &= ~(1ULL << u);
    x->deg[c][u]--; x->deg[c][w]--; x->cnt[c]--;
}

/* smallest admissible colour >= from for edge uw at position p, or -1 */
static int rx_next(RX *x, int p, int u, int w, int from) {
    int lo = from;
    if (x->flags & 1) {
        u64 T = x->tight[p];
        while (T) {
            int v = ctz64(T);
            T &= T - 1;
            int cv = x->col[u * x->n + v];
            if (cv > lo) lo = cv;
        }
    }
    for (int c = lo; c < x->r; c++) {
        if (x->cnt[c] == 0 && x->prevsame[c] >= 0 && x->cnt[x->prevsame[c]] == 0) continue;
        if (x->deg[c][u] >= x->dmax[c] || x->deg[c][w] >= x->dmax[c]) continue;
        int j = x->k[c] - 2;
        if (j <= 0) continue; /* k_c <= 2: colour c can never be used */
        if (rx_has_clique(x->adj[c], x->adj[c][u] & x->adj[c][w], j)) continue;
        return c;
    }
    return -1;
}

RZ_API void *rx_create(int n, int r, const int *k, const int *dmax, int flags) {
    if (n < 1 || n > RX_MAXN || r < 1 || r > MAXR) return NULL;
    RX *x = (RX *)calloc(1, sizeof(RX));
    if (!x) return NULL;
    x->n = n; x->r = r; x->flags = flags; x->E = n * (n - 1) / 2;
    for (int c = 0; c < r; c++) {
        x->k[c] = k[c];
        x->dmax[c] = (flags & 4) ? dmax[c] : n - 1;
        if (x->dmax[c] < 0) x->dmax[c] = 0;
        x->prevsame[c] = -1;
        if (flags & 2)
            for (int j = c - 1; j >= 0; j--)
                if (k[j] == k[c]) { x->prevsame[c] = j; break; }
    }
    for (int c = 0; c < r; c++) {
        int s = 0;
        for (int j = 0; j < r; j++) if (j != c) s += x->dmax[j];
        x->dmin[c] = (flags & 4) && (n - 1 - s > 0) ? n - 1 - s : 0;
    }
    for (int i = 0; i < n * n; i++) x->col[i] = -1;
    int p = 0;
    for (int w = 1; w < n; w++)
        for (int u = 0; u < w; u++) { x->pu[p] = (short)u; x->pw[p] = (short)w; p++; }
    x->p = 0;
    x->cur[0] = -1;
    if (flags & 4) {
        for (int c = 0; c < r; c++) {
            if (x->dmin[c] > x->dmax[c]) { x->status = 2; x->reason = 1; }
            else if (x->status == 0 && x->dmin[c] == x->dmax[c] && (x->dmax[c] & 1) && (n & 1)) {
                x->status = 2; x->reason = 2;
            }
        }
    }
    return x;
}

RZ_API void rx_free(void *h) { free(h); }

/* Runs until a colouring is found (returns 1), impossibility is proven (2), or the node or
   time budget is used up (0; call again to resume).  max_nodes <= 0 / tlim < 0: no limit.
   stats: [nodes, position, deepest position, positions total, seconds, reason] */
RZ_API int rx_run(void *h, ll max_nodes, double tlim, double *stats) {
    RX *x = (RX *)h;
    double t0 = now_sec();
    ll start = x->nodes, iters = 0;
    while (x->status == 0) {
        if (x->p >= x->E) { x->status = 1; break; }
        if (x->p < 0) { x->status = 2; x->reason = 0; break; }
        if ((++iters & 1023) == 0) {
            if (max_nodes > 0 && x->nodes - start >= max_nodes) break;
            if (tlim >= 0 && now_sec() - t0 > tlim) break;
        }
        int p = x->p, u = x->pu[p], w = x->pw[p];
        int old = x->cur[p];
        if (old >= 0) rx_unassign(x, u, w, old);
        else if (u == 0) x->tight[p] = ((1ULL << w) - 1) & ~1ULL; /* v in [1, w-1] (w <= 63) */
        int c = rx_next(x, p, u, w, old + 1);
        if (c < 0) { x->cur[p] = -1; x->p = p - 1; continue; }
        rx_assign(x, u, w, c);
        x->cur[p] = (int8_t)c;
        x->nodes++;
        if (x->flags & 4) { /* u and w must still be able to reach every lower degree bound */
            int rem_u = x->n - 1 - w, rem_w = (w - 1 - u) + (x->n - 1 - w), ok = 1;
            for (int cc = 0; cc < x->r; cc++)
                if (x->deg[cc][u] + rem_u < x->dmin[cc] || x->deg[cc][w] + rem_w < x->dmin[cc]) { ok = 0; break; }
            if (!ok) continue; /* stay at p: the next round tries the next colour */
        }
        if (u == w - 1 && (x->flags & 4)) { /* vertex w complete: check degree lower bounds */
            int rem = x->n - 1 - w, ok = 1;
            for (int v = 0; v <= w && ok; v++)
                for (int cc = 0; cc < x->r; cc++)
                    if (x->deg[cc][v] + rem < x->dmin[cc]) { ok = 0; break; }
            if (!ok) continue; /* stay at p: the next round tries the next colour */
        }
        if (u + 1 < w) {
            u64 T = x->tight[p], T2 = 0;
            while (T) {
                int v = ctz64(T);
                T &= T - 1;
                if (x->col[u * x->n + v] == c) T2 |= 1ULL << v;
            }
            x->tight[p + 1] = T2 & ge_mask(u + 2);
        }
        x->p = p + 1;
        x->cur[x->p] = -1;
        if (x->p > x->maxp) x->maxp = x->p;
    }
    stats[0] = (double)x->nodes; stats[1] = x->p; stats[2] = x->maxp; stats[3] = x->E;
    stats[4] = now_sec() - t0; stats[5] = x->reason;
    return x->status;
}

/* the colouring found (valid only after rx_run returned 1); diagonal = -1 */
RZ_API void rx_get(void *h, int8_t *out) {
    RX *x = (RX *)h;
    memcpy(out, x->col, (size_t)x->n * x->n);
    for (int i = 0; i < x->n; i++) out[i * x->n + i] = -1;
}

/* =====================================================================================
 * ORBIT search (complete): colour the edge orbits of any space one at a time.
 *
 * Works for every space: symmetric (orbits of a group; the partial colouring stays
 * invariant), extension (frozen base graph, singleton orbits at the new vertex) and full.
 * It either finds a valid colouring of the space or exhausts it, which PROVES that the
 * space contains none.
 *
 *  - value (o, c) is feasible iff adding orbit o to colour c closes no K_{k_c}.  Any new
 *    clique contains an edge of o, and the group maps that edge to the orbit's first edge
 *    (u, v) while preserving the colouring, so it suffices to test whether the common
 *    colour-c neighbourhood of u and v (with o added) contains a K_{k_c - 2}.
 *  - forward checking: after an assignment to colour c every unassigned orbit re-tests c;
 *    an empty domain is a conflict.  Domains are therefore always exact.
 *  - variable order: smallest domain first (ties: random rank, fixed per restart).
 *  - value order: the colour with the smallest forbidden clique first (optional, default in
 *    the Python wrapper): fix the sparse, tightly constrained colour and let the rest follow.
 *  - colour symmetry: among colours with equal k that have no edge at all (assigned or
 *    frozen), only the lowest-numbered one is tried; such colours are interchangeable.
 *  - optional degree caps dmax[c] (valid for every colouring, see the exact search above).
 * ===================================================================================== */
typedef struct {
    int n, W, r, V;
    int k[MAXR], dmax[MAXR], prevsame[MAXR];
    ll used[MAXR];            /* edges (frozen + assigned) per colour */
    u64 *adj;                 /* r * n * W */
    int *optr, *oe;
    int8_t *val;              /* colour of each orbit, -1 = unassigned */
    uint8_t *dom, *dom0;      /* feasible colours per orbit (bitmask); dom0 = at the root */
    int *rank;                /* tie-break rank per orbit */
    int vorder[MAXR];
    int *stk_o, *stk_trail;
    int8_t *stk_c;
    uint8_t *stk_tried;
    int *trail_o;
    uint8_t *trail_m;
    int ntrail, trail_cap;
    int depth, maxdepth, fresh;
    int flags;                /* 1 random order, 2 phase saving, 4 conflict-driven variable order */
    int8_t *phase;            /* last colour each orbit had (kept across restarts) */
    double *act, act_inc;     /* conflict activity per orbit (kept across restarts) */
    int status;               /* 0 running, 1 found, 2 exhausted: no colouring in this space */
    ll nodes, run_nodes;
    int8_t *col0;             /* n*n frozen colours (-1 where free) */
    rng_t rng;
} OX;

static inline u64 *OXROW(OX *x, int c, int v) { return x->adj + ((size_t)c * x->n + v) * x->W; }

static int has_clique(const u64 *adj, int W, const u64 *S, int j) {
    if (j <= 0) return 1;
    int sz = 0;
    for (int i = 0; i < W; i++) sz += popc(S[i]);
    if (sz < j) return 0;
    if (j == 1) return 1;
    if (j >= CB_MIN_J) {
        int order[64 * MAXW], cls[64 * MAXW], nc;
        int m = greedy_colour(adj, W, S, order, cls, &nc);
        if (nc < j) return 0;
        u64 R[MAXW];
        memcpy(R, S, sizeof(u64) * W);
        for (int i = m - 1; i >= 0; i--) {
            if (cls[i] < j) break;
            int v = order[i];
            R[v >> 6] &= ~(1ULL << (v & 63));
            const u64 *av = adj + (size_t)v * W;
            u64 S2[MAXW];
            for (int w = 0; w < W; w++) S2[w] = R[w] & av[w];
            if (has_clique(adj, W, S2, j - 1)) return 1;
        }
        return 0;
    }
    u64 T[MAXW];
    memcpy(T, S, sizeof(u64) * W);
    int rem = sz;
    for (int i = 0; i < W; i++) {
        while (T[i]) {
            if (rem < j) return 0;
            int v = i * 64 + ctz64(T[i]);
            T[i] &= T[i] - 1;
            rem--;
            const u64 *av = adj + (size_t)v * W;
            u64 S2[MAXW];
            int s2 = 0;
            for (int w = 0; w < i; w++) S2[w] = 0;
            for (int w = i; w < W; w++) { S2[w] = T[w] & av[w]; s2 += popc(S2[w]); }
            if (j == 2) { if (s2 > 0) return 1; }
            else if (s2 >= j - 1 && has_clique(adj, W, S2, j - 1)) return 1;
        }
    }
    return 0;
}

static void ox_set(OX *x, int o, int c, int on) {
    int n = x->n;
    for (int t = x->optr[o]; t < x->optr[o + 1]; t++) {
        int e = x->oe[t], u = e / n, v = e % n;
        if (on) { setb(OXROW(x, c, u), v); setb(OXROW(x, c, v), u); }
        else { clrb(OXROW(x, c, u), v); clrb(OXROW(x, c, v), u); }
    }
}

static int ox_feasible(OX *x, int o, int c) {
    int n = x->n, W = x->W, j = x->k[c] - 2;
    if (j < 1) return 0;                      /* k_c <= 2: colour c may not contain an edge */
    int e = x->oe[x->optr[o]], u = e / n, v = e % n;
    int single = (x->optr[o + 1] - x->optr[o] == 1);
    if (!single) ox_set(x, o, c, 1);
    const u64 *au = OXROW(x, c, u), *av = OXROW(x, c, v);
    u64 S[MAXW];
    for (int w = 0; w < W; w++) S[w] = au[w] & av[w];
    int ok = !has_clique(x->adj + (size_t)c * n * W, W, S, j);
    if (ok && x->dmax[c] < n - 1) {
        int du = single, dv = single;
        for (int w = 0; w < W; w++) { du += popc(au[w]); dv += popc(av[w]); }
        if (du > x->dmax[c] || dv > x->dmax[c]) ok = 0;
    }
    if (!single) ox_set(x, o, c, 0);
    return ok;
}

static void ox_shuffle(OX *x, int randomise) {
    for (int o = 0; o < x->V; o++) x->rank[o] = o;
    for (int c = 0; c < x->r; c++) x->vorder[c] = c;
    if (randomise) {
        for (int o = x->V - 1; o > 0; o--) {
            int j = (int)rng_below(&x->rng, o + 1), t = x->rank[o];
            x->rank[o] = x->rank[j]; x->rank[j] = t;
        }
        for (int c = x->r - 1; c > 0; c--) {
            int j = (int)rng_below(&x->rng, c + 1), t = x->vorder[c];
            x->vorder[c] = x->vorder[j]; x->vorder[j] = t;
        }
    }
    if (x->flags & 24) {                    /* value order by clique size (stable: ties keep their order) */
        int up = (x->flags & 8) != 0;       /* 8: smallest k first, 16: largest k first */
        for (int i = 1; i < x->r; i++) {
            int v = x->vorder[i], j = i - 1;
            while (j >= 0 && (up ? x->k[x->vorder[j]] > x->k[v] : x->k[x->vorder[j]] < x->k[v])) {
                x->vorder[j + 1] = x->vorder[j]; j--;
            }
            x->vorder[j + 1] = v;
        }
    }
}

RZ_API void ox_free(void *h) {
    OX *x = (OX *)h;
    if (!x) return;
    free(x->adj); free(x->optr); free(x->oe); free(x->val); free(x->dom); free(x->dom0);
    free(x->rank); free(x->stk_o); free(x->stk_trail); free(x->stk_c); free(x->stk_tried);
    free(x->trail_o); free(x->trail_m); free(x->col0); free(x->phase); free(x->act);
    free(x);
}

/* base: n*n colours of the frozen edges (ignored for edges that belong to an orbit; may be
   NULL when every edge is free).  dmax: per-colour degree caps or NULL.
   flags: 1 = random variable ties and value order (0: deterministic), 2 = phase saving (try
   the colour an orbit had last time first), 4 = prefer orbits that caused conflicts,
   8 / 16 = try colours with the smallest / largest clique size first (ties: as before).
   Measured on circulant (3,17), (3,19) and (4,10) record spaces: smallest-first finds
   colourings 1.3-4x sooner than a random order, largest-first is 3-8x slower.
   The flags change the order of the search only, never its completeness.
   Returns NULL on bad input. */
RZ_API void *ox_create(int n, int r, const int *k, int V, const int *optr, const int *oedges,
                       const int8_t *base, const int *dmax, int flags, unsigned long long seed) {
    if (n < 1 || n > 64 * MAXW || r < 1 || r > MAXR || V < 0) return NULL;
    OX *x = (OX *)calloc(1, sizeof(OX));
    if (!x) return NULL;
    x->n = n; x->r = r; x->V = V; x->W = (n + 63) / 64;
    int E = optr[V];
    x->adj = (u64 *)calloc((size_t)r * n * x->W, sizeof(u64));
    x->optr = (int *)malloc(sizeof(int) * (V + 1));
    x->oe = (int *)malloc(sizeof(int) * (E > 0 ? E : 1));
    x->val = (int8_t *)malloc(V + 1);
    x->dom = (uint8_t *)malloc(V + 1);
    x->dom0 = (uint8_t *)malloc(V + 1);
    x->rank = (int *)malloc(sizeof(int) * (V + 1));
    x->stk_o = (int *)malloc(sizeof(int) * (V + 1));
    x->stk_trail = (int *)malloc(sizeof(int) * (V + 1));
    x->stk_c = (int8_t *)malloc(V + 1);
    x->stk_tried = (uint8_t *)malloc(V + 1);
    x->trail_cap = (V + 1) * r + 16;
    x->trail_o = (int *)malloc(sizeof(int) * x->trail_cap);
    x->trail_m = (uint8_t *)malloc(x->trail_cap);
    x->col0 = (int8_t *)malloc((size_t)n * n);
    x->phase = (int8_t *)malloc(V + 1);
    x->act = (double *)calloc(V + 1, sizeof(double));
    x->act_inc = 1.0; x->flags = flags;
    memset(x->phase, -1, V + 1);
    memcpy(x->optr, optr, sizeof(int) * (V + 1));
    memset(x->col0, -1, (size_t)n * n);
    rng_seed(&x->rng, seed);
    for (int c = 0; c < r; c++) {
        x->k[c] = k[c];
        x->dmax[c] = (dmax && dmax[c] >= 0 && dmax[c] < n - 1) ? dmax[c] : n - 1;
        x->prevsame[c] = -1;
        for (int j = c - 1; j >= 0; j--)
            if (k[j] == k[c]) { x->prevsame[c] = j; break; }
    }
    /* mark free edges, load frozen ones */
    char *isfree = (char *)calloc((size_t)n * n, 1);
    int bad = 0;
    for (int o = 0; o < V && !bad; o++)
        for (int t = optr[o]; t < optr[o + 1]; t++) {
            int u = oedges[2 * t], v = oedges[2 * t + 1];
            if (u < 0 || v < 0 || u >= n || v >= n || u == v || isfree[u * n + v]) { bad = 1; break; }
            isfree[u * n + v] = isfree[v * n + u] = 1;
            x->oe[t] = u < v ? u * n + v : v * n + u;
        }
    for (int u = 0; u < n && !bad; u++)
        for (int v = u + 1; v < n; v++) {
            if (isfree[u * n + v]) continue;
            int c = base ? base[u * n + v] : -1;
            if (c < 0 || c >= r) { bad = 1; break; }
            x->col0[u * n + v] = x->col0[v * n + u] = (int8_t)c;
            setb(OXROW(x, c, u), v); setb(OXROW(x, c, v), u);
            x->used[c]++;
        }
    free(isfree);
    if (bad) { ox_free(x); return NULL; }
    {   /* a frozen part that already contains a forbidden clique can never be completed */
        u64 ALL[MAXW];
        memset(ALL, 0, sizeof(ALL));
        for (int v = 0; v < n; v++) setb(ALL, v);
        for (int c = 0; c < r; c++)
            if (x->used[c] > 0 && has_clique(x->adj + (size_t)c * n * x->W, x->W, ALL, k[c])) x->status = 2;
    }
    /* root domains */
    for (int o = 0; o < V; o++) {
        x->val[o] = -1;
        uint8_t d = 0;
        for (int c = 0; c < r; c++)
            if (ox_feasible(x, o, c)) d |= (uint8_t)(1u << c);
        x->dom[o] = x->dom0[o] = d;
        if (!d) x->status = 2;
    }
    ox_shuffle(x, flags & 1);
    x->fresh = 1;
    return x;
}

/* forget the current branch and start again from the root with a new random order */
RZ_API void ox_restart(void *h, unsigned long long seed) {
    OX *x = (OX *)h;
    if (x->status == 2) return;            /* an exhausted space stays exhausted */
    for (int o = 0; o < x->V; o++) {
        if (x->val[o] >= 0) {
            int c = x->val[o];
            ox_set(x, o, c, 0);
            x->used[c] -= x->optr[o + 1] - x->optr[o];
            x->val[o] = -1;
        }
        x->dom[o] = x->dom0[o];
    }
    rng_seed(&x->rng, seed);
    ox_shuffle(x, x->flags & 1);
    x->depth = 0; x->ntrail = 0; x->fresh = 1; x->status = 0; x->run_nodes = 0;
}

/* Runs until a colouring is found (1), the space is exhausted (2), or the budget is used
   up (0; call again to resume).  max_nodes <= 0 / tlim < 0: no limit.
   stats: [nodes total, nodes since restart, depth, deepest, V, seconds] */
RZ_API int ox_run(void *h, ll max_nodes, double tlim, double *stats) {
    OX *x = (OX *)h;
    double t0 = now_sec();
    ll start = x->nodes, iters = 0;
    int V = x->V, r = x->r;
    while (x->status == 0) {
        ++iters;                                /* one step can be slow (V clique tests): check every time */
        if (max_nodes > 0 && x->nodes - start >= max_nodes) break;
        if (tlim >= 0 && now_sec() - t0 > tlim) break;
        int d = x->depth;
        if (x->fresh) {                         /* choose the next orbit: smallest domain */
            int bo = -1, bs = 99, br = 0;
            double ba = 0;
            int useact = x->flags & 4;
            for (int o = 0; o < V; o++) {
                if (x->val[o] >= 0) continue;
                int s = popc((u64)x->dom[o]);
                double a = useact ? x->act[o] : 0.0;
                if (s < bs || (s == bs && (a > ba || (a == ba && x->rank[o] < br)))) {
                    bs = s; bo = o; br = x->rank[o]; ba = a;
                }
            }
            if (bo < 0) { x->status = 1; break; }
            x->stk_o[d] = bo; x->stk_c[d] = -1; x->stk_tried[d] = 0; x->stk_trail[d] = x->ntrail;
            x->fresh = 0;
        }
        int o = x->stk_o[d];
        if (x->stk_c[d] >= 0) {                 /* undo the previous value at this depth */
            int c = x->stk_c[d];
            ox_set(x, o, c, 0);
            x->used[c] -= x->optr[o + 1] - x->optr[o];
            x->val[o] = -1; x->stk_c[d] = -1;
            while (x->ntrail > x->stk_trail[d]) { x->ntrail--; x->dom[x->trail_o[x->ntrail]] = x->trail_m[x->ntrail]; }
        }
        int c = -1;
        for (int i = -1; i < r; i++) {
            int cc = i < 0 ? x->phase[o] : x->vorder[i];    /* saved phase first */
            if (i < 0 && (!(x->flags & 2) || cc < 0)) continue;
            if (!((x->dom[o] >> cc) & 1) || ((x->stk_tried[d] >> cc) & 1)) continue;
            int skip = 0;                       /* an unused colour with an unused equal-k colour before it */
            if (x->used[cc] == 0)
                for (int p = x->prevsame[cc]; p >= 0; p = x->prevsame[p])
                    if (x->used[p] == 0) { skip = 1; break; }
            if (skip) continue;
            c = cc; break;
        }
        if (c < 0) {                            /* domain exhausted: backtrack */
            if (d == 0) { x->status = 2; break; }
            x->depth = d - 1;
            continue;
        }
        x->stk_tried[d] |= (uint8_t)(1u << c);
        x->stk_c[d] = (int8_t)c; x->val[o] = (int8_t)c; x->phase[o] = (int8_t)c;
        ox_set(x, o, c, 1);
        x->used[c] += x->optr[o + 1] - x->optr[o];
        x->nodes++; x->run_nodes++;
        int conflict = 0;
        for (int p = 0; p < V; p++) {           /* forward checking on colour c */
            if (x->val[p] >= 0 || !((x->dom[p] >> c) & 1)) continue;
            if (!ox_feasible(x, p, c)) {
                x->trail_o[x->ntrail] = p; x->trail_m[x->ntrail] = x->dom[p]; x->ntrail++;
                x->dom[p] &= (uint8_t)~(1u << c);
                if (!x->dom[p]) {
                    conflict = 1;
                    x->act[p] += x->act_inc; x->act[o] += x->act_inc;
                    x->act_inc *= 1.02;
                    if (x->act_inc > 1e100) {
                        for (int q = 0; q < V; q++) x->act[q] *= 1e-100;
                        x->act_inc *= 1e-100;
                    }
                    break;
                }
            }
        }
        if (conflict) continue;                 /* same depth: undo and try the next value */
        x->depth = d + 1;
        if (x->depth > x->maxdepth) x->maxdepth = x->depth;
        x->fresh = 1;
    }
    stats[0] = (double)x->nodes; stats[1] = (double)x->run_nodes; stats[2] = x->depth;
    stats[3] = x->maxdepth; stats[4] = V; stats[5] = now_sec() - t0;
    return x->status;
}

/* the colouring found (valid after ox_run returned 1) */
RZ_API void ox_get(void *h, int8_t *out) {
    OX *x = (OX *)h;
    int n = x->n;
    memcpy(out, x->col0, (size_t)n * n);
    for (int o = 0; o < x->V; o++)
        for (int t = x->optr[o]; t < x->optr[o + 1]; t++) {
            int e = x->oe[t], u = e / n, v = e % n;
            out[u * n + v] = out[v * n + u] = x->val[o];
        }
    for (int i = 0; i < n; i++) out[i * n + i] = -1;
}
