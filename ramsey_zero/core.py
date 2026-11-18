"""ctypes wrapper around the C search kernels (compiled automatically on first use)."""
from __future__ import annotations

import ctypes as C
import glob
import hashlib
import os
import platform
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "_core.c")
_WIN = sys.platform.startswith("win")
_BUILD_TAG = "3"   # part of the cache name: bump when build flags change so stale builds are not reused
_DLL_DIRS = []     # keeps os.add_dll_directory() handles alive


def _flag_sets():
    base = ["-O3", "-shared", "-std=gnu99"]
    if _WIN:
        # Link libgcc and winpthread statically, so the DLL needs only KERNEL32 and the C
        # runtime. Python >= 3.8 does not search PATH for a DLL's dependencies, so any
        # MinGW runtime DLL left as a dependency makes loading fail. The non-static
        # fallbacks are safe too: the core uses no winpthread functions any more.
        static = base + ["-static-libgcc", "-static"]
        return [static + ["-march=native"], static, base + ["-march=native"], base]
    base.append("-fPIC")
    return [base + ["-march=native"], base]


def _lib_path() -> str:
    h = hashlib.sha1()
    with open(_SRC, "rb") as f:
        h.update(f.read())
    # -march=native ties the binary to this CPU, so the CPU is part of the name too
    cpu = f"{platform.machine()}|{platform.processor()}"
    h.update(f"|{_BUILD_TAG}|{sys.platform}|{cpu}|{' '.join(_flag_sets()[0])}".encode())
    ext = ".dll" if _WIN else ".so"
    cache = os.environ.get("RAMSEYZERO_CACHE", _HERE)
    return os.path.join(cache, f"_core_{h.hexdigest()[:12]}{ext}")


def _build(path: str) -> None:
    cc = os.environ.get("CC", "gcc")
    last = None
    for flags in _flag_sets():
        tmp = path + f".tmp{os.getpid()}"
        try:
            subprocess.run([cc, *flags, "-o", tmp, _SRC, "-lm"], check=True,
                           capture_output=True, text=True)
            os.replace(tmp, path)
            _remove_stale(path)
            return
        except (subprocess.CalledProcessError, FileNotFoundError) as e:  # try next flag set
            last = e
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    msg = getattr(last, "stderr", "") or str(last)
    raise RuntimeError(f"could not compile {_SRC} with {cc}: {msg}\n"
                       "Install gcc/clang (Linux/macOS) or 64-bit MinGW-w64 gcc (Windows), or set CC.")


def _remove_stale(keep: str) -> None:
    """Delete older builds of the core next to the current one (best effort)."""
    ext = os.path.splitext(keep)[1]
    for old in glob.glob(os.path.join(os.path.dirname(keep), "_core_*" + ext)):
        if os.path.abspath(old) != os.path.abspath(keep):
            try:
                os.remove(old)
            except OSError:      # still loaded by another process (Windows): leave it
                pass


def _open(path: str):
    if not _WIN:
        return C.CDLL(path)
    try:
        return C.CDLL(path)
    except OSError:
        # Fallback for builds that still depend on MinGW runtime DLLs: make the compiler's
        # bin folder (where those DLLs live) visible to the Windows loader, then retry.
        cc = shutil.which(os.environ.get("CC", "gcc"))
        if cc:
            d = os.path.dirname(os.path.realpath(cc))
            try:
                _DLL_DIRS.append(os.add_dll_directory(d))
            except OSError:
                pass
        return C.CDLL(path)


def _diagnose(path: str, err: OSError) -> str:
    lines = [f"could not load the compiled search core {path}: {err}"]
    if _WIN and getattr(err, "winerror", None) == 193:
        lines.append(f"The DLL's architecture does not match this {struct.calcsize('P') * 8}-bit Python. "
                     "Use a 64-bit MinGW-w64 gcc (e.g. MSYS2 UCRT64 or WinLibs x86_64).")
    elif _WIN:
        lines.append('A DLL it depends on is missing. List them with:  objdump -p "<dll>" | findstr "DLL Name"')
    return "\n".join(lines)


def _load():
    path = _lib_path()
    if not os.path.exists(path):
        _build(path)
    try:
        lib = _open(path)
    except OSError:
        try:  # a stale or broken build: rebuild once before giving up
            os.remove(path)
        except OSError:
            pass
        _build(path)
        try:
            lib = _open(path)
        except OSError as e:
            raise RuntimeError(_diagnose(path, e)) from e
    P = C.POINTER
    vp, i, d, ll, ull = C.c_void_p, C.c_int, C.c_double, C.c_longlong, C.c_ulonglong
    sig = {
        "rz_create": (vp, [i, i, P(i), P(d), ull]),
        "rz_free": (None, [vp]),
        "rz_set_weights": (None, [vp, P(d)]),
        "rz_seed": (None, [vp, ull]),
        "rz_set_orbits": (i, [vp, i, P(i), P(i), i, i, P(i), P(i)]),
        "rz_set_coloring": (i, [vp, P(C.c_int8), d]),
        "rz_get_coloring": (None, [vp, P(C.c_int8)]),
        "rz_get_best": (None, [vp, P(C.c_int8)]),
        "rz_get_counts": (None, [vp, P(ll)]),
        "rz_full_counts": (None, [vp, P(ll)]),
        "rz_score": (d, [vp]),
        "rz_count_coloring": (i, [i, i, P(C.c_int8), P(i), d, P(ll)]),
        "rz_vertex_counts": (i, [i, i, P(C.c_int8), P(i), d, P(ll)]),
        "rz_check_bad": (i, [vp]),
        "rz_search_edge": (i, [vp, ll, d, i, i, d, i, i, i, i, d, P(d)]),
        "rz_search_sym": (i, [vp, ll, d, i, i, d, i, i, i, P(d)]),
        "rx_create": (vp, [i, i, P(i), P(i), i]),
        "rx_free": (None, [vp]),
        "rx_run": (i, [vp, ll, d, P(d)]),
        "rx_get": (None, [vp, P(C.c_int8)]),
        "ox_create": (vp, [i, i, P(i), i, P(i), P(i), P(C.c_int8), P(i), i, ull]),
        "ox_free": (None, [vp]),
        "ox_restart": (None, [vp, ull]),
        "ox_run": (i, [vp, ll, d, P(d)]),
        "ox_get": (None, [vp, P(C.c_int8)]),
    }
    for name, (res, args) in sig.items():
        fn = getattr(lib, name)
        fn.restype = res
        fn.argtypes = args
    return lib


LIB = _load()


def _ptr(a: np.ndarray, ctype):
    return a.ctypes.data_as(C.POINTER(ctype))


def _tl(time_limit) -> float:
    """Seconds for the C side: None = no limit (-1); 0 or a past deadline = give up at once."""
    return -1.0 if time_limit is None else max(0.0, float(time_limit))


def count_monochromatic(col: np.ndarray, ks, time_limit=None):
    """Exact number of monochromatic K_{k_c} in each colour c (C bitset counter).

    Counting is quick for good colourings but can take arbitrarily long for bad ones with
    large k (the cliques are listed one by one).  With a time_limit (seconds) the function
    returns None when the count does not finish in time."""
    col = np.ascontiguousarray(col, dtype=np.int8)
    n = col.shape[0]
    k = np.ascontiguousarray(ks, dtype=np.int32)
    out = np.zeros(len(ks), dtype=np.int64)
    rc = LIB.rz_count_coloring(n, len(ks), _ptr(col, C.c_int8), _ptr(k, C.c_int), _tl(time_limit),
                               _ptr(out, C.c_longlong))
    return None if rc == 1 else out


def vertex_counts(col: np.ndarray, ks, time_limit=None):
    """out[c, v] = number of monochromatic K_{k_c} of colour c containing vertex v
    (None if a time_limit in seconds was given and the count did not finish)."""
    col = np.ascontiguousarray(col, dtype=np.int8)
    n = col.shape[0]
    k = np.ascontiguousarray(ks, dtype=np.int32)
    out = np.zeros(len(ks) * n, dtype=np.int64)
    rc = LIB.rz_vertex_counts(n, len(ks), _ptr(col, C.c_int8), _ptr(k, C.c_int), _tl(time_limit),
                              _ptr(out, C.c_longlong))
    return None if rc == 1 else out.reshape(len(ks), n)


@dataclass
class SearchResult:
    best: float
    iters: int
    seconds: float
    solved: bool


class Engine:
    """One search state: n vertices, colours with forbidden clique sizes ks."""

    def __init__(self, n: int, ks, weights=None, seed: int = 0):
        self.n, self.ks, self.r = n, list(ks), len(ks)
        if n > 512:
            raise ValueError("n <= 512 supported (MAXW=8 words)")
        self._k = np.ascontiguousarray(self.ks, dtype=np.int32)
        w = np.ones(self.r) if weights is None else np.asarray(weights, dtype=np.float64)
        self._w = np.ascontiguousarray(w, dtype=np.float64)
        self.h = LIB.rz_create(n, self.r, _ptr(self._k, C.c_int), _ptr(self._w, C.c_double),
                               C.c_ulonglong(seed & 0xFFFFFFFFFFFFFFFF))
        if not self.h:
            raise ValueError("rz_create failed (check n, number of colours)")
        self.space = None
        self.ready = False        # True once a colouring is loaded

    def __del__(self):
        h = getattr(self, "h", None)
        if h:
            LIB.rz_free(h)
            self.h = None

    def seed(self, s: int):
        LIB.rz_seed(self.h, C.c_ulonglong(s & 0xFFFFFFFFFFFFFFFF))

    def set_weights(self, w):
        self._w = np.ascontiguousarray(w, dtype=np.float64)
        LIB.rz_set_weights(self.h, _ptr(self._w, C.c_double))

    def set_space(self, space):
        if space.n != self.n:
            raise ValueError("space size mismatch")
        self.space = space
        reps = space.reps if space.sym else np.zeros(1, np.int32)
        rsz = space.rsz if space.sym else np.zeros(1, np.int32)
        rc = LIB.rz_set_orbits(self.h, space.V, _ptr(space.optr, C.c_int), _ptr(space.oedges, C.c_int),
                               1 if space.sym else 0, len(reps) if space.sym else 0,
                               _ptr(reps, C.c_int), _ptr(rsz, C.c_int))
        if rc != 0:
            raise ValueError(f"rz_set_orbits failed with code {rc}")

    def set_coloring(self, col: np.ndarray, time_limit=None) -> bool:
        """Load a colouring and count its cliques.  Returns True when loaded.  With a
        time_limit (seconds) it returns False if the count did not finish in time; the engine
        then holds no colouring and needs a successful set_coloring before the next search."""
        self._col = np.ascontiguousarray(col, dtype=np.int8)
        rc = LIB.rz_set_coloring(self.h, _ptr(self._col, C.c_int8), _tl(time_limit))
        self.ready = rc == 0
        if rc not in (0, -3):
            raise ValueError(f"rz_set_coloring failed with code {rc} (asymmetric or not invariant)")
        return self.ready

    def coloring(self) -> np.ndarray:
        out = np.empty((self.n, self.n), dtype=np.int8)
        LIB.rz_get_coloring(self.h, _ptr(out, C.c_int8))
        return out

    def best(self) -> np.ndarray:
        out = np.empty((self.n, self.n), dtype=np.int8)
        LIB.rz_get_best(self.h, _ptr(out, C.c_int8))
        return out

    def counts(self) -> np.ndarray:
        out = np.zeros(self.r, dtype=np.int64)
        LIB.rz_get_counts(self.h, _ptr(out, C.c_longlong))
        return out

    def full_counts(self) -> np.ndarray:
        out = np.zeros(self.r, dtype=np.int64)
        LIB.rz_full_counts(self.h, _ptr(out, C.c_longlong))
        return out

    def score(self) -> float:
        return LIB.rz_score(self.h)

    def check_bad(self) -> int:
        """Edge mode self-check: #mismatches between incremental and rebuilt bookkeeping."""
        return LIB.rz_check_bad(self.h)

    def search(self, maxit=10**9, time_limit=1.0, tenure=None, tenure_rand=None, noise=0.0,
               stall=None, kick=None, nsample=None, mode=None, temp=None) -> SearchResult:
        """Local search from the current colouring.  Symmetric spaces: tabu over orbits with
        exact recounts.  Other spaces (full, weak symmetry, extension): see rz_search_edge;
        mode 0 focused tabu, 1 sampled tabu over violated orbits, 2 annealing."""
        sp = self.space
        stats = np.zeros(4)
        if not self.ready:
            raise RuntimeError("no colouring loaded: call set_coloring first (and check its result)")
        time_limit = _tl(time_limit)   # None = unlimited; 0 or a past deadline = stop immediately
        if sp.sym:
            V = sp.V
            tenure = max(1, V // 6) if tenure is None else tenure
            tenure_rand = max(1, V // 6) if tenure_rand is None else tenure_rand
            stall = 20 * V if stall is None else stall
            kick = max(1, V // 10) if kick is None else kick
            nsample = (0 if V <= 80 else 24) if nsample is None else nsample
            rc = LIB.rz_search_sym(self.h, int(maxit), float(time_limit), tenure, tenure_rand,
                                   float(noise), stall, kick, nsample, _ptr(stats, C.c_double))
        else:
            # default: mixed tabu (30% focused steps, 70% best-of-150 sampled steps).  Measured
            # against pure focused tabu, pure sampled tabu and annealing: lowest plateaus on
            # (5,5;42), (4,7;46), (3,9;35), (4,6;33), and 10-20x faster than focused tabu on
            # instances it can solve from scratch, where pure sampled tabu often stalls.
            mode = 3 if mode is None else mode
            tenure = max(4, self.n // 4) if tenure is None else tenure
            tenure_rand = max(2, self.n // 4) if tenure_rand is None else tenure_rand
            stall = 50 * self.n if stall is None else stall
            kick = max(2, self.n // 8) if kick is None else kick
            nsample = (150 if mode in (1, 3) else 0) if nsample is None else nsample
            temp = (0.3 if mode == 3 else 1.0) if temp is None else temp
            rc = LIB.rz_search_edge(self.h, int(maxit), float(time_limit), tenure, tenure_rand,
                                    float(noise), stall, kick, nsample, int(mode), float(temp),
                                    _ptr(stats, C.c_double))
        if rc != 0:
            raise RuntimeError(f"search failed with code {rc}")
        return SearchResult(float(stats[0]), int(stats[1]), float(stats[2]), bool(stats[3]))
