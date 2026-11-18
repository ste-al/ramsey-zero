"""RamseyZero: zero-knowledge search for Ramsey colourings, with proofs where possible.

    from ramsey_zero import solve
    res = solve([5, 5], 42, time=60)      # a 2-colouring of K_42 with no mono K_5
    res.coloring                          # n x n matrix of colours (-1 on diagonal)
"""
from .core import Engine, count_monochromatic, vertex_counts
from .exact import ExactSearch, OrbitSearch, upper_bound
from .solver import Config, RamseyZero, climb, solve
from .spaces import (abelian_group, block_circulant_space, cayley_space, circulant_space,
                     cyclic_group, extension_space, full_space, metacyclic_group,
                     symmetric_catalogue)
from .verify import graph_of, load_coloring, save_coloring, to_graph6, verify

__version__ = "2.0"

__all__ = ["solve", "climb", "Config", "RamseyZero", "Engine", "ExactSearch", "OrbitSearch",
           "upper_bound", "verify", "save_coloring", "load_coloring", "to_graph6", "graph_of",
           "count_monochromatic", "vertex_counts", "circulant_space", "block_circulant_space",
           "cayley_space", "cyclic_group", "abelian_group", "metacyclic_group",
           "symmetric_catalogue", "full_space", "extension_space"]
