from __future__ import annotations
import time
from .pso import pso
from .differential_evolution import differential_evolution
from .gwo import gwo

ALGORITHMS = {"PSO": pso, "DE": differential_evolution, "GWO": gwo}


def run_algorithm(name, objective, bounds, population_size=30, iterations=50, seed=None):
    if name not in ALGORITHMS:
        raise ValueError(f"Unsupported algorithm: {name}")
    started = time.perf_counter()
    result = ALGORITHMS[name](objective, bounds, population_size, iterations, seed)
    result["elapsed_seconds"] = time.perf_counter() - started
    result["algorithm"] = name
    return result
