from __future__ import annotations
import numpy as np


def gwo(objective, bounds, population_size=30, iterations=50, seed=None):
    rng = np.random.default_rng(seed)
    lower, upper = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    wolves = rng.uniform(lower, upper, (population_size, len(lower)))
    history = []
    for iteration in range(iterations + 1):
        scores = np.array([objective(wolf) for wolf in wolves])
        order = np.argsort(scores)
        alpha, beta, delta = wolves[order[:3]].copy()
        history.append(float(scores[order[0]]))
        if iteration == iterations:
            break
        a = 2 - 2 * iteration / max(iterations, 1)
        for i in range(population_size):
            candidates = []
            for leader in (alpha, beta, delta):
                r1, r2 = rng.random(len(lower)), rng.random(len(lower))
                A, C = 2 * a * r1 - a, 2 * r2
                candidates.append(leader - A * np.abs(C * leader - wolves[i]))
            wolves[i] = np.clip(np.mean(candidates, axis=0), lower, upper)
    final_scores = np.array([objective(wolf) for wolf in wolves])
    index = np.argmin(final_scores)
    return {"best_position": wolves[index], "best_fitness": float(final_scores[index]), "history": history}
