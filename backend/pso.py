from __future__ import annotations
import numpy as np


def pso(objective, bounds, population_size=30, iterations=50, seed=None):
    rng = np.random.default_rng(seed)
    lower, upper = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    positions = rng.uniform(lower, upper, (population_size, len(lower)))
    velocities = rng.uniform(-(upper-lower), upper-lower, positions.shape) * 0.1
    scores = np.array([objective(x) for x in positions])
    personal, personal_scores = positions.copy(), scores.copy()
    best_index = np.argmin(scores)
    global_best, global_score = positions[best_index].copy(), scores[best_index]
    history = [float(global_score)]
    for _ in range(iterations):
        r1, r2 = rng.random(positions.shape), rng.random(positions.shape)
        velocities = 0.7 * velocities + 1.5 * r1 * (personal - positions) + 1.5 * r2 * (global_best - positions)
        positions = np.clip(positions + velocities, lower, upper)
        scores = np.array([objective(x) for x in positions])
        improved = scores < personal_scores
        personal[improved], personal_scores[improved] = positions[improved], scores[improved]
        index = np.argmin(personal_scores)
        if personal_scores[index] < global_score:
            global_best, global_score = personal[index].copy(), float(personal_scores[index])
        history.append(global_score)
    return {"best_position": global_best, "best_fitness": global_score, "history": history}
