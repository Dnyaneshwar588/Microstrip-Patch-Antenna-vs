from __future__ import annotations
import numpy as np


def differential_evolution(objective, bounds, population_size=30, iterations=50, seed=None):
    rng = np.random.default_rng(seed)
    lower, upper = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    population = rng.uniform(lower, upper, (population_size, len(lower)))
    scores = np.array([objective(x) for x in population])
    history = [float(scores.min())]
    for _ in range(iterations):
        for i in range(population_size):
            choices = [j for j in range(population_size) if j != i]
            a, b, c = population[rng.choice(choices, 3, replace=False)]
            mutant = np.clip(a + 0.8 * (b - c), lower, upper)
            mask = rng.random(len(lower)) < 0.9
            mask[rng.integers(len(lower))] = True
            trial = np.where(mask, mutant, population[i])
            trial_score = objective(trial)
            if trial_score <= scores[i]:
                population[i], scores[i] = trial, trial_score
        history.append(float(scores.min()))
    index = np.argmin(scores)
    return {"best_position": population[index], "best_fitness": float(scores[index]), "history": history}
