"""
Genetic Algorithm core for optimizing Wi-Fi router placement.
Works with member_B_signal_simulation_engine.signal_physics.fitness_function.
"""

from __future__ import annotations

import random
from typing import List, Tuple, Optional, Dict, Any

from member_B_signal_simulation_engine.signal_physics import fitness_function

Cell = Tuple[int, int]  # (x, y)
Individual = List[Cell]


def get_free_cells(grid) -> List[Cell]:
    """Collect all FREE cells as candidate router locations."""
    free_cells: List[Cell] = []
    height = len(grid)
    if height == 0:
        return free_cells
    width = len(grid[0])
    for y in range(height):
        for x in range(width):
            if grid[y][x] == "FREE":
                free_cells.append((x, y))
    return free_cells


def random_individual(
    num_routers: int,
    candidates: List[Cell],
    rng: random.Random,
) -> Individual:
    """Create a random individual with unique router positions."""
    if num_routers > len(candidates):
        raise ValueError("num_routers exceeds number of candidate cells")
    return rng.sample(candidates, num_routers)


def tournament_selection(
    population: List[Individual],
    fitnesses: List[float],
    k: int,
    rng: random.Random,
) -> Individual:
    """Pick the best among k randomly chosen individuals."""
    chosen_indices = rng.sample(range(len(population)), k)
    best_idx = max(chosen_indices, key=lambda i: fitnesses[i])
    return population[best_idx]


def crossover(
    parent_a: Individual,
    parent_b: Individual,
    rng: random.Random,
) -> Tuple[Individual, Individual]:
    """One-point crossover; ensures same length output."""
    if len(parent_a) != len(parent_b):
        raise ValueError("Parents must have same length")
    if len(parent_a) < 2:
        return parent_a[:], parent_b[:]
    cut = rng.randint(1, len(parent_a) - 1)
    child1 = parent_a[:cut] + parent_b[cut:]
    child2 = parent_b[:cut] + parent_a[cut:]
    return child1, child2


def repair_individual(
    indiv: Individual,
    candidates: List[Cell],
    rng: random.Random,
) -> Individual:
    """Ensure all router positions are unique and valid candidates."""
    unique = []
    seen = set()
    for cell in indiv:
        if cell not in seen:
            unique.append(cell)
            seen.add(cell)
    if len(unique) < len(indiv):
        remaining = [c for c in candidates if c not in seen]
        needed = len(indiv) - len(unique)
        if needed > 0:
            unique.extend(rng.sample(remaining, needed))
    return unique


def mutate(
    indiv: Individual,
    candidates: List[Cell],
    mutation_rate: float,
    rng: random.Random,
) -> Individual:
    """Randomly replace router positions with new candidates."""
    new_indiv = indiv[:]
    for i in range(len(new_indiv)):
        if rng.random() < mutation_rate:
            new_indiv[i] = rng.choice(candidates)
    return new_indiv


def evaluate_population(
    population: List[Individual],
    grid,
    beta: float,
) -> List[float]:
    """Compute fitness for each individual."""
    return [fitness_function(indiv, grid, beta=beta) for indiv in population]


def run_ga(
    grid,
    num_routers: int,
    population_size: int = 30,
    generations: int = 50,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
    elite_size: int = 2,
    tournament_k: int = 3,
    beta: float = 0.0,
    candidates: Optional[List[Cell]] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run GA and return best solution with history.

    Returns dict:
      best_routers: List[(x, y)]
      best_fitness: float
      history: List[float] (best fitness per generation)
    """
    rng = random.Random(seed)

    if candidates is None:
        candidates = get_free_cells(grid)
    if not candidates:
        raise ValueError("No candidate cells available")

    population = [
        random_individual(num_routers, candidates, rng)
        for _ in range(population_size)
    ]

    history: List[float] = []
    best_routers: Individual = []
    best_fitness = float("-inf")

    for _ in range(generations):
        fitnesses = evaluate_population(population, grid, beta)

        # Track best
        gen_best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
        gen_best_fit = fitnesses[gen_best_idx]
        if gen_best_fit > best_fitness:
            best_fitness = gen_best_fit
            best_routers = population[gen_best_idx][:]
        history.append(gen_best_fit)

        # Elitism
        elite_indices = sorted(
            range(len(population)), key=lambda i: fitnesses[i], reverse=True
        )[:elite_size]
        new_population = [population[i][:] for i in elite_indices]

        # Produce offspring
        while len(new_population) < population_size:
            parent_a = tournament_selection(population, fitnesses, tournament_k, rng)
            parent_b = tournament_selection(population, fitnesses, tournament_k, rng)

            if rng.random() < crossover_rate:
                child1, child2 = crossover(parent_a, parent_b, rng)
            else:
                child1, child2 = parent_a[:], parent_b[:]

            child1 = mutate(child1, candidates, mutation_rate, rng)
            child2 = mutate(child2, candidates, mutation_rate, rng)
            child1 = repair_individual(child1, candidates, rng)
            child2 = repair_individual(child2, candidates, rng)

            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population

    return {
        "best_routers": best_routers,
        "best_fitness": best_fitness,
        "history": history,
    }


if __name__ == "__main__":
    # Minimal smoke test (uses a tiny dummy grid)
    # grid with "FREE" everywhere
    grid = [["FREE" for _ in range(10)] for _ in range(10)]
    result = run_ga(grid, num_routers=2, generations=10, seed=42)
    print("Best routers:", result["best_routers"])
    print("Best fitness:", result["best_fitness"])
