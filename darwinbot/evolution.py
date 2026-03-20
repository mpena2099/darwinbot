"""
evolution.py
------------
Evolutionary loop using DEAP.

Strategy:
  - Representation : real-valued vector (float list)
  - Selection      : Tournament (size 3)
  - Crossover      : Simulated Binary Crossover (SBX) — good for reals
  - Mutation       : Gaussian perturbation with bounds clipping
  - Replacement    : (μ + λ) — elitism kept via Hall of Fame
  - Parallelism    : multiprocessing.Pool (one process per individual)

Checkpointing: saves population to a .pkl file every N generations
so you can resume after interruption.
"""

import os
import pickle
import multiprocessing
import numpy as np

from deap import base, creator, tools, algorithms

from darwinbot.genome import (
    GENOME_LENGTH, BOUNDS, N_LIMBS, random_genome, clip_genome
)
from darwinbot.fitness import evaluate

# ── DEAP setup ────────────────────────────────────────────────────────────────

# Maximise a single fitness value
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()


def _init_individual():
    return creator.Individual(random_genome().tolist())


toolbox.register("individual", _init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate",   evaluate)

# SBX crossover — well-suited for bounded real-valued genomes
toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                 low=[b[0] for b in BOUNDS * N_LIMBS],
                 up=[b[1]  for b in BOUNDS * N_LIMBS],
                 eta=20.0)

# Gaussian mutation with per-gene sigma = 5% of range
_sigma = [(b[1] - b[0]) * 0.05 for b in BOUNDS] * N_LIMBS
toolbox.register("mutate", tools.mutGaussian,
                 mu=0, sigma=_sigma, indpb=0.15)

toolbox.register("select", tools.selTournament, tournsize=3)


def _clip_individual(ind):
    """Clip individual genes to valid bounds after variation operators."""
    clipped = clip_genome(np.array(ind)).tolist()
    for i in range(len(ind)):
        ind[i] = clipped[i]
    return ind


# ── Main evolution function ───────────────────────────────────────────────────

def run_evolution(
    pop_size:       int   = 60,
    n_generations:  int   = 100,
    cx_prob:        float = 0.6,
    mut_prob:       float = 0.3,
    n_workers:      int   = None,
    checkpoint_dir: str   = "checkpoints",
    checkpoint_freq: int  = 10,
    resume_from:    str   = None,
):
    """
    Run the genetic algorithm.

    Parameters
    ----------
    pop_size       : number of individuals per generation
    n_generations  : number of generations to run
    cx_prob        : crossover probability per pair
    mut_prob       : mutation probability per individual
    n_workers      : CPU workers for parallel eval (None = all cores)
    checkpoint_dir : directory to save checkpoints
    checkpoint_freq: save every N generations
    resume_from    : path to .pkl checkpoint to resume from

    Returns
    -------
    (population, logbook, hall_of_fame)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    print(f"[darwinbot] Starting evolution: "
          f"pop={pop_size}  gen={n_generations}  workers={n_workers}")

    # -- Resume or initialise --
    if resume_from and os.path.exists(resume_from):
        with open(resume_from, "rb") as f:
            ckpt = pickle.load(f)
        population  = ckpt["population"]
        start_gen   = ckpt["generation"] + 1
        hof         = ckpt["hof"]
        logbook     = ckpt["logbook"]
        print(f"[darwinbot] Resumed from generation {ckpt['generation']}")
    else:
        population = toolbox.population(n=pop_size)
        start_gen  = 0
        hof        = tools.HallOfFame(5)
        logbook    = tools.Logbook()
        logbook.header = ["gen", "nevals", "avg", "std", "min", "max"]

    stats = tools.Statistics(lambda ind: ind.fitness.values[0]
                             if ind.fitness.valid else float("nan"))
    stats.register("avg", lambda v: round(float(np.mean(v)), 3))
    stats.register("std", lambda v: round(float(np.std(v)),  3))
    stats.register("min", lambda v: round(float(np.min(v)),  3))
    stats.register("max", lambda v: round(float(np.max(v)),  3))

    # -- Parallel evaluation pool --
    pool = multiprocessing.Pool(processes=n_workers)
    toolbox.register("map", pool.map)

    try:
        # Evaluate initial population (or unevaluated individuals after resume)
        invalid = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid)
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        hof.update(population)
        record = stats.compile(population)
        logbook.record(gen=start_gen, nevals=len(invalid), **record)
        print(logbook.stream)

        for gen in range(start_gen + 1, start_gen + n_generations + 1):
            # -- Selection --
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            # -- Crossover --
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < cx_prob:
                    toolbox.mate(child1, child2)
                    _clip_individual(child1)
                    _clip_individual(child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # -- Mutation --
            for mutant in offspring:
                if np.random.random() < mut_prob:
                    toolbox.mutate(mutant)
                    _clip_individual(mutant)
                    del mutant.fitness.values

            # -- Evaluate new individuals --
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid)
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

            # -- Replace population with elitism --
            population[:] = offspring
            hof.update(population)

            # Elitism: ensure the best individual survives
            if hof:
                worst_idx = min(
                    range(len(population)),
                    key=lambda k: population[k].fitness.values[0]
                    if population[k].fitness.valid else float('inf'),
                )
                elite = toolbox.clone(hof[0])
                population[worst_idx] = elite

            record = stats.compile(population)
            logbook.record(gen=gen, nevals=len(invalid), **record)
            print(logbook.stream)

            # -- Checkpoint --
            if gen % checkpoint_freq == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"gen_{gen:04d}.pkl")
                with open(ckpt_path, "wb") as f:
                    pickle.dump({
                        "generation": gen,
                        "population": population,
                        "hof":        hof,
                        "logbook":    logbook,
                    }, f)
                print(f"[darwinbot] Checkpoint saved: {ckpt_path}")

    finally:
        pool.close()
        pool.join()

    return population, logbook, hof
