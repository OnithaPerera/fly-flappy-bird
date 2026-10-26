"""
Genetic algorithm over circuit genomes, plus headless evaluation.

Selection uses the harmonic mean of distance flown over SEEDS_PER_GENERATION
courses. The courses are fresh every generation, so a genome can't climb the
ranking by memorising a particular course. Record-setting genomes are re-scored on
validation courses to pick the champion, and the reported score comes from test
courses that neither selection nor the champion pick ever saw.
"""
import numpy as np
from config import (GA_POPULATION_SIZE, GA_ELITE_COUNT, GA_TOURNAMENT, SEEDS_PER_GENERATION,
                    MUT_RATE, MUT_SCALE, STAGNATION_LIMIT, TEST_SEED_BASE, VALIDATION_SEED_BASE, MAX_FRAMES,
                    ARENA_HEIGHT, SHAPING_BONUS)
from circuit import FlyCircuit, GENE_BOUNDS, GENOME_SIZE, random_genomes, clip_genomes
from world import World
import retina


def harmonic_mean(scores):
    """Row-wise harmonic mean of non-negative scores; zero if any score is zero."""
    s = np.asarray(scores, dtype=np.float64)
    inv = np.where(s > 0, 1.0 / np.maximum(s, 1e-12), np.inf)
    return s.shape[1] / inv.sum(axis=1)


def run_episode(genomes, seed, max_frames=MAX_FRAMES):
    """Fly every genome on one course. Returns the finished World."""
    world = World(len(genomes), seed, max_frames)
    brain = FlyCircuit(genomes)
    while not world.done:
        near = retina.sample(world.y, world.obstacles())
        world.step(brain.step(near, world.vy))
    return world


def episode_fitness(world):
    """
    Distance flown, plus a small bonus for dying close to the next gap so that early
    generations, which all die at the first pipe, can still be ranked. The bonus uses
    the game's ground truth; it shapes selection only and is never shown to the fly.
    """
    near_gap = np.clip(1.0 - world.gap_error / (ARENA_HEIGHT / 2), 0.0, 1.0)
    return world.distance + SHAPING_BONUS * np.where(world.alive, 1.0, near_gap)


def evaluate(genomes, seeds, max_frames=MAX_FRAMES):
    """
    Fitness (N, len(seeds)) and pipes passed (N, len(seeds)).

    All courses run in lockstep through one circuit batch of N * len(seeds) flies,
    which amortises the per-frame Python overhead.
    """
    genomes = np.asarray(genomes)
    n = len(genomes)
    worlds = [World(n, seed, max_frames) for seed in seeds]
    brain = FlyCircuit(np.tile(genomes, (len(seeds), 1)))
    near = np.zeros((n * len(seeds), retina.N_OMMATIDIA))
    vy = np.zeros(n * len(seeds))
    while not all(w.done for w in worlds):
        for j, w in enumerate(worlds):
            if not w.done:
                near[j * n:(j + 1) * n] = retina.sample(w.y, w.obstacles())
                vy[j * n:(j + 1) * n] = w.vy
        flaps = brain.step(near, vy)
        for j, w in enumerate(worlds):
            w.step(flaps[j * n:(j + 1) * n])
    fit = np.column_stack([episode_fitness(w) for w in worlds])
    pipes = np.column_stack([w.pipes_passed for w in worlds])
    return fit, pipes


def test_seeds(n):
    return [TEST_SEED_BASE + i for i in range(n)]


def validation_seeds(n):
    """Courses for picking the final champion; disjoint from training and test courses."""
    return [VALIDATION_SEED_BASE + i for i in range(n)]


def pick_champion(candidates, n_courses, max_frames=MAX_FRAMES):
    """The candidate with the most pipes on average over the validation courses."""
    _, pipes = evaluate(np.array(candidates), validation_seeds(n_courses), max_frames)
    best = int(np.argmax(pipes.mean(axis=1)))
    return candidates[best], pipes.mean(axis=1)


class Evolution:
    def __init__(self, pop_size=GA_POPULATION_SIZE, seed=0, init=None):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.pop_size = pop_size
        self.population = random_genomes(pop_size, self.rng)
        if init is not None:
            init = np.atleast_2d(init)[:pop_size]
            self.population[:len(init)] = clip_genomes(init)
        self.generation = 1
        self.best_history, self.mean_history = [], []
        self.record = 0.0
        self.champion = None
        self.hall_of_fame = []     # every genome that set a record, oldest first
        self.stagnant = 0
        self.pulses = 0

    def seeds(self):
        """The courses for the current generation (different every generation)."""
        base = (self.seed * 7919 + self.generation) * 1000
        if base >= TEST_SEED_BASE:          # keep clear of the validation and test courses
            base += 10_000_000
        return [base + j for j in range(SEEDS_PER_GENERATION)]

    def tell(self, scores):
        """scores: (pop_size, courses) fitness on this generation's courses. Breeds the next one."""
        fitness = harmonic_mean(scores)
        best = int(np.argmax(fitness))
        self.best_history.append(float(fitness[best]))
        self.mean_history.append(float(fitness.mean()))
        if fitness[best] > self.record:
            self.record = float(fitness[best])
            self.champion = self.population[best].copy()
            self.hall_of_fame.append(self.champion)
            self.stagnant = 0
        else:
            self.stagnant += 1

        pulse = self.stagnant >= STAGNATION_LIMIT
        if pulse:
            self.stagnant = 0
            self.pulses += 1
        self.population = self._breed(fitness, n_random=int(self.pop_size * 0.25) if pulse else 0,
                                      scale=MUT_SCALE * (2.5 if pulse else 1.0))
        self.generation += 1
        return fitness

    def _breed(self, fitness, n_random, scale):
        order = np.argsort(fitness)[::-1]
        children = [self.population[i].copy() for i in order[:GA_ELITE_COUNT]]
        span = GENE_BOUNDS[:, 1] - GENE_BOUNDS[:, 0]
        while len(children) < self.pop_size - n_random:
            a, b = (max(self.rng.choice(self.pop_size, GA_TOURNAMENT, replace=False),
                        key=lambda i: fitness[i]) for _ in range(2))
            mix = self.rng.random(GENOME_SIZE) < 0.5            # uniform crossover
            child = np.where(mix, self.population[a], self.population[b])
            mutate = self.rng.random(GENOME_SIZE) < MUT_RATE
            child = child + mutate * self.rng.normal(0.0, scale, GENOME_SIZE) * span
            children.append(clip_genomes(child))
        if n_random:
            children.extend(random_genomes(n_random, self.rng))
        return np.array(children)

    @property
    def variance(self):
        span = GENE_BOUNDS[:, 1] - GENE_BOUNDS[:, 0]
        return float(np.var(self.population / span, axis=0).mean())


# ---- checkpoints --------------------------------------------------------------

def save_champion(path, genome, **meta):
    from circuit import GENOME_VERSION, GENE_NAMES
    np.savez(path, genome=np.asarray(genome), version=GENOME_VERSION,
             gene_names=np.array(GENE_NAMES), **{k: np.asarray(v) for k, v in meta.items()})


def load_champion(path):
    """Returns the genome, or None if the file is missing or from an older circuit."""
    from circuit import GENOME_VERSION
    try:
        data = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    if int(data.get("version", -1)) != GENOME_VERSION or data["genome"].shape != (GENOME_SIZE,):
        print(f"[checkpoint] {path} was saved by a different circuit version; ignoring it.")
        return None
    return data["genome"]
