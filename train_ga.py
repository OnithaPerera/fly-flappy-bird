import numpy as np
import time
from game import FlappyWorld
from vision import preprocess_frame, compute_looming_stimulus
from connectome_lif import LoomingCircuitController
from config import GA_POPULATION_SIZE, GA_MUTATION_RATE, GA_ELITE_FRACTION, EYE_RES
from config import BOUND_BETA, BOUND_THRESH, BOUND_GAIN, BOUND_WEIGHT

def initialize_population(size):
    population = []
    for _ in range(size):
        # Randomize within bounds
        beta = np.random.uniform(*BOUND_BETA)
        thresh = np.random.uniform(*BOUND_THRESH)
        gain = np.random.uniform(*BOUND_GAIN)
        
        # Base mask perturbed
        weights = np.ones((EYE_RES, EYE_RES), dtype=np.float32)
        weights[0:EYE_RES//2, :] = 0.2
        weights[EYE_RES//2:, :] = 2.0
        
        # Add uniform noise
        weights += np.random.normal(0, 0.5, (EYE_RES, EYE_RES))
        weights = np.clip(weights, BOUND_WEIGHT[0], BOUND_WEIGHT[1])
        
        genome = np.concatenate(([beta, thresh, gain], weights.flatten()))
        
        brain = LoomingCircuitController(genome=genome)
        population.append(brain)
    return population

def mutate(genome):
    mutated = genome.copy()
    for i in range(len(mutated)):
        if np.random.rand() < GA_MUTATION_RATE:
            mutated[i] += np.random.normal(0, 0.2)
            
    # Enforce bounds
    mutated[0] = np.clip(mutated[0], BOUND_BETA[0], BOUND_BETA[1])
    mutated[1] = np.clip(mutated[1], BOUND_THRESH[0], BOUND_THRESH[1])
    mutated[2] = np.clip(mutated[2], BOUND_GAIN[0], BOUND_GAIN[1])
    mutated[3:] = np.clip(mutated[3:], BOUND_WEIGHT[0], BOUND_WEIGHT[1])
    
    return mutated

def run_generation(population, max_frames=2000):
    world = FlappyWorld(num_agents=len(population))
    prev_frames = [None] * len(population)
    
    frames_run = 0
    while not world.all_dead and frames_run < max_frames:
        surface = world.render()
        curr_frame = preprocess_frame(surface)
        
        flaps = []
        for i, brain in enumerate(population):
            if not world.agents[i].alive:
                flaps.append(False)
                continue
                
            drive, _ = compute_looming_stimulus(curr_frame, prev_frames[i], brain.spatial_weights)
            prev_frames[i] = curr_frame
            flaps.append(brain.step(drive))
            
        world.step(flaps)
        frames_run += 1
        
    return world.agents

def train(generations=10):
    print(f"--- Starting Neuroevolution ---")
    print(f"Population: {GA_POPULATION_SIZE} | Generations: {generations}")
    
    population = initialize_population(GA_POPULATION_SIZE)
    
    best_overall_genome = None
    max_overall_score = -1
    
    for gen in range(generations):
        start_time = time.time()
        
        agents_results = run_generation(population)
        
        # Calculate fitness
        fitness_scores = []
        for a in agents_results:
            fit = a.frames_survived + (a.score * 500) - (a.ceiling_hits * 50)
            fitness_scores.append(fit)
            
        # Rank selection
        sorted_indices = np.argsort(fitness_scores)[::-1]
        
        best_idx = sorted_indices[0]
        best_agent = agents_results[best_idx]
        best_genome = population[best_idx].get_genome()
        
        if best_agent.score >= max_overall_score:
            max_overall_score = best_agent.score
            best_overall_genome = best_genome
            np.save("best_fly_genome.npy", best_overall_genome)
            
        avg_fitness = np.mean(fitness_scores)
        
        print(f"Gen {gen+1}/{generations} | Max Score: {best_agent.score} | Avg Fit: {avg_fitness:.1f} | Best Fit: {fitness_scores[best_idx]} | Time: {time.time()-start_time:.2f}s")
        
        # Elitism
        num_elites = max(1, int(GA_ELITE_FRACTION * GA_POPULATION_SIZE))
        next_population = []
        
        for i in range(num_elites):
            elite_genome = population[sorted_indices[i]].get_genome()
            next_population.append(LoomingCircuitController(genome=elite_genome))
            
        # Tournament selection for the rest
        while len(next_population) < GA_POPULATION_SIZE:
            # Pick 3 random, take best
            tourney = np.random.choice(sorted_indices, 3, replace=False)
            winner_idx = max(tourney, key=lambda i: fitness_scores[i])
            parent_genome = population[winner_idx].get_genome()
            
            child_genome = mutate(parent_genome)
            next_population.append(LoomingCircuitController(genome=child_genome))
            
        population = next_population
        
    print("--- Training Complete ---")
    print("Saved best genome to best_fly_genome.npy")
