import numpy as np
from config import (V_REST, V_RESET, V_THRESH, BETA, REFRACTORY_FRAMES,
                    BOUND_BETA, BOUND_THRESH, HALTERE_DAMPING,
                    BOUND_DORSAL_W, BOUND_VENTRAL_W, BOUND_TONIC, BOUND_HALTERE, BOUND_GROUND_GAIN,
                    INPUT_NODES, OUTPUT_NODES, GA_POPULATION_SIZE, TOTAL_GENOME_SIZE)

class BatchedPooledBiologicalLIF:
    def __init__(self, num_agents=GA_POPULATION_SIZE):
        self.N = num_agents
        
        # State tensors
        self.gf_v = np.full((self.N, OUTPUT_NODES), V_REST, dtype=np.float32)
        self.refractory_timer = np.zeros(self.N, dtype=np.int32)
        self.voltage_history = []
        
        # Genomes/weights (Batched)
        self.W_dorsal = np.zeros((self.N, 8), dtype=np.float32)
        self.W_ventral = np.zeros((self.N, 8), dtype=np.float32)
        self.I_tonic = np.zeros((self.N, 1), dtype=np.float32)
        self.beta = np.zeros((self.N, 1), dtype=np.float32)
        self.v_thresh = np.zeros((self.N, 1), dtype=np.float32)
        self.haltere_damping = np.zeros((self.N, 1), dtype=np.float32)
        self.ground_gain = np.zeros((self.N, 1), dtype=np.float32)
        
        self.genomes = np.zeros((self.N, TOTAL_GENOME_SIZE), dtype=np.float32)

    def set_genomes(self, list_of_vectors):
        """
        Loads a list of 1D genomes (size 22) into the batched weight arrays.
        """
        for i, genome in enumerate(list_of_vectors):
            self.genomes[i] = genome
            
            # Extract 22 parameters
            self.W_dorsal[i] = genome[0:8]
            self.W_ventral[i] = genome[8:16]
            self.I_tonic[i, 0] = genome[16]
            self.beta[i, 0] = genome[17]
            self.v_thresh[i, 0] = genome[18]
            self.haltere_damping[i, 0] = genome[19]
            self.ground_gain[i, 0] = genome[20]
            # Parameter 21 is unused but could be for future expansion, or we can just ignore it since requested size is 22.
            
    def get_elite_genomes(self, elite_indices):
        return [self.genomes[i].copy() for i in elite_indices]
        
    def reproduce_and_mutate(self, elite_indices, fitness_scores, mut_rate, mut_scale):
        new_genomes = []
        # Elitism
        for idx in elite_indices:
            new_genomes.append(self.genomes[idx].copy())
            
        # Tournament selection
        while len(new_genomes) < self.N:
            tourney = np.random.choice(self.N, 3, replace=False)
            winner_idx = tourney[np.argmax(fitness_scores[tourney])]
            
            child = self.genomes[winner_idx].copy()
            mask = np.random.rand(TOTAL_GENOME_SIZE) < mut_rate
            mutations = np.random.normal(0, mut_scale, TOTAL_GENOME_SIZE)
            child += mask * mutations
            
            # Enforce bounds
            child[0:8] = np.clip(child[0:8], BOUND_DORSAL_W[0], BOUND_DORSAL_W[1])
            child[8:16] = np.clip(child[8:16], BOUND_VENTRAL_W[0], BOUND_VENTRAL_W[1])
            child[16] = np.clip(child[16], BOUND_TONIC[0], BOUND_TONIC[1])
            child[17] = np.clip(child[17], BOUND_BETA[0], BOUND_BETA[1])
            child[18] = np.clip(child[18], BOUND_THRESH[0], BOUND_THRESH[1])
            child[19] = np.clip(child[19], BOUND_HALTERE[0], BOUND_HALTERE[1])
            child[20] = np.clip(child[20], BOUND_GROUND_GAIN[0], BOUND_GROUND_GAIN[1])
            
            new_genomes.append(child)
            
        self.set_genomes(new_genomes)

    def reset_states(self):
        self.gf_v.fill(V_REST)
        self.refractory_timer.fill(0)
        self.voltage_history = []
        
    def step_batch(self, inputs_Nx16, velocities_N, y_positions_N):
        """
        Vectorized LIF simulation step for the entire swarm using 4x4 spatial pooled vision.
        inputs_Nx16: shape (N, 16) - The 4x4 grid flattened
        velocities_N: shape (N,)
        y_positions_N: shape (N,)
        Returns boolean array of shape (N,) indicating flaps.
        """
        active_mask = (self.refractory_timer <= 0)
        self.refractory_timer[~active_mask] -= 1
        self.gf_v[~active_mask] = V_RESET
        
        # Split inputs into Dorsal (top 8) and Ventral (bottom 8)
        dorsal_inputs = inputs_Nx16[:, 0:8]
        ventral_inputs = inputs_Nx16[:, 8:16]
        
        # Compute visual currents
        I_dorsal = np.sum(dorsal_inputs * self.W_dorsal, axis=1, keepdims=True)
        I_ventral = np.sum(ventral_inputs * self.W_ventral, axis=1, keepdims=True)
        
        # If bird vertical velocity is upward, scale excitatory current by haltere damping
        haltere_mask = velocities_N < 0
        I_ventral[haltere_mask] *= self.haltere_damping[haltere_mask]
        
        # Net current
        I_net = I_ventral + I_dorsal + self.I_tonic
        
        # Altitude Recovery Reflex (y > 250 and falling vel > 1.0)
        falling_mask = (y_positions_N > 250.0) & (velocities_N > 1.0)
        # We need a shape (N, 1) to match I_net
        I_altitude = np.maximum(0.0, (y_positions_N[falling_mask] - 250.0) / 70.0) * 1.8
        I_net[falling_mask, 0] += I_altitude
        
        # Update membrane potential
        self.gf_v[active_mask] = (self.gf_v[active_mask] - V_REST) * self.beta[active_mask] + V_REST + I_net[active_mask]
        
        # Ceiling Lockdown
        ceiling_mask = (y_positions_N < 120.0)
        self.gf_v[ceiling_mask, 0] = np.minimum(self.gf_v[ceiling_mask, 0], -65.0)
        
        self.voltage_history.append(self.gf_v.copy())
        
        # Spikes
        spikes = (self.gf_v >= self.v_thresh) & active_mask[:, None]
        spiked_indices = np.where(spikes[:, 0])[0]
        
        self.gf_v[spiked_indices] = V_RESET
        self.refractory_timer[spiked_indices] = REFRACTORY_FRAMES
        
        return spikes[:, 0]
