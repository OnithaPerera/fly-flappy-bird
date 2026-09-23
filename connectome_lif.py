import numpy as np
from config import (V_REST, V_RESET, V_THRESH, BETA, REFRACTORY_FRAMES,
                    BOUND_BETA, BOUND_THRESH, HALTERE_DAMPING,
                    BOUND_DORSAL_W, BOUND_VENTRAL_W, BOUND_TONIC,
                    BOUND_ALT, BOUND_VEL,
                    INPUT_NODES, OUTPUT_NODES, GA_POPULATION_SIZE, TOTAL_GENOME_SIZE)

class VectorizedProprioceptiveSNN:
    def __init__(self, num_agents=GA_POPULATION_SIZE):
        self.N = num_agents
        
        # State tensors
        self.gf_v = np.full((self.N, OUTPUT_NODES), V_REST, dtype=np.float32)
        self.refractory_timer = np.zeros(self.N, dtype=np.int32)
        self.voltage_history = []
        
        # Genomes/weights (Batched)
        self.W_dorsal = np.zeros((self.N, 32), dtype=np.float32)
        self.W_ventral = np.zeros((self.N, 32), dtype=np.float32)
        self.W_alt = np.zeros((self.N, 1), dtype=np.float32)
        self.W_vel = np.zeros((self.N, 1), dtype=np.float32)
        
        self.I_tonic = np.zeros((self.N, 1), dtype=np.float32)
        self.beta = np.zeros((self.N, 1), dtype=np.float32)
        self.v_thresh = np.zeros((self.N, 1), dtype=np.float32)
        self.haltere_damping = np.zeros((self.N, 1), dtype=np.float32)
        
        self.genomes = np.zeros((self.N, TOTAL_GENOME_SIZE), dtype=np.float32)

    def set_genomes(self, list_of_vectors):
        """
        Loads a list of 1D genomes (size 70) into the batched weight arrays.
        """
        for i, genome in enumerate(list_of_vectors):
            self.genomes[i] = genome
            
            # Extract 70 parameters
            self.W_dorsal[i] = genome[0:32]
            self.W_ventral[i] = genome[32:64]
            self.W_alt[i, 0] = genome[64]
            self.W_vel[i, 0] = genome[65]
            
            self.I_tonic[i, 0] = genome[66]
            self.beta[i, 0] = genome[67]
            self.v_thresh[i, 0] = genome[68]
            self.haltere_damping[i, 0] = genome[69]
            
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
            child[0:32] = np.clip(child[0:32], BOUND_DORSAL_W[0], BOUND_DORSAL_W[1])
            child[32:64] = np.clip(child[32:64], BOUND_VENTRAL_W[0], BOUND_VENTRAL_W[1])
            child[64] = np.clip(child[64], BOUND_ALT[0], BOUND_ALT[1])
            child[65] = np.clip(child[65], BOUND_VEL[0], BOUND_VEL[1])
            
            child[66] = np.clip(child[66], BOUND_TONIC[0], BOUND_TONIC[1])
            child[67] = np.clip(child[67], BOUND_BETA[0], BOUND_BETA[1])
            child[68] = np.clip(child[68], BOUND_THRESH[0], BOUND_THRESH[1])
            # We don't have BOUND_HALTERE anymore? No we still need it. Let's add it back if we can.
            # I'll just use a generic bound or [0.1, 0.5] like before. Let's use [0.1, 0.5].
            child[69] = np.clip(child[69], 0.1, 0.5)
            
            new_genomes.append(child)
            
        self.set_genomes(new_genomes)

    def reset_states(self):
        self.gf_v.fill(V_REST)
        self.refractory_timer.fill(0)
        self.voltage_history = []
        
    def step_batch(self, inputs_batch, y_positions_N):
        """
        Vectorized LIF simulation step for the entire swarm.
        inputs_batch: shape (N, 66) - [Shared_Grid_64, Altitudes_40, Velocities_40]
        y_positions_N: shape (N,)
        Returns boolean array of shape (N,) indicating flaps.
        """
        active_mask = (self.refractory_timer <= 0)
        self.refractory_timer[~active_mask] -= 1
        self.gf_v[~active_mask] = V_RESET
        
        # Split inputs
        dorsal_inputs = inputs_batch[:, 0:32]
        ventral_inputs = inputs_batch[:, 32:64]
        alt_inputs = inputs_batch[:, 64:65]
        vel_inputs = inputs_batch[:, 65:66]
        
        # Compute visual currents
        I_dorsal = np.sum(dorsal_inputs * self.W_dorsal, axis=1, keepdims=True)
        I_ventral = np.sum(ventral_inputs * self.W_ventral, axis=1, keepdims=True)
        
        # Compute proprioceptive currents
        I_alt = alt_inputs * self.W_alt
        I_vel = vel_inputs * self.W_vel
        
        # Net current
        I_net = I_ventral + I_dorsal + I_alt + I_vel + self.I_tonic
        
        # Ground Emergency Reflex (y > 420)
        ground_mask = y_positions_N > 420.0
        I_ground = (y_positions_N[ground_mask] - 420.0) * 0.15
        I_net[ground_mask, 0] += I_ground
        
        # Update membrane potential
        self.gf_v[active_mask] = (self.gf_v[active_mask] - V_REST) * self.beta[active_mask] + V_REST + I_net[active_mask]
        
        # Ceiling Lockdown (force V_m to V_REST if y < 90)
        ceiling_mask = (y_positions_N < 90.0)
        self.gf_v[ceiling_mask, 0] = V_REST
        
        self.voltage_history.append(self.gf_v.copy())
        
        # Spikes
        spikes = (self.gf_v >= self.v_thresh) & active_mask[:, None]
        spiked_indices = np.where(spikes[:, 0])[0]
        
        self.gf_v[spiked_indices] = V_RESET
        self.refractory_timer[spiked_indices] = REFRACTORY_FRAMES
        
        return spikes[:, 0]
