import numpy as np
from config import (V_REST, V_RESET, V_THRESH, BETA, REFRACTORY_FRAMES,
                    BOUND_BETA, BOUND_THRESH,
                    BOUND_W_CLIMB, BOUND_W_DIVE, BOUND_W_LOOMING,
                    BOUND_W_VEL, BOUND_W_GROUND, BOUND_TONIC,
                    OUTPUT_NODES, GA_POPULATION_SIZE, TOTAL_GENOME_SIZE,
                    GROUND_Y)

class LobulaColumnarSNN:
    def __init__(self, num_agents=GA_POPULATION_SIZE):
        self.N = num_agents
        
        # State tensors
        self.gf_v = np.full((self.N, OUTPUT_NODES), V_REST, dtype=np.float32)
        self.refractory_timer = np.zeros(self.N, dtype=np.int32)
        self.voltage_history = []
        
        # Genomes/weights (Batched)
        self.W_climb = np.zeros((self.N, 1), dtype=np.float32)
        self.W_dive = np.zeros((self.N, 1), dtype=np.float32)
        self.W_looming = np.zeros((self.N, 1), dtype=np.float32)
        self.W_vel = np.zeros((self.N, 1), dtype=np.float32)
        self.W_ground = np.zeros((self.N, 1), dtype=np.float32)
        
        self.I_tonic = np.zeros((self.N, 1), dtype=np.float32)
        self.beta = np.zeros((self.N, 1), dtype=np.float32)
        self.v_thresh = np.zeros((self.N, 1), dtype=np.float32)
        
        self.genomes = np.zeros((self.N, TOTAL_GENOME_SIZE), dtype=np.float32)

    def set_genomes(self, list_of_vectors):
        """
        Loads a list of 1D genomes (size 8) into the batched weight arrays.
        """
        for i, genome in enumerate(list_of_vectors):
            self.genomes[i] = genome
            
            # Extract 8 parameters
            self.W_climb[i, 0] = genome[0]
            self.W_dive[i, 0] = genome[1]
            self.W_looming[i, 0] = genome[2]
            self.W_vel[i, 0] = genome[3]
            self.W_ground[i, 0] = genome[4]
            self.I_tonic[i, 0] = genome[5]
            self.beta[i, 0] = genome[6]
            self.v_thresh[i, 0] = genome[7]
            
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
            child[0] = np.clip(child[0], BOUND_W_CLIMB[0], BOUND_W_CLIMB[1])
            child[1] = np.clip(child[1], BOUND_W_DIVE[0], BOUND_W_DIVE[1])
            child[2] = np.clip(child[2], BOUND_W_LOOMING[0], BOUND_W_LOOMING[1])
            child[3] = np.clip(child[3], BOUND_W_VEL[0], BOUND_W_VEL[1])
            child[4] = np.clip(child[4], BOUND_W_GROUND[0], BOUND_W_GROUND[1])
            child[5] = np.clip(child[5], BOUND_TONIC[0], BOUND_TONIC[1])
            child[6] = np.clip(child[6], BOUND_BETA[0], BOUND_BETA[1])
            child[7] = np.clip(child[7], BOUND_THRESH[0], BOUND_THRESH[1])
            
            new_genomes.append(child)
            
        self.set_genomes(new_genomes)

    def reset_states(self):
        self.gf_v.fill(V_REST)
        self.refractory_timer.fill(0)
        self.voltage_history = []
        
    def step_batch(self, inputs_batch, y_positions_N):
        """
        Vectorized LIF simulation step for the entire swarm.
        inputs_batch: shape (N, 4)
           col 0: looming_drive
           col 1: gap_vertical_offset
           col 2: vertical_velocity
           col 3: ground_hazard
        y_positions_N: shape (N,)
        Returns boolean array of shape (N,) indicating flaps.
        """
        active_mask = (self.refractory_timer <= 0)
        self.refractory_timer[~active_mask] -= 1
        self.gf_v[~active_mask] = V_RESET
        
        # Split inputs
        looming = inputs_batch[:, 0:1]
        gap_offset = inputs_batch[:, 1:2]
        vel = inputs_batch[:, 2:3]
        ground = inputs_batch[:, 3:4]
        
        pos_offset = np.maximum(0, gap_offset)
        neg_offset = np.minimum(0, gap_offset)
        
        # GATED INHIBITION FIX:
        # If bird.y >= y_gap (gap_offset >= 0), dorsal_inhibition is 0.0.
        # If bird.y < y_gap (gap_offset < 0), activate dorsal inhibition using W_dive.
        # We take abs(neg_offset) so that multiplying by negative W_dive yields a negative (inhibitory) current.
        dorsal_inhibition = np.abs(neg_offset) * self.W_dive
        
        I_net = (pos_offset * self.W_climb + 
                 dorsal_inhibition + 
                 looming * self.W_looming + 
                 vel * self.W_vel + 
                 ground * self.W_ground + 
                 self.I_tonic)
        
        # Ground Emergency Reflex (y > config.GROUND_Y - 120.0)
        # Note: We already have ground_hazard, but keeping this extra safeguard just in case
        ground_thresh = GROUND_Y - 120.0
        ground_mask = y_positions_N > ground_thresh
        I_ground_emergency = (y_positions_N[ground_mask] - ground_thresh) * 0.15
        I_net[ground_mask, 0] += I_ground_emergency
        
        # Update membrane potential
        self.gf_v[active_mask] = (self.gf_v[active_mask] - V_REST) * self.beta[active_mask] + V_REST + I_net[active_mask]
        
        # Ceiling Lockdown (force V_m to V_REST if y < 90)
        ceiling_mask = (y_positions_N < 90.0)
        self.gf_v[ceiling_mask, 0] = V_REST
        
        self.voltage_history.append(self.gf_v.copy())
        
        # Spikes
        spikes = (self.gf_v >= self.v_thresh) & active_mask[:, None]
        spiked_indices = np.where(spikes[:, 0])[0]
        
        # Overwrite the recorded history for spiked neurons to simulate an AP peak
        if len(spiked_indices) > 0:
            self.voltage_history[-1][spiked_indices, 0] = 20.0  # +20 mV peak
            
        self.gf_v[spiked_indices] = V_RESET
        self.refractory_timer[spiked_indices] = REFRACTORY_FRAMES
        
        return spikes[:, 0]
