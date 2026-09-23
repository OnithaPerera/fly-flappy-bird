import numpy as np
from config import (V_REST, V_RESET, V_THRESH, BETA, REFRACTORY_FRAMES,
                    BOUND_BETA, BOUND_THRESH, HALTERE_DAMPING,
                    ADAPTIVE_THRESH_INCREMENT, ADAPTIVE_THRESH_DECAY,
                    INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, GA_POPULATION_SIZE, TOTAL_GENOME_SIZE)

class BatchedRecurrentSNN:
    def __init__(self, num_agents=GA_POPULATION_SIZE):
        self.N = num_agents
        
        # State tensors
        self.hidden_v = np.full((self.N, HIDDEN_NODES), V_REST, dtype=np.float32)
        self.gf_v = np.full((self.N, OUTPUT_NODES), V_REST, dtype=np.float32)
        self.refractory_timer = np.zeros(self.N, dtype=np.int32)
        self.adaptive_thresh = np.zeros((self.N, OUTPUT_NODES), dtype=np.float32)
        
        # We need a way to store history for HUD (just tracking agent 0 or the leader)
        # But we don't know who the leader is here, so we might store a full matrix of history
        self.voltage_history = []
        
        # Genomes/weights (Batched)
        self.beta = np.full((self.N, 1), BETA, dtype=np.float32)
        self.v_thresh = np.full((self.N, 1), V_THRESH, dtype=np.float32)
        
        self.W_in = np.zeros((self.N, INPUT_NODES, HIDDEN_NODES), dtype=np.float32)
        self.W_rec = np.zeros((self.N, HIDDEN_NODES, HIDDEN_NODES), dtype=np.float32)
        self.W_out = np.zeros((self.N, HIDDEN_NODES, OUTPUT_NODES), dtype=np.float32)
        
        self.genomes = np.zeros((self.N, TOTAL_GENOME_SIZE), dtype=np.float32)

    def set_genomes(self, list_of_vectors):
        """
        Loads a list of 1D genomes into the batched weight matrices.
        """
        for i, genome in enumerate(list_of_vectors):
            self.genomes[i] = genome
            self.beta[i, 0] = genome[0]
            self.v_thresh[i, 0] = genome[1]
            
            ptr = 2
            
            size_in = HIDDEN_NODES * INPUT_NODES
            self.W_in[i] = genome[ptr:ptr+size_in].reshape((INPUT_NODES, HIDDEN_NODES))
            ptr += size_in
            
            size_rec = HIDDEN_NODES * HIDDEN_NODES
            self.W_rec[i] = genome[ptr:ptr+size_rec].reshape((HIDDEN_NODES, HIDDEN_NODES))
            ptr += size_rec
            
            size_out = OUTPUT_NODES * HIDDEN_NODES
            self.W_out[i] = genome[ptr:ptr+size_out].reshape((HIDDEN_NODES, OUTPUT_NODES))
            
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
            child[0] = np.clip(child[0], BOUND_BETA[0], BOUND_BETA[1])
            child[1] = np.clip(child[1], BOUND_THRESH[0], BOUND_THRESH[1])
            
            new_genomes.append(child)
            
        self.set_genomes(new_genomes)

    def reset_states(self):
        self.hidden_v.fill(V_REST)
        self.gf_v.fill(V_REST)
        self.refractory_timer.fill(0)
        self.adaptive_thresh.fill(0.0)
        self.voltage_history = []
        
    def step_batch(self, inputs_Nx1024, velocities_N):
        """
        Vectorized LIF simulation step for the entire swarm.
        inputs_Nx1024: shape (N, 1024)
        velocities_N: shape (N,)
        Returns boolean array of shape (N,) indicating flaps.
        """
        # Determine who is not in refractory period
        active_mask = (self.refractory_timer <= 0)
        
        # Decrement timers
        self.refractory_timer[~active_mask] -= 1
        
        # For those in refractory, reset V and decay adaptive thresh
        self.gf_v[~active_mask] = V_RESET
        self.adaptive_thresh[~active_mask] *= ADAPTIVE_THRESH_DECAY
        
        # Compute hidden layer spikes (N, 16)
        hidden_spikes = (self.hidden_v >= self.v_thresh).astype(np.float32)
        
        # Reset spiked hidden neurons
        self.hidden_v[hidden_spikes > 0] = V_RESET
        
        # Compute input current to hidden: (N, 1, 1024) @ (N, 1024, 16) -> (N, 1, 16)
        # Using np.einsum or batch matmul
        I_vis = np.einsum('ni,nij->nj', inputs_Nx1024, self.W_in)
        I_rec = np.einsum('ni,nij->nj', hidden_spikes, self.W_rec)
        I_hidden = I_vis + I_rec
        
        # Haltere Proprioceptive Damping (apply to I_hidden if velocity < 0)
        haltere_mask = velocities_N < 0
        I_hidden[haltere_mask] *= HALTERE_DAMPING
        
        # Update hidden membrane potentials (only for active agents? Actually hidden neurons always update)
        self.hidden_v = (self.hidden_v - V_REST) * self.beta + V_REST + I_hidden
        
        # Compute input current to GF
        I_gf = np.einsum('ni,nij->nj', hidden_spikes, self.W_out)
        
        effective_thresh = self.v_thresh + self.adaptive_thresh
        
        # Update GF membrane potential for active agents
        self.gf_v[active_mask] = (self.gf_v[active_mask] - V_REST) * self.beta[active_mask] + V_REST + I_gf[active_mask]
        
        # Decay adaptive thresh for active agents
        self.adaptive_thresh[active_mask] *= ADAPTIVE_THRESH_DECAY
        
        self.voltage_history.append(self.gf_v.copy())
        
        # Check GF spikes
        spikes = (self.gf_v >= effective_thresh) & active_mask[:, None]
        spiked_indices = np.where(spikes[:, 0])[0]
        
        self.gf_v[spiked_indices] = V_RESET
        self.adaptive_thresh[spiked_indices] += ADAPTIVE_THRESH_INCREMENT
        self.refractory_timer[spiked_indices] = REFRACTORY_FRAMES
        
        return spikes[:, 0]
