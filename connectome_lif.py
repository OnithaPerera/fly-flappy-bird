import numpy as np
from config import (V_REST, V_RESET, V_THRESH, BETA, REFRACTORY_FRAMES, SYNAPTIC_GAIN, EYE_RES,
                    BOUND_BETA, BOUND_THRESH, BOUND_GAIN, BOUND_WEIGHT, HALTERE_DAMPING,
                    ADAPTIVE_THRESH_INCREMENT, ADAPTIVE_THRESH_DECAY,
                    INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS)

class RecurrentConnectomeLIF:
    def __init__(self, genome=None):
        self.hidden_v = np.full(HIDDEN_NEURONS, V_REST, dtype=np.float32)
        self.gf_v = V_REST
        self.refractory_timer = 0
        self.adaptive_thresh = 0.0
        self.voltage_history = []
        
        # SNN Hyperparams
        self.beta = BETA
        self.v_thresh = V_THRESH
        
        # Weights
        if genome is not None:
            self.set_genome(genome)
        else:
            # Small random weights
            self.W_in = np.random.normal(0, 0.01, (HIDDEN_NEURONS, INPUT_NEURONS)).astype(np.float32)
            self.W_rec = np.random.normal(0, 0.01, (HIDDEN_NEURONS, HIDDEN_NEURONS)).astype(np.float32)
            self.W_out = np.random.normal(0, 0.01, (OUTPUT_NEURONS, HIDDEN_NEURONS)).astype(np.float32)
            
    def get_genome(self):
        genome = np.concatenate((
            [self.beta, self.v_thresh],
            self.W_in.flatten(),
            self.W_rec.flatten(),
            self.W_out.flatten()
        ))
        return genome
        
    def set_genome(self, genome):
        self.beta = genome[0]
        self.v_thresh = genome[1]
        
        ptr = 2
        
        size_in = HIDDEN_NEURONS * INPUT_NEURONS
        self.W_in = genome[ptr:ptr+size_in].reshape((HIDDEN_NEURONS, INPUT_NEURONS))
        ptr += size_in
        
        size_rec = HIDDEN_NEURONS * HIDDEN_NEURONS
        self.W_rec = genome[ptr:ptr+size_rec].reshape((HIDDEN_NEURONS, HIDDEN_NEURONS))
        ptr += size_rec
        
        size_out = OUTPUT_NEURONS * HIDDEN_NEURONS
        self.W_out = genome[ptr:ptr+size_out].reshape((OUTPUT_NEURONS, HIDDEN_NEURONS))
        
    def clone(self):
        return RecurrentConnectomeLIF(genome=self.get_genome())
        
    def mutate(self, rate, scale):
        genome = self.get_genome()
        
        mask = np.random.rand(len(genome)) < rate
        mutations = np.random.normal(0, scale, len(genome))
        genome += mask * mutations
        
        # Enforce bounds
        genome[0] = np.clip(genome[0], BOUND_BETA[0], BOUND_BETA[1])
        genome[1] = np.clip(genome[1], BOUND_THRESH[0], BOUND_THRESH[1])
        
        self.set_genome(genome)
        
    def step(self, visual_input, vertical_velocity):
        if self.refractory_timer > 0:
            self.refractory_timer -= 1
            self.gf_v = V_RESET
            self.adaptive_thresh *= ADAPTIVE_THRESH_DECAY
            self.voltage_history.append(self.gf_v)
            return False
            
        # Hidden layer
        hidden_spikes = (self.hidden_v >= self.v_thresh).astype(np.float32)
        
        # Reset spiked hidden neurons
        self.hidden_v[hidden_spikes > 0] = V_RESET
        
        # Compute input current to hidden
        I_hidden = self.W_in @ visual_input + self.W_rec @ hidden_spikes
        
        # Haltere Proprioceptive Damping
        if vertical_velocity < 0:
            I_hidden *= HALTERE_DAMPING
            
        # Update hidden membrane potentials
        self.hidden_v = (self.hidden_v - V_REST) * self.beta + V_REST + I_hidden
        
        # Compute input current to GF
        # Assuming OUTPUT_NEURONS is 1
        I_gf = float(self.W_out @ hidden_spikes)
        
        effective_thresh = self.v_thresh + self.adaptive_thresh
        
        # Update GF membrane potential
        self.gf_v = (self.gf_v - V_REST) * self.beta + V_REST + I_gf
        self.adaptive_thresh *= ADAPTIVE_THRESH_DECAY
        
        self.voltage_history.append(self.gf_v)
        
        if self.gf_v >= effective_thresh:
            self.gf_v = V_RESET
            self.adaptive_thresh += ADAPTIVE_THRESH_INCREMENT
            self.refractory_timer = REFRACTORY_FRAMES
            return True
            
        return False
