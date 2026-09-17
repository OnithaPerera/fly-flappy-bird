import numpy as np
from config import V_REST, V_RESET, V_THRESH, BETA, REFRACTORY_PERIOD, SYNAPTIC_GAIN, EYE_RES

class LoomingCircuitController:
    def __init__(self, genome=None):
        self.v = V_REST
        self.refractory_timer = 0
        self.voltage_history = []
        self.genome_enabled = False
        
        # Default baseline parameters
        self.beta = BETA
        self.v_thresh = V_THRESH
        self.synaptic_gain = SYNAPTIC_GAIN
        
        # Spatial weights: 32x32
        self.spatial_weights = np.ones((EYE_RES, EYE_RES), dtype=np.float32)
        self.spatial_weights[0:EYE_RES//2, :] = 0.2
        self.spatial_weights[EYE_RES//2:, :] = 2.0
        
        if genome is not None:
            self.set_genome(genome)
            
    def get_genome(self):
        """
        Serializes the network into a 1D genome vector.
        [beta, v_thresh, synaptic_gain, ...spatial_weights_flat...]
        """
        flat_weights = self.spatial_weights.flatten()
        genome = np.concatenate(([self.beta, self.v_thresh, self.synaptic_gain], flat_weights))
        return genome
        
    def set_genome(self, genome):
        """
        Deserializes a 1D genome vector into network parameters.
        """
        self.genome_enabled = True
        self.beta = genome[0]
        self.v_thresh = genome[1]
        self.synaptic_gain = genome[2]
        self.spatial_weights = genome[3:].reshape((EYE_RES, EYE_RES))
        
    def load_flywire_weights(self, token):
        print(f"FlyWire Connectome hook called with token {token[:5]}... (stub).")
        pass
        
    def step(self, total_drive):
        """
        Leaky Integrate-and-Fire simulation step.
        Note: The visual processing now multiplies spatial_weights, so total_drive
        is the already weighted visual current.
        Returns True if action potential fires (FLAP command).
        """
        if self.refractory_timer > 0:
            self.refractory_timer -= 1
            self.v = V_RESET
            self.voltage_history.append(self.v)
            return False
            
        # LIF Equation
        self.v = (self.v - V_REST) * self.beta + V_REST + (total_drive * self.synaptic_gain)
        
        self.voltage_history.append(self.v)
        
        if self.v >= self.v_thresh:
            self.v = V_RESET
            self.refractory_timer = REFRACTORY_PERIOD
            return True
            
        return False
