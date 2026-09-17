from config import V_REST, V_RESET, V_THRESH, BETA, REFRACTORY_PERIOD, SYNAPTIC_GAIN

class LoomingCircuitController:
    def __init__(self):
        self.v = V_REST
        self.refractory_timer = 0
        self.voltage_history = []
        
    def load_flywire_weights(self, token):
        """
        Optional FlyWire connectome hook.
        Loads real synaptic weights between LPLC2 and GF using fafbseg / CAVEclient.
        Root IDs: 720575940619053911, 720575940625624734
        """
        print(f"FlyWire Connectome hook called with token {token[:5]}... (stub).")
        pass
        
    def step(self, visual_current):
        """
        Leaky Integrate-and-Fire simulation step.
        Returns True if action potential fires (FLAP command).
        """
        if self.refractory_timer > 0:
            self.refractory_timer -= 1
            self.v = V_RESET
            self.voltage_history.append(self.v)
            return False
            
        # LIF Equation:
        # V(t) = (V(t-1) - V_rest) * beta + V_rest + (I * gain)
        self.v = (self.v - V_REST) * BETA + V_REST + (visual_current * SYNAPTIC_GAIN)
        
        self.voltage_history.append(self.v)
        
        if self.v >= V_THRESH:
            self.v = V_RESET
            self.refractory_timer = REFRACTORY_PERIOD
            return True
            
        return False
