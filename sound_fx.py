import numpy as np
import pygame
from config import AUDIO_SAMPLE_RATE, AUDIO_CLICK_MS

class SoundSynthesizer:
    def __init__(self):
        self.enabled = False
        try:
            # Init mixer at standard sample rate, 16-bit, stereo
            pygame.mixer.init(frequency=AUDIO_SAMPLE_RATE, size=-16, channels=2, buffer=512)
            self.enabled = True
            
            # Synthesize the procedural spike click
            self.spike_sound = self._generate_spike_click()
            print("[SoundFX] Procedural audio initialized successfully.")
        except Exception as e:
            print(f"[SoundFX] Failed to initialize mixer: {e}. Audio disabled.")

    def _generate_spike_click(self):
        """
        Synthesize a sharp, 12ms band-passed Geiger-style click using numpy.
        """
        duration = AUDIO_CLICK_MS / 1000.0
        t = np.linspace(0, duration, int(AUDIO_SAMPLE_RATE * duration), endpoint=False)
        
        # Base transient pop (exponential decay white noise + high freq sine)
        noise = np.random.normal(0, 1, len(t))
        sine = np.sin(2 * np.pi * 3500 * t) # High freq ping
        
        # Envelope: very fast attack, fast decay
        envelope = np.exp(-t * 800)
        
        waveform = (noise * 0.3 + sine * 0.7) * envelope
        
        # Normalize and convert to 16-bit signed int
        waveform = np.int16(waveform / np.max(np.abs(waveform)) * 32767)
        
        # Duplicate to 2 channels for stereo
        waveform = np.column_stack((waveform, waveform))
        
        # Create sound from numpy array
        return pygame.sndarray.make_sound(waveform)

    def play_spike_click(self):
        if self.enabled and self.spike_sound:
            self.spike_sound.play()

# Singleton instance
synth = SoundSynthesizer()

def play_spike_click():
    synth.play_spike_click()
