# Fly Flappy Bird Neural Simulation

A complete, modular Python project where a Flappy Bird game is autonomously controlled by an agent simulating the fruit fly (Drosophila melanogaster) visual looming and escape circuit.

## Architecture
- **vision.py**: Downsamples the screen buffer to a 32x32 virtual compound eye and computes absolute temporal luminance differences to detect optical expansion (looming).
- **connectome_lif.py**: Models the Giant Fiber (GF) descending neuron using a Leaky Integrate-and-Fire equation.
- **game.py**: Core Flappy Bird physics and rendering.
- **main.py**: Ties the simulation together, handles the GUI and Neural Debug HUD, and provides a headless benchmarking mode.

## Installation
Ensure you have Python 3.8+ installed. Install the dependencies via:
```bash
pip install -r requirements.txt
```

## Running the Simulation

**GUI Mode:**
Watch the simulated neural circuit play the game in real-time with an interactive HUD.
```bash
python main.py
```

**Headless Benchmarking Mode:**
Run 10 rapid evaluation iterations without Pygame's display overhead to measure average score and survival frames. Outputs a static `gf_voltage_trace.png` showing the GF membrane potential of the final run.
```bash
python main.py --headless
```
