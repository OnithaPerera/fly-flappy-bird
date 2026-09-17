import argparse
import sys
import pygame
import numpy as np
import cv2

from config import WINDOW_WIDTH, WINDOW_HEIGHT, FPS, V_REST, V_THRESH
from game import FlappyGame
from vision import preprocess_frame, compute_looming_stimulus
from connectome_lif import LoomingCircuitController

def run_headless_benchmarks(runs=10):
    print(f"Running {runs} headless benchmark runs...")
    # Import matplotlib only for post-run benchmarking
    import matplotlib.pyplot as plt
    
    scores = []
    survival_frames = []
    last_brain = None
    
    for r in range(runs):
        game = FlappyGame()
        brain = LoomingCircuitController()
        
        prev_frame = None
        
        while not game.game_over:
            # 1. Render to offscreen surface
            surface = game.render()
            
            # 2. Vision processing
            curr_frame = preprocess_frame(surface)
            visual_current, _ = compute_looming_stimulus(curr_frame, prev_frame)
            prev_frame = curr_frame
            
            # 3. Brain step
            flap = brain.step(visual_current)
            
            # 4. Game physics step
            game.step(flap)
            
        print(f"Run {r+1}: Score {game.score}, Frames Survived {game.frames}")
        scores.append(game.score)
        survival_frames.append(game.frames)
        last_brain = brain
        
    print("\n--- Benchmark Results ---")
    print(f"Average Score: {np.mean(scores):.2f}")
    print(f"Average Frames Survived: {np.mean(survival_frames):.2f}")
    
    # Save a static trace of the last run's Giant Fiber voltage
    if last_brain:
        plt.figure(figsize=(10, 4))
        plt.plot(last_brain.voltage_history, color='purple')
        plt.axhline(V_THRESH, color='red', linestyle='--', label='Threshold')
        plt.axhline(V_REST, color='gray', linestyle='--', label='Resting')
        plt.title("Giant Fiber Membrane Potential (Last Run)")
        plt.xlabel("Frames")
        plt.ylabel("Voltage (mV)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("gf_voltage_trace.png")
        print("Saved gf_voltage_trace.png")

def main():
    parser = argparse.ArgumentParser(description="Fly Flappy Bird")
    parser.add_argument("--headless", action="store_true", help="Run rapid evaluation benchmarks")
    args = parser.parse_args()

    if args.headless:
        # Initialize pygame minimally for offscreen surface creation
        pygame.init()
        pygame.display.set_mode((1, 1), pygame.HIDDEN)
        run_headless_benchmarks(runs=10)
        pygame.quit()
        return

    pygame.init()
    
    # Main window will have game on left (WINDOW_WIDTH) and HUD on right (300px)
    HUD_WIDTH = 300
    screen = pygame.display.set_mode((WINDOW_WIDTH + HUD_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Fly Flappy Bird - Neural Controller")
    clock = pygame.time.Clock()
    
    game = FlappyGame()
    brain = LoomingCircuitController()
    
    prev_frame = None
    font = pygame.font.SysFont("Arial", 16)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Space to restart if game over
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and game.game_over:
                    game.reset()
                    brain = LoomingCircuitController()
                    prev_frame = None

        if not game.game_over:
            # Render game state to offscreen surface
            offscreen_surf = game.render()
            
            # Vision
            curr_frame = preprocess_frame(offscreen_surf)
            visual_current, masked_diff = compute_looming_stimulus(curr_frame, prev_frame)
            prev_frame = curr_frame
            
            # Brain
            flap = brain.step(visual_current)
            
            # Physics
            game.step(flap)
        else:
            # Just render the game as it is
            offscreen_surf = game.render()
            flap = False
            
        # -- Drawing to main screen --
        screen.fill((30, 30, 30)) # Dark background for HUD
        
        # 1. Blit game on left
        screen.blit(offscreen_surf, (0, 0))
        
        # 2. Draw HUD on right
        hud_x = WINDOW_WIDTH
        
        # Score
        score_text = font.render(f"Score: {game.score}", True, (255, 255, 255))
        screen.blit(score_text, (hud_x + 10, 10))
        
        # Game Over status
        if game.game_over:
            go_text = font.render("GAME OVER (Press SPACE)", True, (255, 50, 50))
            screen.blit(go_text, (hud_x + 10, 30))
            
        # Flap indicator
        if flap:
            pygame.draw.circle(screen, (255, 0, 0), (hud_x + 250, 20), 10)
            
        # Draw compound eye vision (32x32 -> upscale to 128x128 for visibility)
        if prev_frame is not None:
            # Prepare masked_diff for display
            disp_eye = masked_diff.copy()
            max_val = np.max(disp_eye)
            if max_val > 0:
                disp_eye = (disp_eye / max_val * 255).astype(np.uint8)
            else:
                disp_eye = disp_eye.astype(np.uint8)
                
            # Create RGB by repeating gray channel
            eye_rgb = np.stack((disp_eye, disp_eye, disp_eye), axis=-1)
            # transpose for Pygame (width, height, color)
            eye_rgb = np.transpose(eye_rgb, (1, 0, 2))
            
            eye_surf = pygame.surfarray.make_surface(eye_rgb)
            eye_surf = pygame.transform.scale(eye_surf, (128, 128))
            screen.blit(eye_surf, (hud_x + 10, 60))
            
            label = font.render("LPLC2 Receptive Field", True, (200, 200, 200))
            screen.blit(label, (hud_x + 10, 195))
            
        # Draw GF Membrane Potential Graph using native pygame line drawing
        hist = brain.voltage_history[-150:] # Display last 150 frames
        graph_rect = pygame.Rect(hud_x + 10, 250, 270, 100)
        pygame.draw.rect(screen, (0, 0, 0), graph_rect)
        pygame.draw.rect(screen, (255, 255, 255), graph_rect, 1) # Border
        
        # Title
        gf_label = font.render("Giant Fiber Potential (mV)", True, (200, 200, 200))
        screen.blit(gf_label, (hud_x + 10, 230))
        
        # Scaling for graph
        min_v = -80
        max_v = -40
        v_range = max_v - min_v
        
        if len(hist) > 1:
            pts = []
            for i, v in enumerate(hist):
                x = graph_rect.x + (i / 150) * graph_rect.width
                # map v to y
                y = graph_rect.y + graph_rect.height - ((v - min_v) / v_range) * graph_rect.height
                pts.append((float(x), float(y)))
                
            pygame.draw.lines(screen, (150, 50, 255), False, pts, 2)
            
        # Draw threshold line
        thresh_y = graph_rect.y + graph_rect.height - ((V_THRESH - min_v) / v_range) * graph_rect.height
        pygame.draw.line(screen, (255, 50, 50), (graph_rect.x, thresh_y), (graph_rect.x + graph_rect.width, thresh_y), 1)

        pygame.display.flip()
        # Cap framerate in GUI mode
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
