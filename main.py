import argparse
import sys
import os
import pygame
import numpy as np

from diagnostics import run_preflight_checks
run_preflight_checks() # Run before pygame init

from config import (WINDOW_WIDTH, WINDOW_HEIGHT, FPS, HUD_WIDTH,
                    COLOR_CRT_DARK, COLOR_PHOSPHOR, COLOR_GRID, COLOR_AMBER,
                    V_REST, V_THRESH)
from game import FlappyWorld
from vision import preprocess_frame, compute_looming_stimulus, get_colored_heatmap
from connectome_lif import LoomingCircuitController
import sound_fx
from train_ga import train

def run_game_loop(brain, interactive=True):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH + HUD_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Neuro-Simulation Lab: Drosophila melanogaster")
    clock = pygame.time.Clock()
    
    font = pygame.font.SysFont("Consolas", 14)
    large_font = pygame.font.SysFont("Consolas", 18, bold=True)
    
    world = FlappyWorld(num_agents=1)
    
    prev_frame = None
    show_heatmap = True
    manual_inject = False
    
    running = True
    flap_count = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if world.all_dead:
                        world.reset()
                        brain.v = V_REST
                        prev_frame = None
                        flap_count = 0
                    else:
                        manual_inject = True
                elif event.key == pygame.K_UP:
                    brain.synaptic_gain *= 1.1
                elif event.key == pygame.K_DOWN:
                    brain.synaptic_gain /= 1.1
                elif event.key == pygame.K_t:
                    show_heatmap = not show_heatmap
                elif event.key == pygame.K_m:
                    sound_fx.synth.enabled = not sound_fx.synth.enabled

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    manual_inject = False

        if not world.all_dead:
            offscreen_surf = world.render()
            curr_frame = preprocess_frame(offscreen_surf)
            
            drive, masked_diff = compute_looming_stimulus(curr_frame, prev_frame, brain.spatial_weights)
            prev_frame = curr_frame
            
            # Manual injection override
            if manual_inject:
                drive += 50000.0
                
            flap = brain.step(drive)
            if flap:
                sound_fx.play_spike_click()
                flap_count += 1
                
            world.step([flap])
        else:
            offscreen_surf = world.render()
            flap = False
            masked_diff = np.zeros((32, 32))
            
        # Rendering Main Window
        screen.fill(COLOR_CRT_DARK)
        screen.blit(offscreen_surf, (0, 0))
        
        # Telemetry HUD
        hud_x = WINDOW_WIDTH
        
        # Grid lines for HUD
        for y in range(0, WINDOW_HEIGHT, 40):
            pygame.draw.line(screen, COLOR_GRID, (hud_x, y), (hud_x + HUD_WIDTH, y))
        for x in range(hud_x, hud_x + HUD_WIDTH, 40):
            pygame.draw.line(screen, COLOR_GRID, (x, 0), (x, WINDOW_HEIGHT))
            
        pygame.draw.line(screen, COLOR_PHOSPHOR, (hud_x, 0), (hud_x, WINDOW_HEIGHT), 2)
        
        # Text Metrics
        agent = world.agents[0]
        texts = [
            f"SYSTEM: ACTIVE",
            f"FITNESS: {agent.frames_survived + agent.score * 500}",
            f"PIPES CLEARED: {agent.score}",
            f"SYNAPTIC GAIN: {brain.synaptic_gain:.4f}",
            f"SPIKE COUNT: {flap_count}",
            f"SOUND: {'ON' if sound_fx.synth.enabled else 'MUTED'}"
        ]
        
        for i, t in enumerate(texts):
            color = COLOR_PHOSPHOR if i != 0 else COLOR_AMBER
            surf = font.render(t, True, color)
            screen.blit(surf, (hud_x + 10, 10 + i * 20))
            
        if world.all_dead:
            go_text = large_font.render("AGENT TERMINATED - PRESS SPACE", True, (255, 50, 50))
            screen.blit(go_text, (hud_x + 10, 150))
            
        # Spike Indicator
        pygame.draw.circle(screen, COLOR_AMBER if flap else (30, 30, 30), (hud_x + 270, 20), 8)
        
        # Thermal Compound Eye Feed
        if show_heatmap and prev_frame is not None:
            heatmap_rgb = get_colored_heatmap(masked_diff)
            eye_surf = pygame.surfarray.make_surface(heatmap_rgb)
            eye_surf = pygame.transform.scale(eye_surf, (150, 150))
            screen.blit(eye_surf, (hud_x + 10, 200))
            
            label = font.render("LPLC2 RECEPTIVE FIELD", True, COLOR_PHOSPHOR)
            screen.blit(label, (hud_x + 10, 355))
            
        # Oscilloscope Voltage Trace
        hist = brain.voltage_history[-150:]
        graph_rect = pygame.Rect(hud_x + 10, 420, 270, 120)
        pygame.draw.rect(screen, COLOR_GRID, graph_rect)
        pygame.draw.rect(screen, COLOR_PHOSPHOR, graph_rect, 1)
        
        v_label = font.render(f"GF VOLTAGE (Vm): {brain.v:.1f}mV", True, COLOR_PHOSPHOR)
        screen.blit(v_label, (hud_x + 10, 400))
        
        min_v, max_v = -80.0, -40.0
        v_range = max_v - min_v
        
        if len(hist) > 1:
            pts = []
            for i, v in enumerate(hist):
                x = graph_rect.x + (i / 150) * graph_rect.width
                y = graph_rect.y + graph_rect.height - ((v - min_v) / v_range) * graph_rect.height
                pts.append((float(x), float(y)))
                
            # Draw glow
            pygame.draw.lines(screen, (10, 150, 10), False, pts, 4)
            pygame.draw.lines(screen, COLOR_PHOSPHOR, False, pts, 1)
            
        # Threshold line
        thresh_y = graph_rect.y + graph_rect.height - ((brain.v_thresh - min_v) / v_range) * graph_rect.height
        pygame.draw.line(screen, COLOR_AMBER, (graph_rect.x, thresh_y), (graph_rect.x + graph_rect.width, thresh_y), 1)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

def main():
    parser = argparse.ArgumentParser(description="Neuroevolution Drosophila Flappy Bird")
    parser.add_argument("--play", action="store_true", help="Play mode with best genome")
    parser.add_argument("--train", action="store_true", help="Train using genetic algorithm")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations to train")
    parser.add_argument("--load", type=str, help="Load genome from .npy file")
    parser.add_argument("--mute", action="store_true", help="Disable procedural audio")
    
    args = parser.parse_args()
    
    if args.mute:
        sound_fx.synth.enabled = False

    if args.train:
        train(generations=args.generations)
        return

    brain = LoomingCircuitController()
    if args.load:
        if os.path.exists(args.load):
            genome = np.load(args.load)
            brain.set_genome(genome)
            print(f"Loaded genome from {args.load}")
        else:
            print(f"Genome file {args.load} not found, using baseline.")
    elif args.play:
        if os.path.exists("best_fly_genome.npy"):
            genome = np.load("best_fly_genome.npy")
            brain.set_genome(genome)
            print("Loaded best_fly_genome.npy")
        else:
            print("best_fly_genome.npy not found. Run --train first! Using baseline.")
            
    run_game_loop(brain)

if __name__ == "__main__":
    main()
