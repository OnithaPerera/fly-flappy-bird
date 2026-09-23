import sys
import os
# pyrefly: ignore [missing-import]
import pygame
# pyrefly: ignore [missing-import]
import numpy as np
import random
import time

from diagnostics import run_preflight_checks, validate_frame_tensor
run_preflight_checks() 

from config import (WINDOW_WIDTH, WINDOW_HEIGHT, ARENA_WIDTH, HUD_WIDTH, FPS,
                    COLOR_PANEL, COLOR_PHOSPHOR, COLOR_GRID, COLOR_LEADER, COLOR_TEXT, COLOR_ACCENT,
                    GA_POPULATION_SIZE, GA_ELITE_COUNT, INITIAL_MUT_RATE, MIN_MUT_RATE, INITIAL_MUT_SCALE, MIN_MUT_SCALE, DECAY_RATE,
                    EYE_RES, EVAL_SEEDS)
from game import SwarmWorld
from vision import preprocess_frame, compute_stabilized_looming, get_colored_heatmap
from connectome_lif import RecurrentConnectomeLIF
from assets_loader import load_or_fetch_assets
import sound_fx

def initialize_population(size):
    pop = []
    for _ in range(size):
        brain = RecurrentConnectomeLIF()
        # Initial random mutation
        brain.mutate(1.0, 0.5)
        pop.append(brain)
    return pop

def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Neuro-Simulation Lab: Swarm Evolution")
    clock = pygame.time.Clock()
    
    font = pygame.font.SysFont("Consolas", 14)
    large_font = pygame.font.SysFont("Consolas", 18, bold=True)
    
    try:
        assets = load_or_fetch_assets()
    except Exception as e:
        print(f"Critical error loading assets: {e}")
        sys.exit(1)
        
    population = initialize_population(GA_POPULATION_SIZE)
    world = SwarmWorld(population, assets)
    
    current_eval_idx = 0
    current_seed = EVAL_SEEDS[current_eval_idx]
    world.reset(current_seed)
    
    prev_frames = [None] * GA_POPULATION_SIZE
    
    generation = 1
    max_fitness_history = []
    all_time_record = 0
    all_time_record_seed = current_seed
    all_time_action_tape = []
    best_overall_genome = None
    
    agent_total_fitness = [0] * GA_POPULATION_SIZE
    best_run_tapes = [None] * GA_POPULATION_SIZE
    best_run_seeds = [None] * GA_POPULATION_SIZE
    best_run_scores = [-1] * GA_POPULATION_SIZE
    
    running = True
    paused = False
    replay_mode = False
    replay_frame_idx = 0
    
    # Speed multiplier (1, 2, 5, 15)
    speed_multiplier = 1
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    speed_multiplier = 1
                elif event.key == pygame.K_2:
                    speed_multiplier = 2
                elif event.key == pygame.K_5:
                    speed_multiplier = 5
                elif event.key in (pygame.K_0, pygame.K_f):
                    speed_multiplier = 15
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_r:
                    # Toggle replay mode
                    replay_mode = not replay_mode
                    if replay_mode:
                        if best_overall_genome is not None:
                            print(f"Entering Replay Mode for seed {all_time_record_seed}")
                            replay_brain = RecurrentConnectomeLIF(genome=best_overall_genome)
                            world = SwarmWorld([replay_brain], assets)
                            world.reset(all_time_record_seed)
                            prev_frames = [None]
                            replay_frame_idx = 0
                            paused = False
                            speed_multiplier = 1
                        else:
                            print("No champion genome to replay yet!")
                            replay_mode = False
                    else:
                        print("Exiting Replay Mode. Resuming evolution...")
                        world = SwarmWorld(population, assets)
                        world.reset(current_seed)
                        prev_frames = [None] * GA_POPULATION_SIZE
                        paused = False
                elif event.key == pygame.K_s:
                    if best_overall_genome is not None:
                        np.save("best_fly_genome.npy", best_overall_genome)
                        print("Saved best genome.")

        if not paused:
            for substep in range(speed_multiplier):
                if world.all_dead:
                    if replay_mode:
                        # In replay mode, just loop the replay
                        world.reset(all_time_record_seed)
                        prev_frames = [None]
                        replay_frame_idx = 0
                        break
                        
                    # Evolution step
                    for i, a in enumerate(world.agents):
                        fit = a.get_fitness()
                        agent_total_fitness[i] += fit
                        if fit > best_run_scores[i]:
                            best_run_scores[i] = fit
                            best_run_tapes[i] = a.action_tape.copy()
                            best_run_seeds[i] = current_seed
                            
                    current_eval_idx += 1
                    
                    if current_eval_idx < len(EVAL_SEEDS):
                        current_seed = EVAL_SEEDS[current_eval_idx]
                        world = SwarmWorld(population, assets)
                        world.reset(current_seed)
                        prev_frames = [None] * GA_POPULATION_SIZE
                        break
                    else:
                        # End of generation
                        avg_fitnesses = [f / len(EVAL_SEEDS) for f in agent_total_fitness]
                        best_idx = int(np.argmax(avg_fitnesses))
                        gen_max_fitness = avg_fitnesses[best_idx]
                        max_fitness_history.append(gen_max_fitness)
                        
                        if gen_max_fitness > all_time_record:
                            all_time_record = gen_max_fitness
                            all_time_record_seed = best_run_seeds[best_idx]
                            all_time_action_tape = best_run_tapes[best_idx]
                            best_overall_genome = population[best_idx].get_genome()
                            print(f"New All-Time Record: {all_time_record:.1f} (Seed: {all_time_record_seed})")
                            
                        print(f"Gen {generation} | Max Fit: {gen_max_fitness:.1f} | Avg Fit: {np.mean(avg_fitnesses):.1f}")
                        
                        # Sort indices by fitness descending
                        sorted_indices = np.argsort(avg_fitnesses)[::-1]
                        
                        next_population = []
                        # Elitism
                        for i in range(GA_ELITE_COUNT):
                            next_population.append(population[sorted_indices[i]].clone())
                            
                        mut_rate = max(MIN_MUT_RATE, INITIAL_MUT_RATE * (DECAY_RATE ** generation))
                        mut_scale = max(MIN_MUT_SCALE, INITIAL_MUT_SCALE * (DECAY_RATE ** generation))
                            
                        # Tournament selection
                        while len(next_population) < GA_POPULATION_SIZE:
                            tourney_indices = random.sample(range(GA_POPULATION_SIZE), 3)
                            winner_idx = max(tourney_indices, key=lambda idx: avg_fitnesses[idx])
                            child = population[winner_idx].clone()
                            child.mutate(mut_rate, mut_scale)
                            next_population.append(child)
                            
                        population = next_population
                        generation += 1
                        current_eval_idx = 0
                        current_seed = EVAL_SEEDS[current_eval_idx]
                        
                        agent_total_fitness = [0] * GA_POPULATION_SIZE
                        best_run_tapes = [None] * GA_POPULATION_SIZE
                        best_run_seeds = [None] * GA_POPULATION_SIZE
                        best_run_scores = [-1] * GA_POPULATION_SIZE
                        
                        world = SwarmWorld(population, assets)
                        world.reset(current_seed)
                        prev_frames = [None] * GA_POPULATION_SIZE
                        break # Break out of substeps to render the new generation
                
                # Physics and vision update
                offscreen_surf = world.render()
                curr_frame = preprocess_frame(offscreen_surf)
                try:
                    validate_frame_tensor(curr_frame, (EYE_RES, EYE_RES))
                except Exception as e:
                    print(e)
                    
                flaps = []
                leader = world.get_leader()
                leader_idx = world.agents.index(leader) if leader else -1
                
                masked_diff_leader = np.zeros((EYE_RES, EYE_RES))
                
                for i, agent in enumerate(world.agents):
                    if not agent.alive:
                        flaps.append(False)
                        continue
                        
                    drive = compute_stabilized_looming(curr_frame, prev_frames[i], agent.velocity)
                    prev_frames[i] = curr_frame
                    
                    if i == leader_idx:
                        masked_diff_leader = drive.reshape(32, 32)
                        
                    flap = agent.brain.step(drive, agent.velocity)
                    
                    if replay_mode:
                        if replay_frame_idx < len(all_time_action_tape):
                            flap = all_time_action_tape[replay_frame_idx]
                        else:
                            flap = False
                            
                    flaps.append(flap)
                    
                    # Optional: play click for leader
                    if flap and i == leader_idx:
                        sound_fx.play_spike_click()
                
                if replay_mode and not world.all_dead:
                    replay_frame_idx += 1
                    
                world.step(flaps)
                
        # Rendering
        screen.fill(COLOR_PANEL)
        
        offscreen_surf = world.render()
        screen.blit(offscreen_surf, (0, 0))
        
        # HUD Panel (Right side)
        hud_x = ARENA_WIDTH
        
        # Grid lines for HUD
        for y in range(0, WINDOW_HEIGHT, 40):
            pygame.draw.line(screen, COLOR_GRID, (hud_x, y), (hud_x + HUD_WIDTH, y))
        for x in range(hud_x, hud_x + HUD_WIDTH, 40):
            pygame.draw.line(screen, COLOR_GRID, (x, 0), (x, WINDOW_HEIGHT))
            
        pygame.draw.line(screen, COLOR_ACCENT, (hud_x, 0), (hud_x, WINDOW_HEIGHT), 3)
        
        alive_count = sum(1 for a in world.agents if a.alive)
        leader = world.get_leader()
        current_score = leader.get_fitness() if leader else 0
        
        mode_text = "REPLAY MODE" if replay_mode else f"GEN: {generation} | SEED: {current_eval_idx+1}/{len(EVAL_SEEDS)}"
        
        mut_rate = max(MIN_MUT_RATE, INITIAL_MUT_RATE * (DECAY_RATE ** generation))
        mut_scale = max(MIN_MUT_SCALE, INITIAL_MUT_SCALE * (DECAY_RATE ** generation))
        
        texts = [
            mode_text,
            f"ALIVE: {alive_count} / {len(world.agents)}",
            f"CURRENT FITNESS: {int(current_score)}",
            f"ALL-TIME RECORD: {int(all_time_record)}",
            f"MUT RATE: {mut_rate:.3f} | SCALE: {mut_scale:.3f}",
            f"SIM SPEED: {speed_multiplier}X {'(PAUSED)' if paused else ''}"
        ]
        
        for i, t in enumerate(texts):
            color = COLOR_LEADER if replay_mode and i == 0 else (COLOR_PHOSPHOR if i == 0 else COLOR_TEXT)
            surf = font.render(t, True, color)
            screen.blit(surf, (hud_x + 10, 10 + i * 25))
            
        # Fitness Graph
        graph_rect = pygame.Rect(hud_x + 10, 150, 420, 100)
        pygame.draw.rect(screen, (0, 0, 0), graph_rect)
        pygame.draw.rect(screen, COLOR_GRID, graph_rect, 1)
        
        g_label = font.render("MULTI-SEED FITNESS HISTORY", True, COLOR_TEXT)
        screen.blit(g_label, (hud_x + 10, 130))
        
        if len(max_fitness_history) > 1 and not replay_mode:
            pts = []
            max_val = max(100, max(max_fitness_history))
            min_val = min(max_fitness_history)
            val_range = max(1, max_val - min_val)
            
            for i, val in enumerate(max_fitness_history):
                x = graph_rect.x + (i / max(1, len(max_fitness_history) - 1)) * graph_rect.width
                y = graph_rect.y + graph_rect.height - ((val - min_val) / val_range) * graph_rect.height
                pts.append((x, y))
                
            pygame.draw.lines(screen, COLOR_ACCENT, False, pts, 2)
            
        # Leader Brain View
        lb_label = font.render("STABILIZED COMPOUND EYE", True, COLOR_TEXT)
        screen.blit(lb_label, (hud_x + 10, 270))
        
        if leader:
            heatmap_rgb = get_colored_heatmap(masked_diff_leader)
            eye_surf = pygame.surfarray.make_surface(heatmap_rgb)
            eye_surf = pygame.transform.scale(eye_surf, (150, 150))
            screen.blit(eye_surf, (hud_x + 10, 290))
            
        # Giant Fiber Oscilloscope
        osc_label = font.render("GIANT FIBER VOLTAGE (Vm)", True, COLOR_TEXT)
        screen.blit(osc_label, (hud_x + 10, 460))
        
        osc_rect = pygame.Rect(hud_x + 10, 480, 420, 100)
        pygame.draw.rect(screen, (0, 0, 0), osc_rect)
        pygame.draw.rect(screen, COLOR_GRID, osc_rect, 1)
        
        if leader:
            hist = leader.brain.voltage_history[-200:]
            min_v, max_v = -80.0, -40.0
            v_range = max_v - min_v
            
            if len(hist) > 1:
                pts = []
                for i, v in enumerate(hist):
                    x = osc_rect.x + (i / 200) * osc_rect.width
                    y = osc_rect.y + osc_rect.height - ((v - min_v) / v_range) * osc_rect.height
                    pts.append((x, y))
                    
                pygame.draw.lines(screen, (10, 150, 10), False, pts, 4)
                pygame.draw.lines(screen, COLOR_PHOSPHOR, False, pts, 1)
                
            effective_thresh = leader.brain.v_thresh + leader.brain.adaptive_thresh
            thresh_y = osc_rect.y + osc_rect.height - ((effective_thresh - min_v) / v_range) * osc_rect.height
            pygame.draw.line(screen, COLOR_LEADER, (osc_rect.x, thresh_y), (osc_rect.x + osc_rect.width, thresh_y), 1)

        # Pause Overlay
        if paused:
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            screen.blit(overlay, (0, 0))
            
            pause_text = large_font.render("SIMULATION PAUSED", True, COLOR_TEXT)
            rect = pause_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
            screen.blit(pause_text, rect)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    run_simulation()
