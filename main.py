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
                    EYE_RES, EVAL_SEEDS, TOTAL_GENOME_SIZE, 
                    BOUND_DORSAL_W, BOUND_VENTRAL_W, BOUND_TONIC, BOUND_BETA, BOUND_THRESH, BOUND_HALTERE, BOUND_GROUND_GAIN)
from game import SwarmWorld
from vision import get_sensory_vector, get_colored_heatmap
from connectome_lif import BatchedPooledBiologicalLIF
from assets_loader import load_or_fetch_assets
import sound_fx

def generate_random_genome():
    genome = np.zeros(TOTAL_GENOME_SIZE, dtype=np.float32)
    genome[0:8] = np.random.uniform(BOUND_DORSAL_W[0], BOUND_DORSAL_W[1], 8)
    genome[8:16] = np.random.uniform(BOUND_VENTRAL_W[0], BOUND_VENTRAL_W[1], 8)
    genome[16] = np.random.uniform(BOUND_TONIC[0], BOUND_TONIC[1])
    genome[17] = np.random.uniform(BOUND_BETA[0], BOUND_BETA[1])
    genome[18] = np.random.uniform(BOUND_THRESH[0], BOUND_THRESH[1])
    genome[19] = np.random.uniform(BOUND_HALTERE[0], BOUND_HALTERE[1])
    genome[20] = np.random.uniform(BOUND_GROUND_GAIN[0], BOUND_GROUND_GAIN[1])
    return genome

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
        
    batched_snn = BatchedPooledBiologicalLIF(GA_POPULATION_SIZE)
    initial_genomes = [generate_random_genome() for _ in range(GA_POPULATION_SIZE)]
    
    if os.path.exists("champion_genome.npy"):
        try:
            champ = np.load("champion_genome.npy")
            if len(champ) == TOTAL_GENOME_SIZE:
                initial_genomes[0] = champ
                print("Loaded valid champion genome.")
            else:
                os.rename("champion_genome.npy", "champion_genome_legacy.npy")
                print("Warning: Legacy checkpoint shape mismatch. Renamed to champion_genome_legacy.npy.")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            
    batched_snn.set_genomes(initial_genomes)
    
    world = SwarmWorld(batched_snn.genomes, assets)
    
    current_eval_idx = 0
    current_seed = EVAL_SEEDS[current_eval_idx]
    world.reset(current_seed)
    batched_snn.reset_states()
    
    prev_frames = [None] * GA_POPULATION_SIZE
    
    generation = 1
    max_fitness_history = []
    all_time_record = 0
    all_time_record_seed = current_seed
    all_time_action_tape = set()
    best_overall_genome = None
    
    agent_total_fitness = np.zeros(GA_POPULATION_SIZE, dtype=np.float32)
    best_run_tapes = [None] * GA_POPULATION_SIZE
    best_run_seeds = [None] * GA_POPULATION_SIZE
    best_run_scores = [-1] * GA_POPULATION_SIZE
    
    running = True
    paused = False
    replay_mode = False
    
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
                    replay_mode = not replay_mode
                    if replay_mode:
                        if best_overall_genome is not None:
                            print(f"Entering Replay Mode for seed {all_time_record_seed}")
                            batched_snn = BatchedPooledBiologicalLIF(1)
                            batched_snn.set_genomes([best_overall_genome])
                            batched_snn.reset_states()
                            world = SwarmWorld(batched_snn.genomes, assets)
                            world.reset(all_time_record_seed)
                            prev_frames = [None]
                            paused = False
                            speed_multiplier = 1
                        else:
                            print("No champion genome to replay yet!")
                            replay_mode = False
                    else:
                        print("Exiting Replay Mode. Resuming evolution...")
                        batched_snn = BatchedPooledBiologicalLIF(GA_POPULATION_SIZE)
                        batched_snn.set_genomes(initial_genomes)
                        batched_snn.reset_states()
                        world = SwarmWorld(batched_snn.genomes, assets)
                        world.reset(current_seed)
                        prev_frames = [None] * GA_POPULATION_SIZE
                        paused = False
                elif event.key == pygame.K_s:
                    if best_overall_genome is not None:
                        np.save("champion_genome.npy", best_overall_genome)
                        print("Saved champion genome.")

        if not paused:
            for substep in range(speed_multiplier):
                if world.all_dead:
                    if replay_mode:
                        world.reset(all_time_record_seed)
                        batched_snn.reset_states()
                        prev_frames = [None]
                        break
                        
                    # Evolution step
                    for i, a in enumerate(world.agents):
                        fit = a.get_fitness()
                        agent_total_fitness[i] += fit
                        if fit > best_run_scores[i]:
                            best_run_scores[i] = fit
                            best_run_tapes[i] = set(a.action_tape)
                            best_run_seeds[i] = current_seed
                            
                    current_eval_idx += 1
                    
                    if current_eval_idx < len(EVAL_SEEDS):
                        current_seed = EVAL_SEEDS[current_eval_idx]
                        world.reset(current_seed)
                        batched_snn.reset_states()
                        prev_frames = [None] * GA_POPULATION_SIZE
                        break
                    else:
                        avg_fitnesses = agent_total_fitness / len(EVAL_SEEDS)
                        best_idx = int(np.argmax(avg_fitnesses))
                        gen_max_fitness = avg_fitnesses[best_idx]
                        max_fitness_history.append(gen_max_fitness)
                        
                        if gen_max_fitness > all_time_record:
                            all_time_record = gen_max_fitness
                            all_time_record_seed = best_run_seeds[best_idx]
                            all_time_action_tape = best_run_tapes[best_idx]
                            best_overall_genome = batched_snn.genomes[best_idx].copy()
                            print(f"New All-Time Record: {all_time_record:.1f} (Seed: {all_time_record_seed})")
                            
                        print(f"Gen {generation} | Max Fit: {gen_max_fitness:.1f} | Avg Fit: {np.mean(avg_fitnesses):.1f}")
                        
                        sorted_indices = np.argsort(avg_fitnesses)[::-1]
                        elite_indices = sorted_indices[:GA_ELITE_COUNT]
                        
                        mut_rate = max(MIN_MUT_RATE, INITIAL_MUT_RATE * (DECAY_RATE ** generation))
                        mut_scale = max(MIN_MUT_SCALE, INITIAL_MUT_SCALE * (DECAY_RATE ** generation))
                            
                        batched_snn.reproduce_and_mutate(elite_indices, avg_fitnesses, mut_rate, mut_scale)
                        
                        initial_genomes = [batched_snn.genomes[i].copy() for i in range(GA_POPULATION_SIZE)]
                        
                        generation += 1
                        current_eval_idx = 0
                        current_seed = EVAL_SEEDS[current_eval_idx]
                        
                        agent_total_fitness.fill(0)
                        best_run_tapes = [None] * GA_POPULATION_SIZE
                        best_run_seeds = [None] * GA_POPULATION_SIZE
                        best_run_scores = [-1] * GA_POPULATION_SIZE
                        
                        world = SwarmWorld(batched_snn.genomes, assets)
                        world.reset(current_seed)
                        batched_snn.reset_states()
                        prev_frames = [None] * GA_POPULATION_SIZE
                        break 
                
                offscreen_surf = world.render_for_vision()
                
                N_active = len(world.agents)
                inputs = np.zeros((N_active, 16), dtype=np.float32)
                velocities = np.zeros(N_active, dtype=np.float32)
                y_positions = np.zeros(N_active, dtype=np.float32)
                
                leader = world.get_leader()
                leader_idx = world.agents.index(leader) if leader else -1
                masked_diff_leader = np.zeros((4, 4))
                
                for i, agent in enumerate(world.agents):
                    if not agent.alive:
                        continue
                        
                    diff_tensor, curr_frame = get_sensory_vector(offscreen_surf, agent.rect, prev_frames[i], agent.velocity)
                    prev_frames[i] = curr_frame
                    inputs[i] = diff_tensor
                    velocities[i] = agent.velocity
                    y_positions[i] = agent.y
                    
                    if i == leader_idx:
                        masked_diff_leader = diff_tensor.reshape(4, 4)
                        
                flaps = batched_snn.step_batch(inputs, velocities, y_positions)
                
                if replay_mode:
                    flaps = np.array([world.frames in all_time_action_tape])
                    
                world.step(flaps)
                
                if flaps[leader_idx] if leader_idx != -1 else False:
                    sound_fx.play_spike_click()
                
        # Rendering
        screen.fill(COLOR_PANEL)
        arena_surface = world.render_for_vision()
        world.render_for_display(arena_surface)
        screen.blit(arena_surface, (0, 0))
        
        # HUD Panel (Right side)
        hud_x = ARENA_WIDTH
        
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
        
        y_coords = [20, 50, 80, 110, 140, 170]
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
            screen.blit(surf, (hud_x + 10, y_coords[i]))
            
        # Fitness Graph
        g_label = font.render("MULTI-SEED FITNESS HISTORY", True, COLOR_TEXT)
        screen.blit(g_label, (hud_x + 10, 205))
        
        graph_rect = pygame.Rect(hud_x + 10, 230, 420, 100)
        pygame.draw.rect(screen, (0, 0, 0), graph_rect)
        pygame.draw.rect(screen, COLOR_GRID, graph_rect, 1)
        
        if len(max_fitness_history) > 1 and not replay_mode:
            pts = []
            max_val = max(100, max(max_fitness_history))
            min_val = min(max_fitness_history)
            val_range = max(1, max_val - min_val)
            
            for i, val in enumerate(max_fitness_history):
                x = float(graph_rect.x + (i / max(1, len(max_fitness_history) - 1)) * graph_rect.width)
                y = float(graph_rect.y + graph_rect.height - ((val - min_val) / val_range) * graph_rect.height)
                pts.append((x, y))
                
            pygame.draw.lines(screen, COLOR_ACCENT, False, pts, 2)
            
        # Leader Brain View
        lb_label = font.render("COMPOUND EYE (4x4 POOLED)", True, COLOR_TEXT)
        screen.blit(lb_label, (hud_x + 10, 350))
        
        if leader:
            heatmap_rgb = get_colored_heatmap(masked_diff_leader)
            eye_surf = pygame.surfarray.make_surface(heatmap_rgb)
            # Nearest neighbor scaling preserves the 4x4 grid pixel look
            eye_surf = pygame.transform.scale(eye_surf, (100, 100))
            screen.blit(eye_surf, (hud_x + 10, 375))
            
        # Giant Fiber Oscilloscope
        osc_label = font.render("GIANT FIBER VOLTAGE (Vm)", True, COLOR_TEXT)
        screen.blit(osc_label, (hud_x + 10, 495))
        
        osc_rect = pygame.Rect(hud_x + 10, 520, 420, 65)
        pygame.draw.rect(screen, (0, 0, 0), osc_rect)
        pygame.draw.rect(screen, COLOR_GRID, osc_rect, 1)
        
        if leader:
            hist = [float(v[0, 0]) for v in batched_snn.voltage_history[-200:]] if replay_mode else [float(v[leader_idx, 0]) for v in batched_snn.voltage_history[-200:]]
            min_v, max_v = -80.0, -40.0
            v_range = max_v - min_v
            
            if len(hist) > 1:
                pts = []
                for i, v in enumerate(hist):
                    x = float(osc_rect.x + (i / 200) * osc_rect.width)
                    y = float(osc_rect.y + osc_rect.height - ((v - min_v) / v_range) * osc_rect.height)
                    pts.append((x, y))
                    
                pygame.draw.lines(screen, (10, 150, 10), False, pts, 4)
                pygame.draw.lines(screen, COLOR_PHOSPHOR, False, pts, 1)
                
            eff_t = float(batched_snn.v_thresh[0,0] if replay_mode else batched_snn.v_thresh[leader_idx,0])
            thresh_y = float(osc_rect.y + osc_rect.height - ((eff_t - min_v) / v_range) * osc_rect.height)
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
