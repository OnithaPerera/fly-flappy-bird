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

from config import (WINDOW_WIDTH, WINDOW_HEIGHT, ARENA_WIDTH, HUD_WIDTH, LAB_WIDTH, FPS,
                    COLOR_PANEL, COLOR_PHOSPHOR, COLOR_GRID, COLOR_LEADER, COLOR_TEXT, COLOR_ACCENT,
                    GA_POPULATION_SIZE, GA_ELITE_COUNT, INITIAL_MUT_RATE, MIN_MUT_RATE, INITIAL_MUT_SCALE, MIN_MUT_SCALE, DECAY_RATE,
                    EYE_RES, EVAL_SEEDS, TOTAL_GENOME_SIZE, 
                    BOUND_W_CLIMB, BOUND_W_DIVE, BOUND_W_LOOMING, BOUND_W_VEL, BOUND_W_GROUND, BOUND_TONIC, BOUND_BETA, BOUND_THRESH)
from game import SwarmWorld
from vision import extract_forward_binary_grid, get_colored_heatmap
from connectome_lif import LobulaColumnarSNN
from brain_visualizer import BrainVisualizer
from assets_loader import load_or_fetch_assets
import sound_fx

def generate_random_genome():
    genome = np.zeros(TOTAL_GENOME_SIZE, dtype=np.float32)
    genome[0] = np.random.uniform(BOUND_W_CLIMB[0], BOUND_W_CLIMB[1])
    genome[1] = np.random.uniform(BOUND_W_DIVE[0], BOUND_W_DIVE[1])
    genome[2] = np.random.uniform(BOUND_W_LOOMING[0], BOUND_W_LOOMING[1])
    genome[3] = np.random.uniform(BOUND_W_VEL[0], BOUND_W_VEL[1])
    genome[4] = np.random.uniform(BOUND_W_GROUND[0], BOUND_W_GROUND[1])
    genome[5] = np.random.uniform(BOUND_TONIC[0], BOUND_TONIC[1])
    genome[6] = np.random.uniform(BOUND_BETA[0], BOUND_BETA[1])
    genome[7] = np.random.uniform(BOUND_THRESH[0], BOUND_THRESH[1])
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
        
    batched_snn = LobulaColumnarSNN(GA_POPULATION_SIZE)
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
    
    lab_rect = (ARENA_WIDTH, 0, LAB_WIDTH, 680)
    brain_visualizer = BrainVisualizer(lab_rect)
    generation = 1
    max_fitness_history = []
    all_time_record = 0
    all_time_record_seed = current_seed
    all_time_action_tape = set()
    best_overall_genome = None
    
    agent_scores_per_seed = np.zeros((GA_POPULATION_SIZE, len(EVAL_SEEDS)), dtype=np.float32)
    stagnant_generations = 0
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
                            batched_snn = LobulaColumnarSNN(1)
                            batched_snn.set_genomes([best_overall_genome])
                            batched_snn.reset_states()
                            world = SwarmWorld(batched_snn.genomes, assets)
                            world.reset(all_time_record_seed)
                            paused = False
                            speed_multiplier = 1
                        else:
                            print("No champion genome to replay yet!")
                            replay_mode = False
                    else:
                        print("Exiting Replay Mode. Resuming evolution...")
                        batched_snn = LobulaColumnarSNN(GA_POPULATION_SIZE)
                        batched_snn.set_genomes(initial_genomes)
                        batched_snn.reset_states()
                        world = SwarmWorld(batched_snn.genomes, assets)
                        world.reset(current_seed)
                        paused = False
                elif event.key == pygame.K_s:
                    if best_overall_genome is not None:
                        np.save("champion_genome.npy", best_overall_genome)
                        print("Saved champion genome.")
                elif event.key == pygame.K_F11:
                    is_fullscreen = screen.get_flags() & pygame.FULLSCREEN
                    if is_fullscreen:
                        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
                    else:
                        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.FULLSCREEN | pygame.SCALED)

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
                        agent_scores_per_seed[i, current_eval_idx] = fit
                        if fit > best_run_scores[i]:
                            best_run_scores[i] = fit
                            best_run_tapes[i] = set(a.action_tape)
                            best_run_seeds[i] = current_seed
                            
                    current_eval_idx += 1
                    
                    if current_eval_idx < len(EVAL_SEEDS):
                        current_seed = EVAL_SEEDS[current_eval_idx]
                        world.reset(current_seed)
                        batched_snn.reset_states()
                        break
                    else:
                        s1 = agent_scores_per_seed[:, 0]
                        s2 = agent_scores_per_seed[:, 1]
                        # Harmonic Scoring: severely punishes extreme specialists
                        avg_fitnesses = 2.0 * (s1 * s2) / (s1 + s2 + 1e-5)
                        best_idx = int(np.argmax(avg_fitnesses))
                        gen_max_fitness = avg_fitnesses[best_idx]
                        max_fitness_history.append(gen_max_fitness)
                        
                        if gen_max_fitness > all_time_record:
                            all_time_record = gen_max_fitness
                            all_time_record_seed = best_run_seeds[best_idx]
                            all_time_action_tape = best_run_tapes[best_idx]
                            best_overall_genome = batched_snn.genomes[best_idx].copy()
                            stagnant_generations = 0
                            print(f"New All-Time Record: {all_time_record:.1f} (Seed: {all_time_record_seed})")
                        else:
                            stagnant_generations += 1
                            
                        print(f"Gen {generation} | Max Fit: {gen_max_fitness:.1f} | Avg Fit: {np.mean(avg_fitnesses):.1f}")
                        
                        sorted_indices = np.argsort(avg_fitnesses)[::-1]
                        elite_indices = sorted_indices[:GA_ELITE_COUNT]
                        
                        mut_rate = max(MIN_MUT_RATE, INITIAL_MUT_RATE * (DECAY_RATE ** generation))
                        mut_scale = max(MIN_MUT_SCALE, INITIAL_MUT_SCALE * (DECAY_RATE ** generation))
                        
                        if stagnant_generations >= 3:
                            print("Diversity Pulse Triggered!")
                            mut_scale = 0.25
                            stagnant_generations = 0
                            
                            # Keep elites
                            new_genomes = [batched_snn.genomes[idx].copy() for idx in elite_indices]
                            
                            # Re-randomize bottom 30% (12 agents)
                            num_random = int(GA_POPULATION_SIZE * 0.3)
                            
                            # Fill the rest with mutated tournaments
                            while len(new_genomes) < GA_POPULATION_SIZE - num_random:
                                tourney = np.random.choice(GA_POPULATION_SIZE, 3, replace=False)
                                winner_idx = tourney[np.argmax(avg_fitnesses[tourney])]
                                
                                child = batched_snn.genomes[winner_idx].copy()
                                mask = np.random.rand(TOTAL_GENOME_SIZE) < mut_rate
                                mutations = np.random.normal(0, mut_scale, TOTAL_GENOME_SIZE)
                                child += mask * mutations
                                
                                child[0] = np.clip(child[0], BOUND_W_CLIMB[0], BOUND_W_CLIMB[1])
                                child[1] = np.clip(child[1], BOUND_W_DIVE[0], BOUND_W_DIVE[1])
                                child[2] = np.clip(child[2], BOUND_W_LOOMING[0], BOUND_W_LOOMING[1])
                                child[3] = np.clip(child[3], BOUND_W_VEL[0], BOUND_W_VEL[1])
                                child[4] = np.clip(child[4], BOUND_W_GROUND[0], BOUND_W_GROUND[1])
                                child[5] = np.clip(child[5], BOUND_TONIC[0], BOUND_TONIC[1])
                                child[6] = np.clip(child[6], BOUND_BETA[0], BOUND_BETA[1])
                                child[7] = np.clip(child[7], BOUND_THRESH[0], BOUND_THRESH[1])
                                new_genomes.append(child)
                                
                            # Add random agents
                            for _ in range(num_random):
                                new_genomes.append(generate_random_genome())
                                
                            batched_snn.set_genomes(new_genomes)
                        else:
                            batched_snn.reproduce_and_mutate(elite_indices, avg_fitnesses, mut_rate, mut_scale)
                        
                        initial_genomes = [batched_snn.genomes[i].copy() for i in range(GA_POPULATION_SIZE)]
                        
                        generation += 1
                        current_eval_idx = 0
                        current_seed = EVAL_SEEDS[current_eval_idx]
                        
                        agent_scores_per_seed.fill(0)
                        best_run_tapes = [None] * GA_POPULATION_SIZE
                        best_run_seeds = [None] * GA_POPULATION_SIZE
                        best_run_scores = [-1] * GA_POPULATION_SIZE
                        
                        world = SwarmWorld(batched_snn.genomes, assets)
                        world.reset(current_seed)
                        batched_snn.reset_states()
                        break 
                
                offscreen_surf = world.render_for_vision()
                
                N_active = len(world.agents)
                inputs = np.zeros((N_active, 4), dtype=np.float32)
                y_positions = np.zeros(N_active, dtype=np.float32)
                
                leader = world.get_leader()
                leader_idx = world.agents.index(leader) if leader else -1
                
                # Single-pass vision for HUD only
                vision_x = leader.x if leader else 60.0
                shared_grid, disp_matrix = extract_forward_binary_grid(offscreen_surf, vision_x)
                
                for i, agent in enumerate(world.agents):
                    if not agent.alive:
                        continue
                        
                    closest_pipe = next((p for p in world.pipes if p.x + p.width > agent.x), None)
                    if closest_pipe:
                        looming = max(0.0, 1.0 - (max(0, closest_pipe.x - agent.x) / 300.0))
                        gap_offset = (agent.y - closest_pipe.gap_y) / 200.0
                    else:
                        looming = 0.0
                        gap_offset = (agent.y - 250.0) / 200.0
                        
                    vel = np.clip(agent.velocity / 10.0, -1.0, 1.0)
                    ground = max(0.0, (agent.y - 380.0) / 100.0)
                        
                    # Build 4-element input
                    inputs[i, 0] = looming
                    inputs[i, 1] = gap_offset
                    inputs[i, 2] = vel
                    inputs[i, 3] = ground
                    
                    y_positions[i] = agent.y
                    
                flaps = batched_snn.step_batch(inputs, y_positions)
                
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
        
        # Center Deck: Neurophysiology Lab
        lab_x = ARENA_WIDTH
        for y in range(0, WINDOW_HEIGHT, 40):
            pygame.draw.line(screen, COLOR_GRID, (lab_x, y), (lab_x + LAB_WIDTH, y))
        for x in range(lab_x, lab_x + LAB_WIDTH, 40):
            pygame.draw.line(screen, COLOR_GRID, (x, 0), (x, WINDOW_HEIGHT))
            
        pygame.draw.line(screen, COLOR_ACCENT, (lab_x, 0), (lab_x, WINDOW_HEIGHT), 3)
        
        # Right Deck: Telemetry
        hud_x = ARENA_WIDTH + LAB_WIDTH
        for y in range(0, WINDOW_HEIGHT, 40):
            pygame.draw.line(screen, COLOR_GRID, (hud_x, y), (WINDOW_WIDTH, y))
        for x in range(hud_x, WINDOW_WIDTH, 40):
            pygame.draw.line(screen, COLOR_GRID, (x, 0), (x, WINDOW_HEIGHT))
            
        pygame.draw.line(screen, COLOR_ACCENT, (hud_x, 0), (hud_x, WINDOW_HEIGHT), 3)
        
        alive_count = sum(1 for a in world.agents if a.alive)
        leader = world.get_leader()
        current_score = leader.get_fitness() if leader else 0
        
        # Update and Draw Brain Visualizer
        if leader and leader_idx != -1:
            leader_inputs = inputs[leader_idx]
            v = float(batched_snn.gf_v[leader_idx, 0])
            spiked = bool(flaps[leader_idx])
            brain_visualizer.update_and_draw(screen, leader_inputs, v, spiked)
        
        mode_text = "REPLAY MODE" if replay_mode else f"GEN: {generation} | SEED: {current_eval_idx+1}/{len(EVAL_SEEDS)}"
        
        mut_rate = max(MIN_MUT_RATE, INITIAL_MUT_RATE * (DECAY_RATE ** generation))
        mut_scale = max(MIN_MUT_SCALE, INITIAL_MUT_SCALE * (DECAY_RATE ** generation))
        
        if leader and leader_idx != -1:
            s1 = int(agent_scores_per_seed[leader_idx, 0]) if current_eval_idx > 0 else int(current_score)
            s2 = int(current_score) if current_eval_idx == 1 else 0
        else:
            s1, s2 = 0, 0
            
        y_coords = [20, 50, 80, 110, 140, 170]
        texts = [
            mode_text,
            f"ALIVE: {alive_count} / {len(world.agents)}",
            f"SCORE 101: {s1} | SCORE 202: {s2}",
            f"ALL-TIME RECORD: {int(all_time_record)}",
            f"MUT RATE: {mut_rate:.3f} | SCALE: {mut_scale:.3f}",
            f"SIM SPEED: {speed_multiplier}X {'(PAUSED)' if paused else ''}"
        ]
        
        for i, t in enumerate(texts):
            color = COLOR_LEADER if replay_mode and i == 0 else (COLOR_PHOSPHOR if i == 0 else COLOR_TEXT)
            surf = font.render(t, True, color)
            screen.blit(surf, (hud_x + 10, y_coords[i]))
            
        # Fitness Graph (Right Deck)
        g_label = font.render("MULTI-SEED FITNESS HISTORY", True, COLOR_TEXT)
        screen.blit(g_label, (hud_x + 10, 220))
        
        graph_rect = pygame.Rect(hud_x + 10, 245, 320, 100)
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
            
        # Leader Brain View (Right Deck)
        lb_label = font.render("COMPOUND EYE (8x8 BINARY)", True, COLOR_TEXT)
        screen.blit(lb_label, (hud_x + 10, 370))
        
        if leader:
            heatmap_rgb = get_colored_heatmap(disp_matrix)
            eye_surf = pygame.surfarray.make_surface(heatmap_rgb)
            eye_surf = pygame.transform.scale(eye_surf, (128, 128))
            screen.blit(eye_surf, (hud_x + 10, 395))
            
        # Hotkeys (Right Deck)
        keys_label = font.render("HOTKEYS:", True, COLOR_ACCENT)
        screen.blit(keys_label, (hud_x + 10, 560))
        hotkeys = [
            "[P] Pause/Unpause",
            "[R] Toggle Replay Champion",
            "[1,2,5,0] Set Sim Speed",
            "[F11] Toggle Fullscreen"
        ]
        for i, hk in enumerate(hotkeys):
            surf = font.render(hk, True, COLOR_TEXT)
            screen.blit(surf, (hud_x + 10, 585 + i*25))
            
        # Giant Fiber Oscilloscope (Center Deck - Bottom)
        osc_label = font.render("GIANT FIBER VOLTAGE (Vm)", True, COLOR_TEXT)
        screen.blit(osc_label, (lab_x + 10, 680))
        
        osc_rect = pygame.Rect(lab_x + 10, 705, 520, 80)
        pygame.draw.rect(screen, (0, 0, 0), osc_rect)
        pygame.draw.rect(screen, COLOR_GRID, osc_rect, 1)
        
        if leader:
            hist = [float(v[0, 0]) for v in batched_snn.voltage_history[-260:]] if replay_mode else [float(v[leader_idx, 0]) for v in batched_snn.voltage_history[-260:]]
            min_v, max_v = -80.0, -40.0
            v_range = max_v - min_v
            
            if len(hist) > 1:
                pts = []
                for i, v in enumerate(hist):
                    x = float(osc_rect.x + (i / max(1, len(hist)-1)) * osc_rect.width)
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
