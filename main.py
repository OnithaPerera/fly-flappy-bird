import sys
import os
# pyrefly: ignore [missing-import]
import pygame
# pyrefly: ignore [missing-import]
import numpy as np
import random
import time
import math

from diagnostics import run_preflight_checks, validate_frame_tensor
run_preflight_checks() 

from config import (WINDOW_WIDTH, WINDOW_HEIGHT, CANVAS_WIDTH, CANVAS_HEIGHT, ARENA_WIDTH, HUD_WIDTH, FPS,
                    COLOR_PANEL, COLOR_PHOSPHOR, COLOR_GRID, COLOR_LEADER, COLOR_TEXT, COLOR_ACCENT,
                    GA_POPULATION_SIZE, GA_ELITE_COUNT, INITIAL_MUT_RATE, MIN_MUT_RATE, INITIAL_MUT_SCALE, MIN_MUT_SCALE, DECAY_RATE,
                    EYE_RES, EVAL_SEEDS, TOTAL_GENOME_SIZE, GROUND_Y,
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
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
    virtual_screen = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT))
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
    
    lab_rect = (ARENA_WIDTH, 0, 500, 750)
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
    replay_finished = False
    
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
                    replay_finished = False
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
                elif event.key == pygame.K_SPACE:
                    if replay_mode and replay_finished:
                        world.reset(all_time_record_seed)
                        batched_snn.reset_states()
                        replay_finished = False
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
                        replay_finished = True
                        paused = True
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
                        gap_offset = (agent.y - closest_pipe.gap_y) / (GROUND_Y * 0.4)
                    else:
                        looming = 0.0
                        gap_offset = (agent.y - (GROUND_Y / 2.0)) / (GROUND_Y * 0.4)
                        
                    vel = np.clip(agent.velocity / 10.0, -1.0, 1.0)
                    ground = max(0.0, (agent.y - (GROUND_Y - 120.0)) / 100.0)
                        
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
        virtual_screen.fill(COLOR_PANEL)
        arena_surface = world.render_for_vision()
        world.render_for_display(arena_surface)
        virtual_screen.blit(arena_surface, (0, 0))
        
        # Left Sub-Deck: Neurophysiology Lab
        lab_x = 740
        for y in range(0, CANVAS_HEIGHT, 40):
            pygame.draw.line(virtual_screen, COLOR_GRID, (lab_x - 20, y), (lab_x + 500, y))
        for x in range(lab_x - 20, lab_x + 500, 40):
            pygame.draw.line(virtual_screen, COLOR_GRID, (x, 0), (x, CANVAS_HEIGHT))
            
        pygame.draw.line(virtual_screen, COLOR_ACCENT, (ARENA_WIDTH, 0), (ARENA_WIDTH, CANVAS_HEIGHT), 3)
        
        # Right Sub-Deck: Telemetry
        hud_x = 1260
        for y in range(0, CANVAS_HEIGHT, 40):
            pygame.draw.line(virtual_screen, COLOR_GRID, (hud_x - 20, y), (CANVAS_WIDTH, y))
        for x in range(hud_x - 20, CANVAS_WIDTH, 40):
            pygame.draw.line(virtual_screen, COLOR_GRID, (x, 0), (x, CANVAS_HEIGHT))
        
        alive_count = sum(1 for a in world.agents if a.alive)
        leader = world.get_leader()
        current_score = leader.get_fitness() if leader else 0
        
        # Update and Draw Brain Visualizer
        if leader and leader_idx != -1:
            leader_inputs = inputs[leader_idx]
            v = float(batched_snn.gf_v[leader_idx, 0])
            spiked = bool(flaps[leader_idx])
            brain_visualizer.update_and_draw(virtual_screen, leader_inputs, v, spiked)
        
        if replay_mode:
            mode_text = f"REPLAYING ALL-TIME CHAMPION (Record: {int(all_time_record)}) | Press [R] to Exit"
        else:
            mode_text = f"GEN: {generation} | SEED: {current_eval_idx+1}/{len(EVAL_SEEDS)}"
        
        mut_rate = max(MIN_MUT_RATE, INITIAL_MUT_RATE * (DECAY_RATE ** generation))
        mut_scale = max(MIN_MUT_SCALE, INITIAL_MUT_SCALE * (DECAY_RATE ** generation))
        
        if leader and leader_idx != -1:
            s1 = int(agent_scores_per_seed[leader_idx, 0]) if current_eval_idx > 0 else int(current_score)
            s2 = int(current_score) if current_eval_idx == 1 else 0
        else:
            s1, s2 = 0, 0
            
        # Collect Telemetry Data
        v_current = float(batched_snn.gf_v[leader_idx, 0]) if leader_idx != -1 else -70.0
        v_thresh_val = float(batched_snn.v_thresh[leader_idx, 0]) if leader_idx != -1 else -50.0
        
        spike_rate = 0.0
        syn_drive = 0.0
        vy = 0.0
        tilt = 0.0
        gap_delta = 0.0
        obs_dist = 0.0
        
        if leader and leader_idx != -1:
            recent_flaps = leader.action_tape[-60:] if len(leader.action_tape) > 0 else []
            if len(recent_flaps) > 0:
                spike_rate = (sum(recent_flaps) / float(len(recent_flaps))) * FPS
                
            looming = inputs[leader_idx, 0]
            gap_offset = inputs[leader_idx, 1]
            ground_val = inputs[leader_idx, 3]
            lpi_act = max(0.0, -gap_offset)
            lplc2_act = max(0.0, gap_offset) + looming + ground_val
            syn_drive = lplc2_act - lpi_act + ground_val
            
            vy = leader.velocity
            tilt = max(-45.0, min(30.0, -math.degrees(math.atan2(vy, 10))))
            
            closest_pipe = next((p for p in world.pipes if p.x + p.width > leader.x), None)
            if closest_pipe:
                gap_delta = leader.y - closest_pipe.gap_y
                obs_dist = max(0.0, closest_pipe.x - leader.x)
                
        gen_div = float(np.var(batched_snn.genomes, axis=0).mean()) if not replay_mode else 0.0

        # Draw Cards
        card_w = 320
        def draw_card(title, lines, x, y):
            pygame.draw.rect(virtual_screen, (17, 34, 51), (x, y, card_w, 35 + len(lines)*20), 0, 4)
            pygame.draw.rect(virtual_screen, (0, 102, 136), (x, y, card_w, 35 + len(lines)*20), 1, 4)
            header = font.render(title, True, (0, 255, 255))
            virtual_screen.blit(header, (x + 10, y + 8))
            for idx, (k, v) in enumerate(lines):
                key_surf = font.render(k, True, COLOR_TEXT)
                val_surf = font.render(v, True, COLOR_PHOSPHOR)
                virtual_screen.blit(key_surf, (x + 10, y + 30 + idx*20))
                virtual_screen.blit(val_surf, (x + card_w - val_surf.get_width() - 10, y + 30 + idx*20))
                
        # 1. Biophysical Telemetry
        draw_card("BIOPHYSICAL TELEMETRY", [
            ("MEMBRANE (Vm):", f"{v_current:.1f} mV"),
            ("THRESHOLD (Vth):", f"{v_thresh_val:.1f} mV"),
            ("SPIKE RATE:", f"{spike_rate:.1f} Hz"),
            ("SYNAPTIC DRIVE:", f"{syn_drive:.2f} Inet")
        ], hud_x + 10, 20)
        
        # 2. Kinematic & Sensorimotor
        vy_dir = "▲ climbing" if vy < 0 else "▼ falling"
        draw_card("KINEMATIC & SENSORIMOTOR", [
            ("VERTICAL VELOCITY (Vy):", f"{abs(vy):.1f} px/f {vy_dir}"),
            ("ATTITUDE TILT:", f"{tilt:.1f}°"),
            ("GAP OFFSET (Δy):", f"{gap_delta:.1f} px"),
            ("OBSTACLE DISTANCE:", f"{obs_dist:.1f} px")
        ], hud_x + 10, 150)
        
        # 3. Evolutionary Population Health
        draw_card("EVOLUTIONARY POPULATION HEALTH", [
            ("GENOME DIVERSITY:", f"σ² = {gen_div:.4f}"),
            ("SEED A / SEED B SCORES:", f"{s1} / {s2}"),
            ("SWARM SURVIVAL:", f"Alive: {alive_count} / {len(world.agents)}")
        ], hud_x + 10, 280)
        
        # Mode Text & Speed
        m_color = COLOR_LEADER if replay_mode else COLOR_TEXT
        m_surf = font.render(mode_text, True, m_color)
        virtual_screen.blit(m_surf, (hud_x + 10, 390))
        sp_surf = font.render(f"SIM SPEED: {speed_multiplier}X {'(PAUSED)' if paused else ''}", True, COLOR_TEXT)
        virtual_screen.blit(sp_surf, (hud_x + 10, 410))
        
        # Fitness Graph (Right Deck)
        g_label = font.render("MULTI-SEED FITNESS HISTORY", True, COLOR_TEXT)
        virtual_screen.blit(g_label, (hud_x + 10, 440))
        
        graph_rect = pygame.Rect(hud_x + 10, 460, 300, 100)
        pygame.draw.rect(virtual_screen, (0, 0, 0), graph_rect)
        pygame.draw.rect(virtual_screen, COLOR_GRID, graph_rect, 1)
        
        if len(max_fitness_history) > 1 and not replay_mode:
            pts = []
            max_val = max(100, max(max_fitness_history))
            min_val = min(max_fitness_history)
            val_range = max(1, max_val - min_val)
            
            for i, val in enumerate(max_fitness_history):
                x = float(graph_rect.x + (i / max(1, len(max_fitness_history) - 1)) * graph_rect.width)
                y = float(graph_rect.y + graph_rect.height - ((val - min_val) / val_range) * graph_rect.height)
                pts.append((x, y))
                
            pygame.draw.lines(virtual_screen, COLOR_ACCENT, False, pts, 2)
            
        # Leader Brain View (Right Deck)
        lb_label = font.render("COMPOUND EYE (8x8 BINARY)", True, COLOR_TEXT)
        virtual_screen.blit(lb_label, (hud_x + 10, 580))
        
        if leader:
            heatmap_rgb = get_colored_heatmap(disp_matrix)
            eye_surf = pygame.surfarray.make_surface(heatmap_rgb)
            eye_surf = pygame.transform.scale(eye_surf, (128, 128))
            virtual_screen.blit(eye_surf, (hud_x + 10, 600))
            
        # Hotkeys (Right Deck)
        keys_label = font.render("HOTKEYS:", True, COLOR_ACCENT)
        virtual_screen.blit(keys_label, (hud_x + 10, 750))
        hotkeys = [
            "[P] Pause/Unpause",
            "[R] Toggle Replay Champion",
            "[1,2,5,0] Set Sim Speed",
            "[F11] Toggle Fullscreen"
        ]
        for i, hk in enumerate(hotkeys):
            surf = font.render(hk, True, COLOR_TEXT)
            virtual_screen.blit(surf, (hud_x + 10, 770 + i*20))
            
        # Giant Fiber Oscilloscope (Center Deck - Bottom)
        osc_label = font.render("GIANT FIBER VOLTAGE (Vm)", True, COLOR_TEXT)
        virtual_screen.blit(osc_label, (lab_x + 10, 755))
        
        osc_rect = pygame.Rect(lab_x + 10, 780, 480, 100)
        pygame.draw.rect(virtual_screen, (0, 0, 0), osc_rect)
        pygame.draw.rect(virtual_screen, COLOR_GRID, osc_rect, 1)
        
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
                    
                pygame.draw.lines(virtual_screen, (10, 150, 10), False, pts, 4)
                pygame.draw.lines(virtual_screen, COLOR_PHOSPHOR, False, pts, 1)
                
            eff_t = float(batched_snn.v_thresh[0,0] if replay_mode else batched_snn.v_thresh[leader_idx,0])
            thresh_y = float(osc_rect.y + osc_rect.height - ((eff_t - min_v) / v_range) * osc_rect.height)
            pygame.draw.line(virtual_screen, COLOR_LEADER, (osc_rect.x, thresh_y), (osc_rect.x + osc_rect.width, thresh_y), 1)

        # Pause Overlay
        if paused:
            overlay = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            virtual_screen.blit(overlay, (0, 0))
            
            if replay_mode and replay_finished:
                pause_text = large_font.render("REPLAY FINISHED - Press [SPACE] to replay again or [R] to return to evolution", True, COLOR_TEXT)
            else:
                pause_text = large_font.render("SIMULATION PAUSED", True, COLOR_TEXT)
            rect = pause_text.get_rect(center=(CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2))
            virtual_screen.blit(pause_text, rect)

        # Scale virtual_screen to actual screen with letterboxing
        win_w, win_h = screen.get_size()
        scale = min(win_w / CANVAS_WIDTH, win_h / CANVAS_HEIGHT)
        new_w = int(CANVAS_WIDTH * scale)
        new_h = int(CANVAS_HEIGHT * scale)
        scaled_surf = pygame.transform.smoothscale(virtual_screen, (new_w, new_h))
        screen.fill((7, 10, 19))
        screen.blit(scaled_surf, ((win_w - new_w) // 2, (win_h - new_h) // 2))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    run_simulation()
