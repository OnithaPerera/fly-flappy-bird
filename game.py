import pygame
import random
import math
import numpy as np
from config import (ARENA_WIDTH, WINDOW_HEIGHT, GRAVITY, FLAP_STRENGTH, 
                    PIPE_SPEED, PIPE_SPACING, PIPE_GAP, CEILING_DEATH_PENALTY)

class FlyAgent:
    def __init__(self, brain):
        self.brain = brain
        self.x = 50
        self.y = WINDOW_HEIGHT // 2
        self.velocity = 0
        self.rect = pygame.Rect(self.x - 10, self.y - 10, 20, 20)
        self.is_flapping = False
        self.flap_timer = 0
        self.alive = True
        self.death_frame = -1
        
        # Fitness tracking
        self.score = 0
        self.frames_survived = 0
        self.ceiling_hits = 0
        self.gap_alignment_reward = 0.0
        self.oscillation_penalty = 0.0
        self.action_tape = []
        self.last_velocity_sign = 0
        
        # Generate lineage color
        self.color = self._generate_color()

    def _generate_color(self):
        # Map parameters to RGB colors
        r = int(np.clip((self.brain.beta - 0.5) / 0.48, 0, 1) * 255)
        g = int(np.clip((self.brain.v_thresh + 60) / 25, 0, 1) * 255)
        b = 150 # Default for 3rd component
        return (r, g, b)

    def flap(self):
        if not self.alive: return
        self.velocity = FLAP_STRENGTH
        self.is_flapping = True
        self.flap_timer = 5

    def update(self):
        if not self.alive: return
        
        current_sign = 1 if self.velocity > 0 else (-1 if self.velocity < 0 else 0)
        if self.last_velocity_sign != 0 and current_sign != 0 and current_sign != self.last_velocity_sign:
            self.oscillation_penalty += 15.0
        self.last_velocity_sign = current_sign
        
        self.velocity += GRAVITY
        self.y += self.velocity
        self.rect.y = int(self.y - 10)
        self.frames_survived += 1
        
        if self.flap_timer > 0:
            self.flap_timer -= 1
        else:
            self.is_flapping = False
            
        if self.y <= 0:
            self.y = 0
            self.velocity = 0
            self.rect.y = 0
            self.alive = False
            self.ceiling_hits += 1

    def get_fitness(self):
        return self.frames_survived + (self.score * 1500) + self.gap_alignment_reward - self.oscillation_penalty - (self.ceiling_hits * CEILING_DEATH_PENALTY)

    def draw(self, surface, assets, is_leader=False):
        if not self.alive: return
        
        alpha = 89 if not is_leader else 255  # ~35% alpha = 89
        
        angle = -math.degrees(math.atan2(self.velocity, 10))
        angle = max(min(angle, 30), -45)
        
        if self.is_flapping:
            bird_surf = assets["bird_up"].copy()
        elif self.velocity > 0:
            bird_surf = assets["bird_down"].copy()
        else:
            bird_surf = assets["bird_mid"].copy()
            
        # Tint the bird based on genome
        tint = pygame.Surface(bird_surf.get_size(), flags=pygame.SRCALPHA)
        tint.fill((*self.color, 100))
        bird_surf.blit(tint, (0,0), special_flags=pygame.BLEND_RGBA_MULT)
        
        bird_surf.set_alpha(alpha)
        
        rotated_bird = pygame.transform.rotate(bird_surf, angle)
        rect = rotated_bird.get_rect(center=(int(self.x), int(self.y)))
        
        surface.blit(rotated_bird, rect)
        
        if is_leader:
            pygame.draw.circle(surface, (255, 179, 0, 150), (int(self.x), int(self.y)), 30, 2)

class PipePair:
    def __init__(self, x_pos, pipe_surf, ground_h):
        self.x = x_pos
        self.pipe_surf = pipe_surf
        self.width = pipe_surf.get_width()
        
        min_y = 100
        max_y = WINDOW_HEIGHT - ground_h - 100
        self.gap_y = random.randint(min_y, max_y)
        
        # Bottom pipe
        bottom_y = self.gap_y + PIPE_GAP // 2
        self.bottom_rect = pygame.Rect(self.x, bottom_y, self.width, WINDOW_HEIGHT - bottom_y)
        
        # Top pipe
        self.top_rect = pygame.Rect(self.x, 0, self.width, self.gap_y - PIPE_GAP // 2)
        
        self.passed = False

    def update(self):
        self.x -= PIPE_SPEED
        self.top_rect.x = self.x
        self.bottom_rect.x = self.x

    def draw(self, surface):
        # Draw bottom pipe
        surface.blit(self.pipe_surf, (self.top_rect.x, self.bottom_rect.y))
        
        # Draw top pipe (flipped vertically)
        flipped_pipe = pygame.transform.flip(self.pipe_surf, False, True)
        
        # We need to slice the bottom part of the flipped pipe so it ends at top_rect.bottom
        pipe_h = flipped_pipe.get_height()
        blit_y = self.top_rect.bottom - pipe_h
        surface.blit(flipped_pipe, (self.top_rect.x, blit_y))

class SwarmWorld:
    def __init__(self, population, assets):
        self.surface = pygame.Surface((ARENA_WIDTH, WINDOW_HEIGHT))
        self.population = population
        self.assets = assets
        self.ground_h = self.assets["ground"].get_height()
        self.ground_x = 0
        self.reset(seed=42)

    def reset(self, seed):
        random.seed(seed)
        self.agents = [FlyAgent(brain) for brain in self.population]
        self.pipes = []
        self.frames = 0
        self.all_dead = False
        
        # Spawn first pipe
        self.pipes.append(PipePair(ARENA_WIDTH + 200, self.assets["pipe"], self.ground_h))
        # Reset python random seed back to normal time-based behavior if we want,
        # but leaving it deterministic for the generation is good too.

    def get_leader(self):
        alive_agents = [a for a in self.agents if a.alive]
        if not alive_agents:
            return None
        return max(alive_agents, key=lambda a: a.get_fitness())
        
    def get_best_agent(self):
        return max(self.agents, key=lambda a: a.get_fitness())

    def step(self, flaps):
        if self.all_dead:
            return
            
        for i, agent in enumerate(self.agents):
            if agent.alive:
                agent.action_tape.append(bool(flaps[i]))
                if flaps[i]:
                    agent.flap()
                agent.update()
                
                # Gap alignment reward
                if agent.alive:
                    closest_pipe = next((p for p in self.pipes if p.x + p.width > agent.x), None)
                    if closest_pipe:
                        gap_center = closest_pipe.gap_y
                        dist = abs(agent.y - gap_center)
                        agent.gap_alignment_reward += max(0.0, 1.0 - dist / 180.0) * 8.0
        
        # Spawn pipes based on distance
        if len(self.pipes) > 0 and (ARENA_WIDTH - self.pipes[-1].x) >= PIPE_SPACING:
            self.pipes.append(PipePair(ARENA_WIDTH, self.assets["pipe"], self.ground_h))
            
        for pipe in self.pipes:
            pipe.update()
            
            if not pipe.passed:
                first_alive = next((a for a in self.agents if a.alive), None)
                if first_alive and pipe.x + pipe.width < first_alive.x:
                    pipe.passed = True
                    for agent in self.agents:
                        if agent.alive:
                            agent.score += 1
                
        self.pipes = [p for p in self.pipes if p.x + p.width > 0]
        
        # Update ground
        self.ground_x = (self.ground_x - PIPE_SPEED) % -ARENA_WIDTH
        
        alive_count = 0
        for agent in self.agents:
            if not agent.alive: continue
            
            if agent.y >= WINDOW_HEIGHT - self.ground_h - 10:
                agent.alive = False
                agent.death_frame = self.frames
                
            for pipe in self.pipes:
                if agent.rect.colliderect(pipe.top_rect) or agent.rect.colliderect(pipe.bottom_rect):
                    agent.alive = False
                    agent.death_frame = self.frames
                    
            if agent.alive:
                alive_count += 1
                
        if alive_count == 0:
            self.all_dead = True
                
        self.frames += 1

    def render(self):
        # Draw background
        self.surface.blit(self.assets["background"], (0, 0))
        
        for pipe in self.pipes:
            pipe.draw(self.surface)
            
        # Draw ground
        self.surface.blit(self.assets["ground"], (self.ground_x, WINDOW_HEIGHT - self.ground_h))
        self.surface.blit(self.assets["ground"], (self.ground_x + ARENA_WIDTH, WINDOW_HEIGHT - self.ground_h))
            
        leader = self.get_leader()
        for agent in self.agents:
            if agent.alive:
                agent.draw(self.surface, self.assets, is_leader=(agent == leader))
        
        return self.surface
