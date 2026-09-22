import pygame
import random
import math
import numpy as np
from config import (ARENA_WIDTH, WINDOW_HEIGHT, GRAVITY, FLAP_STRENGTH, 
                    PIPE_SPEED, PIPE_SPAWN_FRAMES, PIPE_GAP, 
                    COLOR_WING, COLOR_LEADER, COLOR_BG, COLOR_FLY_EYE)

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
        
        # Generate lineage color
        self.color = self._generate_color()

    def _generate_color(self):
        # Map parameters to RGB colors
        r = int(np.clip((self.brain.beta - 0.5) / 0.48, 0, 1) * 255)
        g = int(np.clip((self.brain.v_thresh + 60) / 25, 0, 1) * 255)
        b = int(np.clip((self.brain.synaptic_gain - 0.005) / 0.095, 0, 1) * 255)
        return (r, g, b)

    def flap(self):
        if not self.alive: return
        self.velocity = FLAP_STRENGTH
        self.is_flapping = True
        self.flap_timer = 5

    def update(self):
        if not self.alive: return
        
        self.velocity += GRAVITY
        self.y += self.velocity
        self.rect.y = int(self.y - 10)
        self.frames_survived += 1
        
        if self.flap_timer > 0:
            self.flap_timer -= 1
        else:
            self.is_flapping = False
            
        if self.y < 10:
            self.y = 10
            self.velocity = 0
            self.rect.y = 0
            self.ceiling_hits += 1

    def get_fitness(self):
        return self.frames_survived + (self.score * 1000) - (self.ceiling_hits * 100)

    def draw(self, surface, is_leader=False):
        if not self.alive: return
        
        alpha = 150 if not is_leader else 255
        temp_surf = pygame.Surface((40, 40), pygame.SRCALPHA)
        cx, cy = 20, 20
        
        if is_leader:
            pygame.draw.circle(temp_surf, (*COLOR_LEADER, 100), (cx, cy), 18)
            pygame.draw.circle(temp_surf, (*COLOR_LEADER, 255), (cx, cy), 18, 1)

        angle = -math.degrees(math.atan2(self.velocity, 10))
        angle = max(min(angle, 30), -45)
        
        pygame.draw.ellipse(temp_surf, (*self.color, alpha), (cx - 10, cy - 6, 20, 12))
        pygame.draw.circle(temp_surf, (*self.color, alpha), (cx + 8, cy), 6)
        pygame.draw.circle(temp_surf, (*COLOR_FLY_EYE, alpha), (cx + 8, cy - 3), 3)
        pygame.draw.circle(temp_surf, (*COLOR_FLY_EYE, alpha), (cx + 8, cy + 3), 3)
        
        if self.is_flapping:
            pygame.draw.ellipse(temp_surf, (*COLOR_WING[:3], alpha), (cx - 5, cy + 2, 12, 8))
        else:
            pygame.draw.ellipse(temp_surf, (*COLOR_WING[:3], alpha), (cx - 8, cy - 12, 12, 8))
            
        surface.blit(temp_surf, (int(self.x) - 20, int(self.y) - 20))


class PipePair:
    def __init__(self):
        self.x = ARENA_WIDTH
        self.width = 50
        min_y = 150
        max_y = WINDOW_HEIGHT - 150
        self.gap_y = random.randint(min_y, max_y)
        
        self.top_rect = pygame.Rect(self.x, 0, self.width, self.gap_y - PIPE_GAP // 2)
        bottom_y = self.gap_y + PIPE_GAP // 2
        self.bottom_rect = pygame.Rect(self.x, bottom_y, self.width, WINDOW_HEIGHT - bottom_y)
        self.passed = False

    def update(self):
        self.x -= PIPE_SPEED
        self.top_rect.x = self.x
        self.bottom_rect.x = self.x

    def draw(self, surface):
        pygame.draw.rect(surface, (0, 200, 100), self.top_rect)
        pygame.draw.rect(surface, (0, 200, 100), self.bottom_rect)


class SwarmWorld:
    def __init__(self, population):
        self.surface = pygame.Surface((ARENA_WIDTH, WINDOW_HEIGHT))
        self.population = population
        self.reset()

    def reset(self):
        self.agents = [FlyAgent(brain) for brain in self.population]
        self.pipes = []
        self.frames = 0
        self.all_dead = False
        
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
                if flaps[i]:
                    agent.flap()
                agent.update()
        
        if self.frames % PIPE_SPAWN_FRAMES == 0:
            self.pipes.append(PipePair())
            
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
        
        alive_count = 0
        for agent in self.agents:
            if not agent.alive: continue
            
            if agent.y >= WINDOW_HEIGHT - 10:
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
        self.surface.fill(COLOR_BG)
        
        for pipe in self.pipes:
            pipe.draw(self.surface)
            
        leader = self.get_leader()
        for agent in self.agents:
            if agent.alive:
                agent.draw(self.surface, is_leader=(agent == leader))
        
        return self.surface
