import pygame
import random
import math
from config import (WINDOW_WIDTH, WINDOW_HEIGHT, GRAVITY, FLAP_STRENGTH, 
                    PIPE_SPEED, PIPE_SPAWN_FRAMES, PIPE_GAP, 
                    COLOR_FLY_BODY, COLOR_FLY_EYE, COLOR_WING)

class FlyAgent:
    def __init__(self):
        self.x = 50
        self.y = WINDOW_HEIGHT // 2
        self.velocity = 0
        self.rect = pygame.Rect(self.x - 10, self.y - 10, 20, 20)
        self.is_flapping = False
        self.flap_timer = 0
        self.alive = True
        
        # Fitness tracking
        self.score = 0
        self.frames_survived = 0
        self.ceiling_hits = 0

    def flap(self):
        if not self.alive: return
        self.velocity = FLAP_STRENGTH
        self.is_flapping = True
        self.flap_timer = 5 # Frames to keep wings down

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
            
        # Ceiling collision constraint
        if self.y < 10:
            self.y = 10
            self.velocity = 0
            self.rect.y = 0
            self.ceiling_hits += 1

    def draw(self, surface, alpha=255):
        if not self.alive: return
        
        # We can draw directly to surface if alpha=255, 
        # otherwise create a temp surface for transparency.
        if alpha < 255:
            temp_surf = pygame.Surface((30, 30), pygame.SRCALPHA)
            cx, cy = 15, 15
            target_surf = temp_surf
        else:
            cx, cy = int(self.x), int(self.y)
            target_surf = surface

        # Body (Ellipse)
        angle = -math.degrees(math.atan2(self.velocity, 10))
        angle = max(min(angle, 30), -45) # limit tilt
        
        # Render a simple fly
        pygame.draw.ellipse(target_surf, (*COLOR_FLY_BODY, alpha), (cx - 10, cy - 6, 20, 12))
        # Head
        pygame.draw.circle(target_surf, (*COLOR_FLY_BODY, alpha), (cx + 8, cy), 6)
        # Red Eyes
        pygame.draw.circle(target_surf, (*COLOR_FLY_EYE, alpha), (cx + 8, cy - 3), 3)
        pygame.draw.circle(target_surf, (*COLOR_FLY_EYE, alpha), (cx + 8, cy + 3), 3)
        
        # Wings
        if self.is_flapping:
            # Wings down
            pygame.draw.ellipse(target_surf, COLOR_WING, (cx - 5, cy + 2, 12, 8))
        else:
            # Wings up
            pygame.draw.ellipse(target_surf, COLOR_WING, (cx - 8, cy - 12, 12, 8))
            
        if alpha < 255:
            surface.blit(temp_surf, (int(self.x) - 15, int(self.y) - 15))


class PipePair:
    def __init__(self):
        self.x = WINDOW_WIDTH
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
        pygame.draw.rect(surface, (0, 200, 0), self.top_rect)
        pygame.draw.rect(surface, (0, 200, 0), self.bottom_rect)


class FlappyWorld:
    def __init__(self, num_agents=1):
        self.surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.num_agents = num_agents
        self.reset()

    def reset(self):
        self.agents = [FlyAgent() for _ in range(self.num_agents)]
        self.pipes = []
        self.frames = 0
        self.all_dead = False
        
    def get_best_agent(self):
        return max(self.agents, key=lambda a: a.score * 500 + a.frames_survived - a.ceiling_hits * 50)

    def step(self, flaps):
        """
        Advances physics by one tick.
        flaps is a list of booleans indicating if each agent flapped.
        """
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
                # Assuming all agents share the same X, check the first alive one
                first_alive = next((a for a in self.agents if a.alive), None)
                if first_alive and pipe.x + pipe.width < first_alive.x:
                    pipe.passed = True
                    for agent in self.agents:
                        if agent.alive:
                            agent.score += 1
                
        self.pipes = [p for p in self.pipes if p.x + p.width > 0]
        
        # Collisions
        alive_count = 0
        for agent in self.agents:
            if not agent.alive: continue
            
            if agent.y >= WINDOW_HEIGHT - 10:
                agent.alive = False
                
            for pipe in self.pipes:
                if agent.rect.colliderect(pipe.top_rect) or agent.rect.colliderect(pipe.bottom_rect):
                    agent.alive = False
                    
            if agent.alive:
                alive_count += 1
                
        if alive_count == 0:
            self.all_dead = True
                
        self.frames += 1

    def render(self):
        self.surface.fill((135, 206, 235))
        
        for pipe in self.pipes:
            pipe.draw(self.surface)
            
        # Draw agents
        alpha = 255 if self.num_agents == 1 else max(50, 255 // min(self.num_agents, 5))
        for agent in self.agents:
            if agent.alive:
                agent.draw(self.surface, alpha)
        
        return self.surface
