import pygame
import random
from config import WINDOW_WIDTH, WINDOW_HEIGHT, GRAVITY, FLAP_STRENGTH, PIPE_SPEED, PIPE_SPAWN_FRAMES, PIPE_GAP

class Bird:
    def __init__(self):
        self.x = 50
        self.y = WINDOW_HEIGHT // 2
        self.velocity = 0
        self.rect = pygame.Rect(self.x, self.y, 20, 20)

    def flap(self):
        self.velocity = FLAP_STRENGTH

    def update(self):
        self.velocity += GRAVITY
        self.y += self.velocity
        self.rect.y = int(self.y)
        
        # Ceiling collision constraint - zeroes vertical velocity and keeps within screen
        if self.y < 0:
            self.y = 0
            self.velocity = 0
            self.rect.y = 0

    def draw(self, surface):
        pygame.draw.rect(surface, (255, 200, 0), self.rect)


class PipePair:
    def __init__(self):
        self.x = WINDOW_WIDTH
        self.width = 50
        
        # Random gap y position (center of the gap)
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


class FlappyGame:
    def __init__(self):
        self.surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.reset()

    def reset(self):
        self.bird = Bird()
        self.pipes = []
        self.score = 0
        self.frames = 0
        self.game_over = False

    def step(self, flap=False):
        """Advances physics by one tick"""
        if self.game_over:
            return
            
        if flap:
            self.bird.flap()
            
        self.bird.update()
        
        # Spawn pipes
        if self.frames % PIPE_SPAWN_FRAMES == 0:
            self.pipes.append(PipePair())
            
        # Update pipes
        for pipe in self.pipes:
            pipe.update()
            
            # Score
            if not pipe.passed and pipe.x + pipe.width < self.bird.x:
                pipe.passed = True
                self.score += 1
                
        # Remove offscreen pipes
        self.pipes = [p for p in self.pipes if p.x + p.width > 0]
        
        # Check collisions
        if self.bird.y >= WINDOW_HEIGHT - self.bird.rect.height:
            self.game_over = True
            
        for pipe in self.pipes:
            if self.bird.rect.colliderect(pipe.top_rect) or self.bird.rect.colliderect(pipe.bottom_rect):
                self.game_over = True
                
        self.frames += 1

    def render(self):
        """Draws to the offscreen surface and returns it."""
        self.surface.fill((135, 206, 235)) # Sky blue
        
        for pipe in self.pipes:
            pipe.draw(self.surface)
            
        self.bird.draw(self.surface)
        
        return self.surface
