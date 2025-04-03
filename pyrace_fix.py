"""
Patch for PyRace2D to fix visualization issues
This script monkey patches the PyRace2D class to ensure proper visualization.
"""
import pygame
from gym_race.envs.pyrace_2d import PyRace2D

# Store original init method
original_init = PyRace2D.__init__

# Define patched init method
def patched_init(self, is_render=True, car=True, mode=0):
    # Call original init
    original_init(self, is_render, car, mode)
    
    # Force mode to 0 for visibility
    self.mode = 0
    self.is_render = True
    
    # Ensure proper display initialization
    if not pygame.get_init():
        pygame.init()
    
    # Make sure we have a display
    if pygame.display.get_surface() is None:
        pygame.display.set_mode((1500, 800))
        pygame.display.set_caption("PyRace Environment")

# Store original view_ method
original_view = PyRace2D.view_

# Define patched view method
def patched_view(self, msgs=[]):
    # Process events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_m:
                self.mode += 1
                self.mode = self.mode % 3
            if event.key == pygame.K_p:
                self.mode += 1
                self.mode = self.mode % 3
            elif event.key == pygame.K_q:
                done = True
                exit()
    
    # Force mode to 0 temporarily for rendering
    old_mode = self.mode
    self.mode = 0
    
    # Call original view method
    result = original_view(self, msgs)
    
    # Restore mode
    self.mode = old_mode
    
    # Force screen update
    pygame.display.flip()
    
    return result

# Apply the monkey patches
PyRace2D.__init__ = patched_init
PyRace2D.view_ = patched_view

print("PyRace2D patched for improved visualization") 