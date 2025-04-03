import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gym_race.envs.pyrace_2d import PyRace2D
from gym_race.envs.race_env import RaceEnv
import pygame

# Helper function to find the pyrace object in the environment
def find_pyrace_attr(env, depth=0, max_depth=5):
    if depth > max_depth:
        return None
    
    if hasattr(env, 'pyrace'):
        return env.pyrace
    
    if hasattr(env, 'env'):
        return find_pyrace_attr(env.env, depth+1)
    
    return None

class PyraceContinuousEnv(gym.Wrapper):
    """
    Wrapper for PyRace environment that accepts continuous actions.
    Action space: [throttle, steering] where both values are in [-1, 1]
    throttle: -1 = full brake, 1 = full throttle
    steering: -1 = full left, 1 = full right
    """
    
    metadata = {'render_modes': ['human'], 'render_fps': 30}
    
    def __init__(self, render_mode=None):
        # Init pygame first if rendering
        if render_mode == "human":
            # Make sure pygame is initialized
            if not pygame.get_init():
                pygame.init()
                
            # Ensure display is set up
            if pygame.display.get_surface() is None:
                pygame.display.set_mode((1500, 800))
                pygame.display.set_caption("PyRace Continuous Environment")
            
        # Create the original environment
        env = RaceEnv(render_mode=render_mode)
        super().__init__(env)
        
        # Store render mode as an instance variable, not a property
        self._render_mode = render_mode
        
        # Find pyrace object
        self.pyrace_obj = find_pyrace_attr(self.env)
        
        # Force mode to 0 to ensure the track is visible (not black screen)
        if self.pyrace_obj and hasattr(self.pyrace_obj, 'mode'):
            self.pyrace_obj.mode = 0
            # Also force is_render to True
            self.pyrace_obj.is_render = True
            
        # Ensure view is enabled in the RaceEnv
        if hasattr(self.env, 'set_view'):
            self.env.set_view(True)
        
        # Override action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )
        
        # Keep the original observation space
        self.observation_space = env.observation_space
        
        # Store last throttle value for smoother control
        self.last_throttle = 0
        
        # Action history for debugging
        self.action_history = []
        
        # Custom status messages for visualization
        self.debug_msgs = []
        
    def step(self, action):
        """
        Convert continuous actions to discrete actions in a more sophisticated way.
        This version focuses on making efficient use of both throttle and steering
        to navigate the track effectively.
        
        Args:
            action: [throttle, steering] both in range [-1, 1]
        """
        throttle, steering = action
        
        # Store action for debugging
        self.action_history.append((throttle, steering))
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]
        
        # We'll use a simpler approach - determine one discrete action to take
        # but use a more sophisticated mapping that combines both dimensions
        
        # If primarily accelerating and not turning much
        if throttle > 0.5 and abs(steering) < 0.3:
            discrete_action = 0  # Accelerate
            
        # If primarily turning right
        elif steering > 0.3:
            discrete_action = 1  # Turn right
            
        # If primarily turning left
        elif steering < -0.3:
            discrete_action = 2  # Turn left
            
        # Default to acceleration (neutral/forward)
        else:
            discrete_action = 0  # Accelerate
        
        # Take the action in the environment
        obs, reward, done, trunc, info = self.env.step(discrete_action)
        
        # Apply continuous reward shaping based on both throttle and steering
        reward_factor = 1.0
        
        # Throttle contribution - higher throttle = better reward when going straight
        throttle_factor = max(0.1, (throttle + 1) / 2)  # Convert from [-1, 1] to [0.1, 1]
        
        # Steering contribution - penalize excessive steering
        steering_penalty = min(0.5, abs(steering) * 0.3)
        
        # Combined reward factor - emphasize throttle when going straight
        if abs(steering) < 0.3:
            # Going straight - throttle is more important
            reward_factor = 0.5 + 0.5 * throttle_factor
        else:
            # Turning - steering precision is more important
            reward_factor = 0.5 + 0.5 * (1.0 - steering_penalty)
        
        # Apply the reward scaling
        shaped_reward = reward * reward_factor
        
        # Favor moving forward
        if info.get("dist", 0) > 500:
            shaped_reward += 100
            
        # Extra reward for reaching higher checkpoints
        checkpoint_bonus = info.get("check", 0) * 200
        shaped_reward += checkpoint_bonus
        
        # Ensure mode is still 0 for proper visualization
        if self.pyrace_obj and hasattr(self.pyrace_obj, 'mode'):
            self.pyrace_obj.mode = 0
            # Also ensure is_render is True
            self.pyrace_obj.is_render = True
            
        # Update debug messages
        self.debug_msgs = [
            f"Throttle: {throttle:.2f}",
            f"Steering: {steering:.2f}",
            f"Action: {discrete_action}",
            f"Reward: {shaped_reward:.2f}",
            f"Checkpoints: {info.get('check', 0)}/7",
            f"Distance: {info.get('dist', 0):.1f}",
            f"Crash: {info.get('crash', False)}"
        ]
        
        # If environment supports messages, set them
        if hasattr(self.env, 'set_msgs'):
            self.env.set_msgs(self.debug_msgs)
        
        # Explicitly call render for visualization
        if self._render_mode == "human":
            self.render()
            # Force pygame display update
            pygame.display.flip()
        
        # Update last throttle value
        self.last_throttle = throttle
        
        # Add debug information
        info["throttle"] = throttle
        info["steering"] = steering
        info["reward_factor"] = reward_factor
        info["shaped_reward"] = shaped_reward
        
        return obs, shaped_reward, done, trunc, info
    
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        
        # Reinitialize pygame
        if self._render_mode == "human":
            pygame.init()
            
        # Find pyrace object again after reset (might have changed)
        self.pyrace_obj = find_pyrace_attr(self.env)
        
        # Reset action history
        self.action_history = []
        
        # Reset last throttle
        self.last_throttle = 0
        
        # Force mode to 0 to ensure the track is visible (not black screen)
        if self.pyrace_obj and hasattr(self.pyrace_obj, 'mode'):
            self.pyrace_obj.mode = 0
            # Make sure is_render is True
            self.pyrace_obj.is_render = True
            
        # Clear debug messages
        self.debug_msgs = []
        
        # Ensure view is enabled in the RaceEnv
        if hasattr(self.env, 'set_view'):
            self.env.set_view(True)
            
        # Explicitly call render to ensure visualization
        if self._render_mode == "human":
            self.render()
            # Force pygame display update
            pygame.display.flip()
            
        return result
    
    def render(self):
        # Check if environment has been reset
        if not hasattr(self, 'pyrace_obj') or self.pyrace_obj is None:
            # In case render is called before reset, initialize pygame
            if not pygame.get_init():
                pygame.init()
                
            # Make sure we have a display
            if pygame.display.get_surface() is None:
                pygame.display.set_mode((1500, 800))
                pygame.display.set_caption("PyRace Continuous Environment")
            
            # Attempt to find pyrace object or silently return if can't render yet
            self.pyrace_obj = find_pyrace_attr(self.env)
            if not self.pyrace_obj:
                return None
        
        # Explicitly set the mode to 0 before rendering
        if self.pyrace_obj and hasattr(self.pyrace_obj, 'mode'):
            self.pyrace_obj.mode = 0
            # Ensure is_render is True
            self.pyrace_obj.is_render = True
            
        # If environment supports messages, set them again to ensure they're shown
        if hasattr(self.env, 'set_msgs') and self.debug_msgs:
            self.env.set_msgs(self.debug_msgs)
            
        result = self.env.render()
        
        # Force pygame display update
        if self._render_mode == "human":
            pygame.display.flip()
            
        return result

# Register the environment
gym.register(
    id='Pyrace-v3',
    entry_point='pyrace_continuous_wrapper:PyraceContinuousEnv',
    max_episode_steps=1000,
)