import gymnasium as gym
import numpy as np
import pygame
from typing import Optional, Tuple, Dict, Any
from gym_race.envs.race_env import RaceEnv  # Import RaceEnv directly

class PyraceContinuousWrapper(gym.Wrapper):
    """
    Convert the discrete RaceEnv environment to a continuous action space.
    This wrapper maps continuous actions [-1, 1] for each dimension to the 
    appropriate discrete actions in the base environment.
    """
    
    metadata = {'render_modes': ['human'], 'render_fps': 30}
    
    def __init__(self, render_mode=None):
        # Create the base environment
        env = RaceEnv(render_mode=render_mode)
        super().__init__(env)
        
        # Define a continuous action space with 2 dimensions:
        # [0]: steering (-1=left, 0=straight, 1=right)
        # [1]: throttle (-1=brake, 0=coast, 1=accelerate)
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Keep track of the PyRace2D object if available for visualization control
        self.pyrace_obj = None
        
        # Try to access the PyRace2D object for visualization control
        if hasattr(env, 'pyrace'):
            self.pyrace_obj = env.pyrace
            print("PyRace2D patched for improved visualization")
        
        # Store rendering mode
        self._render_mode = render_mode
        
        # Debug info
        self.debug_msgs = []
        self.action_history = []
        self.last_throttle = 0.0
        
        # Initialize pygame if not already
        if self._render_mode == "human" and not pygame.get_init():
            pygame.init()
    
    def map_continuous_to_discrete(self, cont_action):
        """Map continuous actions from [-1, 1] to discrete actions.
        
        Parameters:
        - cont_action: array of shape (2,) 
          cont_action[0]: steering, from -1 (full left) to 1 (full right)
          cont_action[1]: throttle, from -1 (full brake) to 1 (full acceleration)
        
        Returns:
        - discrete action (0, 1, or 2)
          0: accelerate - this should be the default action to encourage forward progress
          1: turn right
          2: turn left
        """
        # Clip actions to valid range
        cont_action = np.clip(cont_action, -1, 1)
        
        # Extract continuous values
        steer_cont, throttle_cont = cont_action
        
        # Bias toward acceleration by default
        default_action = 0  # accelerate
        
        # Threshold for steering action - make it easier to drive straight
        steering_threshold = 0.5
        
        # Only turn if steering is significant
        if abs(steer_cont) > steering_threshold:
            if steer_cont > 0:
                return 1  # Turn right
            else:
                return 2  # Turn left
        else:
            return default_action  # Default to acceleration
    
    def step(self, action):
        # Store action for debugging
        self.action_history.append(action.copy())
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]
        
        # Map continuous actions to discrete action space
        discrete_action = self.map_continuous_to_discrete(action)
        
        # Take step in the environment
        observation, reward, terminated, truncated, info = self.env.step(discrete_action)
        
        # Extract values for debug visualization
        steering, throttle = action
        
        # Apply reward shaping to encourage forward progress
        shaped_reward = reward
        
        # Extra reward for reaching checkpoints - make this the primary reward signal
        checkpoint_bonus = info.get("check", 0) * 300  # Increased from 200
        shaped_reward += checkpoint_bonus
        
        # Bonus for distance covered when not crashing
        if not info.get('crash', False):
            # Bonus proportional to distance
            distance_bonus = info.get('dist', 0) * 0.5  # Small bonus per unit of distance
            shaped_reward += distance_bonus
        
        # Add a speed bonus
        if not info.get('crash', False):
            # Reward faster movement
            speed_bonus = throttle * 20  # Higher throttle = higher reward
            shaped_reward += speed_bonus
        
        # Ensure mode is still 0 for proper visualization if needed
        if self.pyrace_obj and hasattr(self.pyrace_obj, 'mode'):
            self.pyrace_obj.mode = 0
            # Also ensure is_render is True
            self.pyrace_obj.is_render = True
        
        # Update debug messages with the real reward value 
        self.debug_msgs = [
            f"Throttle: {throttle:.2f}",
            f"Steering: {steering:.2f}",
            f"Action: {discrete_action}",
            f"Raw Reward: {reward:.2f}",
            f"Shaped Reward: {shaped_reward:.2f}",
            f"Checkpoints: {info.get('check', 0)}/7",
            f"Distance: {info.get('dist', 0):.1f}",
            f"Crash: {info.get('crash', False)}"
        ]
        
        # If environment supports messages, set them
        if hasattr(self.env, 'set_msgs'):
            self.env.set_msgs(self.debug_msgs)
            
            # Make sure the messages get displayed - some environments may need this
            if hasattr(self.env, 'pyrace') and hasattr(self.env.pyrace, 'set_msgs'):
                self.env.pyrace.set_msgs(self.debug_msgs)
        
        # Directly set messages on PyRace2D object if available
        if self.pyrace_obj and hasattr(self.pyrace_obj, 'set_msgs'):
            self.pyrace_obj.set_msgs(self.debug_msgs)
        
        # Explicitly call render for visualization
        if self._render_mode == "human":
            self.render()
            # Force pygame display update
            pygame.display.flip()
        
        # Add diagnostic info
        info["continuous_action"] = action
        info["discrete_action"] = discrete_action
        info["shaped_reward"] = shaped_reward
        
        # Return standard gym step result
        return observation, shaped_reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        # Reset the environment
        obs, info = self.env.reset(**kwargs)
        
        # Reset action history
        self.action_history = []
        self.last_throttle = 0.0
        
        # Set up visualization if needed
        if self.pyrace_obj and hasattr(self.pyrace_obj, 'mode'):
            self.pyrace_obj.mode = 0
            # Only print this message once during initialization
            if not hasattr(self, '_init_complete'):
                print("Set PyRace2D visualization mode")
                self._init_complete = True
        
        # Clear debug messages
        self.debug_msgs = []
        
        return obs, info
    
    def render(self):
        return self.env.render()
        
    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()
        if pygame.get_init():
            pygame.quit()

# Register the environment
gym.register(
    id="Pyrace-v3",
    entry_point="pyrace_continuous_wrapper:PyraceContinuousWrapper",
    max_episode_steps=500,
)