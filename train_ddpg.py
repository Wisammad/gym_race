import gymnasium as gym
import pygame
import numpy as np
import os
import time
import argparse

# First, make sure pygame is initialized
pygame.init()

# Import our patch to fix PyRace2D rendering
import pyrace_fix

# Now import the wrapper (which will use the patched PyRace2D)
import pyrace_continuous_wrapper  # Registers Pyrace-v3

# Import stable baselines
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train DDPG agent on PyRace environment')
parser.add_argument('--render', action='store_true', help='Enable rendering during training')
args = parser.parse_args()

# Create log directory
log_dir = "./ddpg_pyrace_tensorboard/"
os.makedirs(log_dir, exist_ok=True)

# Create models directory for checkpoints
models_dir = "./ddpg_models/"
os.makedirs(models_dir, exist_ok=True)

# Define training parameters
TOTAL_TIMESTEPS = 100000  # Longer training run
SAVE_FREQ = 10000  # Save a checkpoint every 10k steps
LEARNING_RATE = 3e-4  # Slightly higher learning rate
BATCH_SIZE = 256  # Larger batch size
BUFFER_SIZE = 100000  # Larger replay buffer
EXPLORATION_NOISE = 0.3  # Higher exploration noise

print(f"Training with the following parameters:")
print(f"Total timesteps: {TOTAL_TIMESTEPS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Buffer size: {BUFFER_SIZE}")
print(f"Exploration noise: {EXPLORATION_NOISE}")
print(f"Rendering enabled: {args.render}")

# Helper function to find the pyrace object in the environment
def find_pyrace_attr(env, depth=0, max_depth=5):
    if depth > max_depth:
        return None
    
    if hasattr(env, 'pyrace'):
        return env.pyrace
    
    if hasattr(env, 'env'):
        return find_pyrace_attr(env.env, depth+1)
    
    return None

# Make sure we have a display surface
if pygame.display.get_surface() is None:
    screen = pygame.display.set_mode((1500, 800))
    pygame.display.set_caption("PyRace DDPG Training")

# Load environment with appropriate render mode
env = gym.make("Pyrace-v3", render_mode="human" if args.render else None)

# If rendering, ensure visualization mode is correctly set
if args.render:
    # Find PyRace object
    pyrace_obj = find_pyrace_attr(env)
    if pyrace_obj and hasattr(pyrace_obj, 'mode'):
        pyrace_obj.mode = 0
        pyrace_obj.is_render = True
        print("Set PyRace2D visualization mode")
        
    # Set view if it's a RaceEnv
    if hasattr(env, 'env') and hasattr(env.env, 'unwrapped'):
        race_env = env.env.unwrapped
        if hasattr(race_env, 'set_view'):
            race_env.set_view(True)
            print("Set view=True on RaceEnv")
    
    # Reset the environment to initialize visualization
    env.reset()
    
    # Force render
    env.render()
    
    # Update display
    pygame.display.flip()
    print("Initial visualization setup complete")

# Create an action noise object with higher noise for better exploration
action_noise = NormalActionNoise(
    mean=np.zeros(2),  # 2 dimensions: throttle and steering
    sigma=EXPLORATION_NOISE * np.ones(2)
)

# Create checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=SAVE_FREQ,
    save_path=models_dir,
    name_prefix="ddpg_pyrace_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

# Custom callback to handle visualization updates if rendering is enabled
class VisualizationCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.pyrace_obj = find_pyrace_attr(env)
        # Track steps for throttling render frequency
        self.steps = 0
        
    def _init_callback(self) -> None:
        # Initialize any custom callback variables
        pass
    
    def _on_step(self) -> bool:
        # Process pygame events if rendering is enabled
        if args.render and pygame.get_init():
            self.steps += 1
            
            # Process events every few steps to improve performance
            if self.steps % 5 == 0:
                # Process events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            return False
                        elif event.key == pygame.K_m and self.pyrace_obj:
                            # Toggle visualization mode manually
                            self.pyrace_obj.mode = (self.pyrace_obj.mode + 1) % 3
                            print(f"Changed mode to {self.pyrace_obj.mode}")
            
            # Refresh pyrace_obj reference (might change during training)
            self.pyrace_obj = find_pyrace_attr(self.env)
            
            # Ensure mode is set for visualization
            if self.pyrace_obj and hasattr(self.pyrace_obj, 'mode'):
                self.pyrace_obj.mode = 0
                self.pyrace_obj.is_render = True
            
            # Only render on certain steps to improve performance
            if self.steps % 2 == 0:
                try:
                    # Explicitly call render
                    if hasattr(self.env, 'render'):
                        self.env.render()
                        
                    # Force pygame display update
                    pygame.display.flip()
                except Exception as e:
                    print(f"Render error (non-fatal): {e}")
            
        return True

# Setup callbacks
callbacks = []
callbacks.append(checkpoint_callback)

# Add visualization callback if rendering is enabled
if args.render:
    visualization_callback = VisualizationCallback(env)
    callbacks.append(visualization_callback)

# Create the DDPG model with updated hyperparameters
model = DDPG(
    "MlpPolicy",
    env,
    action_noise=action_noise,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=LEARNING_RATE,
    buffer_size=BUFFER_SIZE,
    learning_starts=1000,
    batch_size=BATCH_SIZE,
    gamma=0.99,
    tau=0.005,
    policy_kwargs=dict(
        net_arch=dict(
            pi=[256, 256, 128],  # Actor network with larger layers
            qf=[256, 256, 128]   # Critic network with larger layers
        )
    ),
)

# Train the model
print("\nStarting training...")
start_time = time.time()

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        log_interval=10,
        callback=callbacks
    )
except KeyboardInterrupt:
    print("\nTraining interrupted by user")
except Exception as e:
    print(f"\nError during training: {e}")
finally:
    training_time = time.time() - start_time
    
    # Save the final model
    final_model_path = os.path.join(models_dir, "ddpg_pyrace_v3_final")
    model.save(final_model_path)
    
    print(f"\nTraining completed in {training_time:.2f} seconds!")
    print(f"Final model saved to {final_model_path}")
    print(f"Checkpoints saved to {models_dir}")
    print(f"Training logs saved to {log_dir}")
    print("\nTo evaluate the model, run: python evaluate_ddpg.py")
    
    # Clean up pygame if it was initialized
    if pygame.get_init():
        pygame.quit()
