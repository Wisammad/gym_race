import gymnasium as gym
import pygame
import numpy as np
import os
import time
import argparse
import datetime
import torch

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
from stable_baselines3.common.type_aliases import TrainFreq
from stable_baselines3.common.utils import TrainFrequencyUnit

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train DDPG model for Pyrace-v3")
parser.add_argument('--timesteps', type=int, default=100000, help="Number of timesteps to train for")
parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate")
parser.add_argument('--batch_size', type=int, default=256, help="Batch size")
parser.add_argument('--buffer_size', type=int, default=300000, help="Replay buffer size")
parser.add_argument('--noise', type=float, default=0.4, help="Exploration noise")
parser.add_argument('--visualize', action='store_true', help="Enable visualization during training")
parser.add_argument('--checkpoint', type=str, default=None, help="Resume from checkpoint")
args = parser.parse_args()

# Create log directory with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"./ddpg_pyrace_tensorboard/{timestamp}/"
os.makedirs(log_dir, exist_ok=True)

# Create models directory for checkpoints
models_dir = f"./ddpg_models/quick_train_{timestamp}/"
os.makedirs(models_dir, exist_ok=True)

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
env = gym.make("Pyrace-v3", render_mode="human" if args.visualize else None)

# If rendering, ensure visualization mode is correctly set
if args.visualize:
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
    sigma=args.noise * np.ones(2)
)

# Create checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=5000,
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
        if args.visualize and pygame.get_init():
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

# Custom callback to log performance metrics and handle visualization
class LoggingCallback(BaseCallback):
    def __init__(self, log_interval=1000, visualize=False, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.start_time = time.time()
        self.visualize = visualize
        self.pyrace_obj = None
        self.episode_count = 0
        self.best_reward = -float('inf')
        
    def _on_training_start(self):
        if self.visualize:
            # Try to find the PyRace2D object for visualization
            self.pyrace_obj = find_pyrace_attr(self.training_env)
            if self.pyrace_obj:
                print("Found PyRace2D object for visualization")
                # Force mode to 0 for visualization
                if hasattr(self.pyrace_obj, 'mode'):
                    self.pyrace_obj.mode = 0
                    print(f"Set mode={self.pyrace_obj.mode} on PyRace2D object")
            else:
                print("Could not find PyRace2D object directly")
        
    def _on_step(self):
        # Handle pygame events for visualization
        if self.visualize and pygame.get_init():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_m and self.pyrace_obj:
                        # Toggle visualization mode manually
                        self.pyrace_obj.mode = (self.pyrace_obj.mode + 1) % 3
                        print(f"Changed visualization mode to {self.pyrace_obj.mode}")
                    elif event.key == pygame.K_q:
                        pygame.quit()
                        return False
            
            # Update pygame display
            pygame.display.flip()
            
            # Small delay to make visualization visible without slowing down too much
            time.sleep(0.01)
        
        # Log progress at intervals
        if self.n_calls % self.log_interval == 0:
            elapsed_time = time.time() - self.start_time
            timesteps_per_second = self.n_calls / elapsed_time
            remaining_steps = args.timesteps - self.n_calls
            estimated_time_remaining = remaining_steps / timesteps_per_second if timesteps_per_second > 0 else 0
            
            # Format estimated time remaining as minutes:seconds
            minutes, seconds = divmod(estimated_time_remaining, 60)
            
            print(f"Steps: {self.n_calls}/{args.timesteps}, "
                 f"FPS: {timesteps_per_second:.1f}, "
                 f"Time remaining: {int(minutes)}m {int(seconds)}s")
            
            # Check if we have new episode info to report
            if len(self.model.ep_info_buffer) > 0:
                # Calculate and display latest rewards
                latest_rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer[-10:]]
                if latest_rewards:
                    avg_reward = sum(latest_rewards) / len(latest_rewards)
                    max_reward = max(latest_rewards)
                    print(f"Latest avg reward: {avg_reward:.2f}, Latest max reward: {max_reward:.2f}")
                    
                    # Save model if we have a new best reward
                    if max_reward > self.best_reward:
                        self.best_reward = max_reward
                        self.model.save(os.path.join(models_dir, f"ddpg_pyrace_best_reward"))
                        print(f"New best reward {max_reward:.2f} - model saved!")
        
        return True

# Setup callbacks
callbacks = []
callbacks.append(checkpoint_callback)

# Add visualization callback if rendering is enabled
if args.visualize:
    visualization_callback = VisualizationCallback(env)
    callbacks.append(visualization_callback)

# Create the DDPG model with improved architecture
train_freq = TrainFreq(1, TrainFrequencyUnit.STEP)
model = DDPG(
    "MlpPolicy",
    env,
    action_noise=action_noise,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=args.learning_rate,
    buffer_size=args.buffer_size,
    learning_starts=2000,  # Fewer steps before starting learning
    batch_size=args.batch_size,
    train_freq=train_freq,
    gradient_steps=1,
    gamma=0.99,
    tau=0.001,  # Slower target network update for stability
    policy_kwargs=dict(
        net_arch=dict(
            pi=[512, 400, 300],  # Deeper actor network
            qf=[512, 400, 300]   # Deeper critic network
        ),
        activation_fn=torch.nn.ReLU  # ReLU activation for better gradients
    ),
)

# Train the model
print("\nStarting training...")
if args.visualize:
    print("Visualization enabled. Press 'm' to toggle visualization mode, 'q' to quit.")

try:
    # Start training
    start_time = time.time()
    
    model.learn(
        total_timesteps=args.timesteps,
        log_interval=10,
        callback=callbacks
    )
    
    # Calculate and print training time
    training_time = time.time() - start_time
    training_minutes = training_time / 60
    
    # Save the final model
    final_model_path = os.path.join(models_dir, "ddpg_pyrace_v3_final")
    model.save(final_model_path)
    
    print(f"\nTraining completed in {training_minutes:.2f} minutes!")
    print(f"Final model saved to {final_model_path}")
    print(f"Best model saved to {os.path.join(models_dir, 'ddpg_pyrace_best_reward')}")
    print(f"Checkpoints saved to {models_dir}")
    
    # Summarize final performance
    if len(model.ep_info_buffer) > 0:
        recent_rewards = [ep_info["r"] for ep_info in model.ep_info_buffer[-10:]]
        recent_lengths = [ep_info["l"] for ep_info in model.ep_info_buffer[-10:]]
        print(f"Final average reward (last 10 episodes): {sum(recent_rewards) / len(recent_rewards):.2f}")
        print(f"Final average episode length (last 10 episodes): {sum(recent_lengths) / len(recent_lengths):.2f}")
    
    print("\nTo evaluate the model, run: python evaluate_any_checkpoint.py --checkpoint", final_model_path)
    
except KeyboardInterrupt:
    print("\nTraining interrupted. Saving model...")
    model.save(os.path.join(models_dir, "ddpg_pyrace_v3_interrupted"))
    print("Model saved after interruption.")
except Exception as e:
    print(f"Error during training: {e}")
    try:
        model.save(os.path.join(models_dir, "ddpg_pyrace_v3_error"))
        print("Model saved despite error.")
    except:
        print("Could not save model after error.")
finally:
    # Clean up pygame if initialized
    if pygame.get_init():
        pygame.quit()
