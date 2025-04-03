import gymnasium as gym
import pyrace_continuous_wrapper  # Registers Pyrace-v3
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.type_aliases import TrainFreq
from stable_baselines3.common.utils import TrainFrequencyUnit
import numpy as np
import os
import time
import datetime
import signal
import sys
import pygame
import argparse
import torch

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train DDPG model overnight")
parser.add_argument('--visualize', action='store_true', help="Enable visualization during training")
parser.add_argument('--timesteps', type=int, default=2000000, help="Number of timesteps to train for")
args = parser.parse_args()

# Initialize pygame if visualization is enabled
if args.visualize:
    pygame.init()
    print("Pygame initialized for visualization")

# Create log directory with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"./ddpg_pyrace_tensorboard/{timestamp}/"
os.makedirs(log_dir, exist_ok=True)

# Create models directory for checkpoints with timestamp
models_dir = f"./ddpg_models/{timestamp}/"
os.makedirs(models_dir, exist_ok=True)

# Create a log file
log_file = os.path.join(models_dir, "training_log.txt")

# Define training parameters for overnight training
TOTAL_TIMESTEPS = args.timesteps  # Default to 2M steps or use command line arg
SAVE_FREQ = 10000  # Save a checkpoint every 10k steps
LEARNING_RATE = 1e-4  # Reduced learning rate for more stable learning
BATCH_SIZE = 256  # Smaller batch size for better exploration
BUFFER_SIZE = 1000000  # Much larger buffer to remember more experiences
EXPLORATION_NOISE = 0.4  # Increased exploration noise
TRAIN_FREQ = TrainFreq(1, TrainFrequencyUnit.STEP)  # Update the policy every step
GRADIENT_STEPS = 1  # How many gradient steps to do at each update

# Try to find the base PyRace environment
def find_pyrace_obj(env, max_depth=10):
    """Recursively search for the PyRace2D object."""
    if max_depth <= 0:
        return None
    
    if hasattr(env, 'pyrace'):
        return env.pyrace
    
    if hasattr(env, 'env'):
        return find_pyrace_obj(env.env, max_depth-1)
    
    return None

# Logging helper
def log_message(message):
    print(message)
    with open(log_file, "a") as f:
        f.write(f"{datetime.datetime.now()}: {message}\n")

# Log initial parameters
log_message(f"Starting overnight training with the following parameters:")
log_message(f"Total timesteps: {TOTAL_TIMESTEPS}")
log_message(f"Learning rate: {LEARNING_RATE}")
log_message(f"Batch size: {BATCH_SIZE}")
log_message(f"Buffer size: {BUFFER_SIZE}")
log_message(f"Exploration noise: {EXPLORATION_NOISE}")
log_message(f"Models directory: {models_dir}")
log_message(f"Log directory: {log_dir}")
log_message(f"Visualization enabled: {args.visualize}")

# Custom callback to log performance metrics and handle visualization
class LoggingCallback(BaseCallback):
    def __init__(self, log_interval=1000, visualize=False, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.start_time = time.time()
        self.visualize = visualize
        self.pyrace_obj = None
        self.episode_count = 0
        
    def _on_training_start(self):
        if self.visualize:
            # Try to find the PyRace2D object for visualization
            self.pyrace_obj = find_pyrace_obj(self.training_env)
            if self.pyrace_obj:
                log_message("Found PyRace2D object for visualization")
                # Force mode to 0 for visualization
                if hasattr(self.pyrace_obj, 'mode'):
                    self.pyrace_obj.mode = 0
                    log_message(f"Set mode={self.pyrace_obj.mode} on PyRace2D object")
            else:
                log_message("Could not find PyRace2D object directly")
        
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
                        log_message(f"Changed visualization mode to {self.pyrace_obj.mode}")
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
            remaining_steps = TOTAL_TIMESTEPS - self.n_calls
            estimated_time_remaining = remaining_steps / timesteps_per_second if timesteps_per_second > 0 else 0
            
            # Format estimated time remaining as hours:minutes:seconds
            hours, remainder = divmod(estimated_time_remaining, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            log_message(f"Steps: {self.n_calls}/{TOTAL_TIMESTEPS}, "
                        f"FPS: {timesteps_per_second:.1f}, "
                        f"Time remaining: {int(hours)}h {int(minutes)}m {int(seconds)}s")
            
            # Reset the PyRace visualization mode after each episode
            if self.visualize and self.pyrace_obj and hasattr(self.pyrace_obj, 'mode'):
                self.pyrace_obj.mode = 0
        
        return True

# Handle graceful shutdown on Ctrl+C
def signal_handler(sig, frame):
    log_message("Received interrupt signal. Saving model before exiting...")
    try:
        model.save(os.path.join(models_dir, "ddpg_pyrace_v3_interrupted"))
        log_message("Model saved successfully after interruption.")
    except Exception as e:
        log_message(f"Error saving model after interruption: {e}")
    # Clean up pygame if initialized
    if pygame.get_init():
        pygame.quit()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

try:
    # Load environment - with render mode if visualization is enabled
    if args.visualize:
        env = gym.make("Pyrace-v3", render_mode="human")
        
        # For deep access to the RaceEnv
        if hasattr(env, 'env') and hasattr(env.env, 'unwrapped'):
            race_env = env.env.unwrapped
            if hasattr(race_env, 'set_view'):
                race_env.set_view(True)
                log_message("Set view=True on RaceEnv")
    else:
        env = gym.make("Pyrace-v3")  # No rendering for speed
    
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
    
    # Create logging callback with visualization option
    logging_callback = LoggingCallback(log_interval=1000, visualize=args.visualize)
    
    # Create the DDPG model with updated hyperparameters
    model = DDPG(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=5000,  # Wait for more samples before starting to learn
        batch_size=BATCH_SIZE,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
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
    
    # Load previous model if available to continue training
    latest_checkpoint = None
    if os.path.exists("./ddpg_models/ddpg_pyrace_v3_final.zip"):
        latest_checkpoint = "./ddpg_models/ddpg_pyrace_v3_final.zip"
    else:
        checkpoint_files = sorted([f for f in os.listdir("./ddpg_models/") if f.startswith("ddpg_pyrace_model_") and f.endswith(".zip")])
        if checkpoint_files:
            latest_checkpoint = os.path.join("./ddpg_models/", checkpoint_files[-1])
    
    if latest_checkpoint:
        log_message(f"Loading checkpoint {latest_checkpoint} to continue training")
        # Load the model but update its parameters
        model = DDPG.load(latest_checkpoint, env=env)
        # Update the learning rate and other parameters
        model.learning_rate = LEARNING_RATE
        model.batch_size = BATCH_SIZE
        model.buffer_size = BUFFER_SIZE
        model.train_freq = TRAIN_FREQ
        model.gradient_steps = GRADIENT_STEPS
        model.tau = 0.001
        # Update the action noise
        model.action_noise = action_noise
        log_message("Updated model parameters after loading checkpoint")
    
    # Train the model
    log_message("\nStarting overnight training...")
    if args.visualize:
        log_message("Visualization enabled. Press 'm' to toggle visualization mode, 'q' to quit.")
    
    start_time = time.time()
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        log_interval=10,
        callback=[checkpoint_callback, logging_callback]
    )
    
    training_time = time.time() - start_time
    training_hours = training_time / 3600
    
    # Save the final model
    final_model_path = os.path.join(models_dir, "ddpg_pyrace_v3_final")
    model.save(final_model_path)
    
    log_message(f"\nTraining completed in {training_hours:.2f} hours!")
    log_message(f"Final model saved to {final_model_path}")
    log_message(f"Checkpoints saved to {models_dir}")
    log_message(f"Training logs saved to {log_dir}")
    log_message("\nTo evaluate the model, run: python evaluate_any_checkpoint.py")

except Exception as e:
    log_message(f"Error during training: {e}")
    # Try to save the model even if training fails
    try:
        model.save(os.path.join(models_dir, "ddpg_pyrace_v3_error"))
        log_message("Model saved despite error.")
    except:
        log_message("Could not save model after error.")
    # Clean up pygame if initialized
    if pygame.get_init():
        pygame.quit()
    raise 

# Clean up pygame properly at the end
if pygame.get_init():
    pygame.quit() 