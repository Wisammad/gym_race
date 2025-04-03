import gymnasium as gym
import pyrace_continuous_wrapper  # Registers Pyrace-v3
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import numpy as np
import os
import time
import datetime
import signal
import sys
import glob
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Resume DDPG training from a checkpoint")
parser.add_argument('--checkpoint', type=str, help="Path to the checkpoint to resume from")
parser.add_argument('--timesteps', type=int, default=1000000, help="Number of timesteps to train for")
args = parser.parse_args()

# Create log directory with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"./ddpg_pyrace_tensorboard/resumed_{timestamp}/"
os.makedirs(log_dir, exist_ok=True)

# Create models directory for checkpoints with timestamp
models_dir = f"./ddpg_models/resumed_{timestamp}/"
os.makedirs(models_dir, exist_ok=True)

# Create a log file
log_file = os.path.join(models_dir, "training_log.txt")

# Define training parameters
TOTAL_TIMESTEPS = args.timesteps  # Default to 1M steps or use command line
SAVE_FREQ = 10000  # Save a checkpoint every 10k steps
LEARNING_RATE = 3e-4
BATCH_SIZE = 512  # Larger batch size for better stability
BUFFER_SIZE = 200000  # Larger replay buffer
EXPLORATION_NOISE = 0.3  # Higher exploration noise

# Logging helper
def log_message(message):
    print(message)
    with open(log_file, "a") as f:
        f.write(f"{datetime.datetime.now()}: {message}\n")

# Custom callback to log performance metrics
class LoggingCallback(BaseCallback):
    def __init__(self, log_interval=1000, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.start_time = time.time()
        
    def _on_step(self):
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
        return True

# Handle graceful shutdown on Ctrl+C
def signal_handler(sig, frame):
    log_message("Received interrupt signal. Saving model before exiting...")
    try:
        model.save(os.path.join(models_dir, "ddpg_pyrace_v3_interrupted"))
        log_message("Model saved successfully after interruption.")
    except Exception as e:
        log_message(f"Error saving model after interruption: {e}")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

try:
    # Find checkpoint to resume from
    checkpoint_path = args.checkpoint
    
    if not checkpoint_path:
        # Try to find the latest checkpoint
        log_message("No checkpoint specified, looking for the latest one...")
        
        # Look in main directory
        if os.path.exists("./ddpg_models/ddpg_pyrace_v3_final.zip"):
            checkpoint_path = "./ddpg_models/ddpg_pyrace_v3_final.zip"
            log_message(f"Found final model: {checkpoint_path}")
        else:
            # Look for checkpoints in any subdirectory
            all_checkpoints = []
            for root, dirs, files in os.walk("./ddpg_models/"):
                for file in files:
                    if file.endswith(".zip") and ("ddpg_pyrace_model_" in file or "ddpg_pyrace_v3_final" in file):
                        all_checkpoints.append(os.path.join(root, file))
            
            if all_checkpoints:
                # Get creation time for each checkpoint
                checkpoint_times = [(path, os.path.getctime(path)) for path in all_checkpoints]
                # Sort by creation time (newest first)
                checkpoint_times.sort(key=lambda x: x[1], reverse=True)
                checkpoint_path = checkpoint_times[0][0]
                log_message(f"Found latest checkpoint: {checkpoint_path}")
            else:
                log_message("No checkpoints found. Please train a model first.")
                exit(1)
    
    log_message(f"Starting with the following parameters:")
    log_message(f"Resuming from checkpoint: {checkpoint_path}")
    log_message(f"Total timesteps: {TOTAL_TIMESTEPS}")
    log_message(f"Learning rate: {LEARNING_RATE}")
    log_message(f"Batch size: {BATCH_SIZE}")
    log_message(f"Buffer size: {BUFFER_SIZE}")
    log_message(f"Exploration noise: {EXPLORATION_NOISE}")
    log_message(f"Models directory: {models_dir}")
    log_message(f"Log directory: {log_dir}")
    
    # Load environment for training (no rendering for speed)
    env = gym.make("Pyrace-v3")
    
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
    
    # Create logging callback
    logging_callback = LoggingCallback(log_interval=1000)
    
    # Load the model from checkpoint
    log_message(f"Loading model from {checkpoint_path}")
    model = DDPG.load(checkpoint_path, env=env, action_noise=action_noise)
    
    # Set parameters explicitly in case they differ from the checkpoint
    model.learning_rate = LEARNING_RATE
    model.batch_size = BATCH_SIZE
    model.buffer_size = BUFFER_SIZE
    
    # Train the model
    log_message("\nResuming training...")
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
    log_message("\nTo evaluate the model, run: python evaluate_ddpg.py")

except Exception as e:
    log_message(f"Error during training: {e}")
    # Try to save the model even if training fails
    try:
        model.save(os.path.join(models_dir, "ddpg_pyrace_v3_error"))
        log_message("Model saved despite error.")
    except:
        log_message("Could not save model after error.")
    raise 