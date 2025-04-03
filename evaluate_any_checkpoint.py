import gymnasium as gym
import pyrace_continuous_wrapper  # Registers Pyrace-v3
from stable_baselines3 import DDPG
import time
import numpy as np
import os
import pygame
import glob
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Evaluate a trained DDPG model")
parser.add_argument('--checkpoint', type=str, help="Path to the checkpoint to evaluate")
parser.add_argument('--episodes', type=int, default=5, help="Number of episodes to evaluate")
args = parser.parse_args()

# Initialize pygame up front
pygame.init()
print("Pygame initialized")

# Find the best model to evaluate
if args.checkpoint:
    model_path = args.checkpoint
    if not os.path.exists(model_path):
        print(f"Checkpoint {model_path} not found!")
        exit(1)
else:
    # Find the best model - either the final one or the latest checkpoint
    model_paths = [
        "./ddpg_models/ddpg_pyrace_v3_final.zip",  # The final model
        "./ddpg_pyrace_v3.zip"  # Legacy path
    ]
    
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
        model_path = checkpoint_times[0][0]
    else:
        print("No checkpoints found. Please train a model first.")
        exit(1)

print(f"Using model: {model_path}")

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

# Load environment with rendering
env = gym.make("Pyrace-v3", render_mode="human")

# For deep access to the RaceEnv
if hasattr(env, 'env') and hasattr(env.env, 'unwrapped'):
    race_env = env.env.unwrapped
    if hasattr(race_env, 'set_view'):
        race_env.set_view(True)
        print("Set view=True on RaceEnv")

# Try to get the pyrace object for direct access
pyrace_obj = find_pyrace_obj(env)
if pyrace_obj:
    print("Found PyRace2D object for direct visualization")
    # Force mode to 0 for visualization
    if hasattr(pyrace_obj, 'mode'):
        pyrace_obj.mode = 0
        print(f"Set mode={pyrace_obj.mode} on PyRace2D object")
else:
    print("Could not find PyRace2D object directly")

# Load trained model
try:
    model = DDPG.load(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

num_episodes = args.episodes
all_rewards = []
all_checkpoints = []
finished_races = 0

print("\nStarting evaluation...")
print("Press 'm' to toggle visualization mode if the track is not visible")
print("Press 'q' to quit")

for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step = 0
    highest_check = 0
    
    # Track position history for debugging
    positions = []

    print(f"\nEvaluation Episode {episode+1}/{num_episodes}")
    
    # Force mode again after reset
    pyrace_obj = find_pyrace_obj(env)
    if pyrace_obj and hasattr(pyrace_obj, 'mode'):
        pyrace_obj.mode = 0
        
    while not done:
        # Handle pygame events to prevent freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m and pyrace_obj:
                    # Toggle visualization mode manually
                    pyrace_obj.mode = (pyrace_obj.mode + 1) % 3
                    print(f"Changed mode to {pyrace_obj.mode}")
                elif event.key == pygame.K_q:
                    pygame.quit()
                    exit()
                    
        action, _ = model.predict(obs, deterministic=True)
        
        # Print the action being taken
        print(f"Step {step}: Action = [{action[0]:.2f}, {action[1]:.2f}]", end="\r")
        
        # Take a step
        obs, reward, done, _, info = env.step(action)

        # Record car position if possible
        if pyrace_obj and hasattr(pyrace_obj, 'car'):
            if hasattr(pyrace_obj.car, 'center'):
                positions.append(pyrace_obj.car.center)

        # Update stats
        total_reward += reward
        step += 1
        highest_check = max(highest_check, info.get("check", 0))
                
        # Ensure proper delay for visualization
        time.sleep(0.01)
        
        # Explicitly update pygame display
        pygame.display.flip()
        
    # Track metrics
    all_rewards.append(total_reward)
    all_checkpoints.append(highest_check)
    if highest_check == 7:  # Car finished the race
        finished_races += 1
    
    # Print episode summary
    print(f"\nEpisode {episode+1} completed after {step} steps")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Highest checkpoint: {highest_check}/7")
    if highest_check == 7:
        print("Race COMPLETED! 🏁")
    else:
        print("Race not completed")

# Print overall summary
print("\n===== Evaluation Summary =====")
print(f"Model evaluated: {model_path}")
print(f"Average reward: {np.mean(all_rewards):.2f}")
print(f"Average checkpoints reached: {np.mean(all_checkpoints):.2f}/7")
print(f"Races finished: {finished_races}/{num_episodes} ({finished_races/num_episodes*100:.1f}%)")
print("===============================") 