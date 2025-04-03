import gymnasium as gym
import pyrace_continuous_wrapper
from stable_baselines3 import DDPG
import time
import pygame
import argparse
import numpy as np
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description="Test DDPG model for PyRace environment")
parser.add_argument("--model", type=str, required=True, help="Path to the model file")
parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
parser.add_argument("--delay", type=float, default=0.05, help="Delay between steps (seconds)")
parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
args = parser.parse_args()

# Helper function to find the pyrace object in the environment
def find_pyrace_attr(env, depth=0, max_depth=5):
    if depth > max_depth:
        return None
    
    if hasattr(env, 'pyrace'):
        return env.pyrace
    
    if hasattr(env, 'env'):
        return find_pyrace_attr(env.env, depth+1)
    
    return None

# Create environment with rendering
env = gym.make("Pyrace-v3", render_mode="human")

# Initialize pygame
pygame.init()

# Find and set pyrace mode
pyrace = find_pyrace_attr(env)
if pyrace and hasattr(pyrace, 'mode'):
    pyrace.mode = 0
    pyrace.is_render = True
    print(f"Found and set PyRace mode to {pyrace.mode}")
else:
    print("Could not find pyrace.mode attribute - visualization may not work")

# If there's a RaceEnv with set_view method, use it
if hasattr(env, 'env') and hasattr(env.env, 'unwrapped'):
    race_env = env.env.unwrapped
    if hasattr(race_env, 'set_view'):
        race_env.set_view(True)
        print("Set view=True on RaceEnv")

# Load the trained model
print(f"Loading model from: {args.model}")
try:
    # Use custom_objects to handle deserialize errors
    custom_objects = {
        "learning_rate": 0.0003,
        "lr_schedule": lambda _: 0.0003,
        "clip_range": lambda _: 0.2
    }
    model = DDPG.load(args.model, custom_objects=custom_objects)
    print(f"Successfully loaded model from {args.model}")
    
    # Print model information
    print(f"Model architecture: {model.policy}")
    print(f"Action noise: {model.action_noise}")
    
except FileNotFoundError:
    print(f"Model file {args.model} not found. Please check the path.")
    exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Running with random actions instead.")
    model = None

# Setup to track performance
episode_rewards = []
episode_steps = []
checkpoints_reached = []
max_speeds = []

# Test the model
try:
    for episode in range(args.episodes):
        obs, _ = env.reset()
        print(f"Initial observation: {obs}")
        
        done = False
        total_reward = 0
        steps = 0
        checkpoint_count = 0
        speeds = []
        
        # Reset mode to 0 after reset
        pyrace = find_pyrace_attr(env)
        if pyrace and hasattr(pyrace, 'mode'):
            pyrace.mode = 0
            pyrace.is_render = True
            # Update the display
            pygame.display.flip()
        
        print(f"\nEpisode {episode+1}/{args.episodes} starting...")
        
        while not done:
            # Process pygame events to keep window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_m and pyrace:
                        # Toggle visualization mode manually
                        pyrace.mode = (pyrace.mode + 1) % 3
                        print(f"Changed mode to {pyrace.mode}")
                    elif event.key == pygame.K_q:
                        raise KeyboardInterrupt
            
            # Get action from model or random action if model not loaded
            if model:
                action, _ = model.predict(obs, deterministic=args.deterministic)
            else:
                action = env.action_space.sample()
            
            # Take step in environment
            obs, reward, done, trunc, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Track performance metrics
            if 'speed' in info:
                speeds.append(info['speed'])
            if 'check' in info and info['check'] > checkpoint_count:
                checkpoint_count = info['check']
                print(f"Checkpoint {checkpoint_count} reached at step {steps}")
            
            # Explicitly render and update screen
            env.render()
            pygame.display.flip()
            
            # Ensure mode is still 0
            if pyrace and hasattr(pyrace, 'mode'):
                pyrace.mode = 0
            
            # Print progress with detailed action info
            throttle, steering = action
            print(f"Step {steps}: Action = [throttle={throttle:.2f}, steering={steering:.2f}] | Reward: {reward:.2f} | Obs: {obs}", end="\r")
            
            # Slow down visualization
            time.sleep(args.delay)
            
            if done or trunc:
                done_reason = "crash" if info.get('crash', False) else "timeout"
                print(f"\nEpisode ended due to: {done_reason}")
                break
        
        # Record episode stats
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        checkpoints_reached.append(checkpoint_count)
        max_speeds.append(max(speeds) if speeds else 0)
        
        print(f"\nEpisode {episode+1}: Steps: {steps}, Reward: {total_reward:.2f}, Checkpoints: {checkpoint_count}, Max Speed: {max(speeds) if speeds else 0:.2f}")

    # Print summary statistics
    print("\n===== Performance Summary =====")
    print(f"Average steps per episode: {np.mean(episode_steps):.1f}")
    print(f"Average reward per episode: {np.mean(episode_rewards):.1f}")
    print(f"Average checkpoints reached: {np.mean(checkpoints_reached):.1f}")
    print(f"Average max speed: {np.mean(max_speeds):.2f}")
    print(f"Best episode reward: {max(episode_rewards):.1f}")
    print(f"Maximum checkpoints reached: {max(checkpoints_reached)}")

except KeyboardInterrupt:
    print("\nTesting interrupted by user")
except Exception as e:
    print(f"\nError during testing: {e}")
finally:
    try:
        env.close()
        pygame.quit()
        print("Environment closed")
    except:
        print("Failed to close environment cleanly")

print("Testing completed!") 