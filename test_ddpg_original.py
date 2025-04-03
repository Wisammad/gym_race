import gymnasium as gym
import pyrace_continuous_wrapper
from stable_baselines3 import DDPG
import time
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

# Initialize pygame
pygame.init()

# Create environment with rendering
env = gym.make("Pyrace-v3", render_mode="human")

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
model_name = "ddpg_pyrace_v3"
try:
    model = DDPG.load(model_name)
    print(f"Loaded model from {model_name}")
except FileNotFoundError:
    print(f"Model file {model_name} not found. Running with random actions instead.")
    model = None

# Test the model
episodes = 5
try:
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        # Reset mode to 0 after reset
        pyrace = find_pyrace_attr(env)
        if pyrace and hasattr(pyrace, 'mode'):
            pyrace.mode = 0
            pyrace.is_render = True
            # Force pygame display update
            pygame.display.flip()
        
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
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            
            # Take step in environment
            obs, reward, done, trunc, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Explicitly render and update screen
            env.render()
            pygame.display.flip()
            
            # Ensure mode is still 0
            if pyrace and hasattr(pyrace, 'mode'):
                pyrace.mode = 0
                pyrace.is_render = True
            
            # Print progress
            print(f"Step {steps}: Action = [{action[0]:.2f}, {action[1]:.2f}]", end="\r")
            
            # Slow down visualization
            time.sleep(0.05)
            
            if done or trunc:
                break
        
        print(f"Episode {episode+1}: Steps: {steps}, Reward: {total_reward:.2f}")

except KeyboardInterrupt:
    print("Testing interrupted by user")
except Exception as e:
    print(f"Error during testing: {e}")
finally:
    try:
        env.close()
        pygame.quit()
        print("Environment closed")
    except:
        print("Failed to close environment cleanly")

print("Testing completed!") 