"""
Test script to verify PyRace visualization is working properly
This script will make the car move in random directions to test visualization.
"""
import gymnasium as gym
import pygame
import time
import numpy as np
import sys

# First, make sure pygame is initialized
pygame.init()

# Import our patch to fix PyRace2D rendering
import pyrace_fix

# Now import the wrapper (which will use the patched PyRace2D)
import pyrace_continuous_wrapper

def manual_control_test():
    """Test visualization using manual keyboard control."""
    print("Starting manual control test...")
    
    # Create environment
    env = gym.make("Pyrace-v3", render_mode="human")
    
    # Reset to get initial state
    obs, _ = env.reset()
    
    # Main control loop
    running = True
    action = np.array([0.0, 0.0])  # [throttle, steering]
    step = 0
    
    print("\nControls:")
    print("  Up Arrow: Accelerate")
    print("  Down Arrow: Brake")
    print("  Left/Right Arrows: Steer")
    print("  M: Toggle visualization mode")
    print("  Q: Quit")
    
    while running:
        step += 1
        
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_m:
                    # Find PyRace object to toggle mode
                    for depth in range(5):
                        env_obj = env
                        for _ in range(depth):
                            if hasattr(env_obj, 'env'):
                                env_obj = env_obj.env
                            else:
                                break
                        if hasattr(env_obj, 'pyrace'):
                            pyrace = env_obj.pyrace
                            if hasattr(pyrace, 'mode'):
                                pyrace.mode = (pyrace.mode + 1) % 3
                                print(f"Changed visualization mode to {pyrace.mode}")
                                break
        
        # Get pressed keys for continuous control
        keys = pygame.key.get_pressed()
        
        # Update action based on key presses
        if keys[pygame.K_UP]:
            action[0] = 1.0  # Full throttle
        elif keys[pygame.K_DOWN]:
            action[0] = -1.0  # Full brake
        else:
            action[0] = 0.0  # No throttle/brake
            
        if keys[pygame.K_LEFT]:
            action[1] = -1.0  # Full left
        elif keys[pygame.K_RIGHT]:
            action[1] = 1.0  # Full right
        else:
            action[1] = 0.0  # Center steering
        
        # Take step with current action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Print info
        print(f"\rStep {step} | Action: [{action[0]:+.1f}, {action[1]:+.1f}] | Reward: {reward:+.1f}", end="")
        
        # Render (should be automatic with human render mode, but just to be sure)
        env.render()
        
        # Update display
        pygame.display.flip()
        
        # Exit if done
        if terminated or truncated:
            print("\nEpisode finished!")
            obs, _ = env.reset()
            step = 0
        
        # Cap framerate
        time.sleep(0.05)
    
    # Clean up
    env.close()
    pygame.quit()
    print("\nManual control test completed.")

def automated_test():
    """Test visualization using automated random actions."""
    print("Starting automated visualization test...")
    
    # Create environment
    env = gym.make("Pyrace-v3", render_mode="human")
    
    # Number of episodes to run
    num_episodes = 3
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        
        print(f"\nEpisode {episode+1}/{num_episodes}")
        
        while not done:
            step += 1
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        env.close()
                        pygame.quit()
                        sys.exit()
            
            # Generate smooth random actions (change gradually)
            if step == 1:
                # Initial action
                action = np.array([0.5, 0.0])  # Start with some throttle, no steering
            else:
                # Random changes to previous action with smoothing
                throttle_change = np.random.uniform(-0.1, 0.1)
                steer_change = np.random.uniform(-0.2, 0.2)
                
                action[0] = np.clip(action[0] + throttle_change, -1.0, 1.0)
                action[1] = np.clip(action[1] + steer_change, -1.0, 1.0)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Print info
            print(f"\rStep {step} | Action: [{action[0]:+.1f}, {action[1]:+.1f}] | Reward: {reward:+.1f} | Check: {info.get('check', 0)}/7", end="")
            
            # Force render
            env.render()
            pygame.display.flip()
            
            # Done?
            done = terminated or truncated
            
            # Cap framerate
            time.sleep(0.05)
    
    # Clean up
    env.close()
    pygame.quit()
    print("\nAutomated test completed.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test PyRace visualization")
    parser.add_argument("--mode", choices=["manual", "auto"], default="manual",
                      help="Test mode: manual (keyboard control) or auto (random actions)")
    
    args = parser.parse_args()
    
    try:
        print("PyRace Visualization Test")
        print("-----------------------")
        print("Press 'Q' to quit at any time")
        
        if args.mode == "manual":
            manual_control_test()
        else:
            automated_test()
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        # Make sure pygame is properly shut down
        pygame.quit() 