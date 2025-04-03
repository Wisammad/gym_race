import subprocess
import sys
import os

def install_packages():
    # Upgrade pip first
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # Install packages in the correct order
    packages = [
        "numpy==1.23.5",  # Specific version for better compatibility
        "pygame",
        "cloudpickle",
        "torch==1.12.1",  # Older version with better compatibility
        "gymnasium==0.28.1",  # Specific version
        "stable-baselines3==1.8.0"  # Specific version
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")
            return False
    
    return True

if __name__ == "__main__":
    success = install_packages()
    if success:
        print("\nAll packages installed successfully!")
        print("Now you can run: python train_ddpg_overnight.py")
    else:
        print("\nFailed to install some packages. Please check the errors above.") 