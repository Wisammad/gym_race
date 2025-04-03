@echo off
echo Activating conda environment and running RL training...

:: Activate conda
call %USERPROFILE%\miniconda3\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
  echo Failed to activate conda. Please run setup_conda.bat first.
  exit /b 1
)

:: Activate the RL environment
call conda activate rl_gym
if %ERRORLEVEL% NEQ 0 (
  echo Failed to activate rl_gym environment. Please run setup_conda.bat first.
  exit /b 1
)

:: Run the training script
echo Starting training script...
python train_ddpg_overnight.py

echo Training complete or interrupted.
pause 