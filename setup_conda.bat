@echo off
echo Setting up a Python environment for RL project...

:: Check if Miniconda exists
if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" (
    echo Miniconda already installed
    goto :setup_env
)

:: Download Miniconda
echo Downloading Miniconda...
curl -o miniconda.exe https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe

:: Install Miniconda
echo Installing Miniconda...
start /wait "" miniconda.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%USERPROFILE%\miniconda3

:: Delete installer
del miniconda.exe

:setup_env
:: Add conda to path for this session
set PATH=%USERPROFILE%\miniconda3;%USERPROFILE%\miniconda3\Scripts;%USERPROFILE%\miniconda3\Library\bin;%PATH%

:: Create conda environment
echo Creating conda environment with Python 3.8...
call conda create -y -n rl_gym python=3.8

:: Activate environment
echo Activating environment...
call conda activate rl_gym

:: Install packages
echo Installing required packages...
call conda install -y -c pytorch pytorch=1.12.1 cpuonly
call conda install -y pip
call pip install gymnasium==0.28.1
call pip install pygame==2.1.2
call pip install stable-baselines3==1.8.0

echo.
echo Setup complete! 
echo.
echo To run your script:
echo 1. Open a new command prompt
echo 2. Run: call %USERPROFILE%\miniconda3\Scripts\activate.bat
echo 3. Run: conda activate rl_gym
echo 4. Run: python train_ddpg_overnight.py
echo.
echo Press any key to exit...
pause > nul 