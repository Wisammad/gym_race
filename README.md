# Pyrace Continuous Control with DDPG

This project implements a Deep Deterministic Policy Gradient (DDPG) approach to control a race car in the Pyrace environment with continuous actions.

## Environment

`Pyrace-v3` is a wrapper around the original Pyrace environment that accepts continuous actions:
- Action space: `[throttle, steering]` where both values are in `[-1, 1]`
- `throttle`: -1 = full brake, 1 = full throttle
- `steering`: -1 = full left, 1 = full right

## Files

- `pyrace_continuous_wrapper.py`: Wrapper that converts the Pyrace environment to use continuous actions
- `train_ddpg.py`: Script to train the DDPG agent
- `evaluate_ddpg.py`: Script to evaluate the trained agent
- `requirements.txt`: Dependencies required to run the code

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train_ddpg.py
```

This will:
- Train a DDPG agent for 200,000 timesteps
- Save the trained model as `ddpg_pyrace_v3.zip`
- Log training metrics to TensorBoard under `./ddpg_pyrace_tensorboard/`

### Evaluation

```bash
python evaluate_ddpg.py
```

This will:
- Run 5 evaluation episodes with visualization
- Display statistics including average reward, checkpoints reached, and races completed

## TensorBoard Visualization

```bash
tensorboard --logdir ./ddpg_pyrace_tensorboard/
``` 