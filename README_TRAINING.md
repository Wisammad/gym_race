# Overnight Training Instructions for Pyrace-v3 DDPG

This document provides instructions for running an overnight training session (~8 hours) for the DDPG agent on the Pyrace-v3 environment.

## Training Scripts

There are three main scripts you can use:

1. **`train_ddpg_overnight.py`**: The main script for overnight training with robust checkpointing
2. **`resume_training.py`**: For resuming training from any checkpoint if it gets interrupted
3. **`evaluate_any_checkpoint.py`**: For evaluating any saved checkpoint

## Running Overnight Training

To start a fresh overnight training (approx. 8 hours):

```bash
python train_ddpg_overnight.py
```

This will:
- Create a timestamped directory for logs and checkpoints
- Save checkpoints every 10,000 steps
- Handle graceful interruptions with Ctrl+C
- Log progress to a training log file
- Load from previous checkpoints if available

## If Training Gets Interrupted

If the training process gets interrupted for any reason, you can resume from the latest checkpoint:

```bash
python resume_training.py
```

Or from a specific checkpoint:

```bash
python resume_training.py --checkpoint ./ddpg_models/[path_to_checkpoint.zip] --timesteps 1000000
```

## Evaluating Models

To evaluate the best available model:

```bash
python evaluate_any_checkpoint.py
```

Or to evaluate a specific checkpoint:

```bash
python evaluate_any_checkpoint.py --checkpoint ./ddpg_models/[path_to_checkpoint.zip] --episodes 5
```

## Tips for Overnight Training

1. **Minimize other processes** on your machine to maximize training speed
2. Consider running the training in a **terminal multiplexer** like `screen` or `tmux` so you can detach and let it run
3. The training automatically **saves checkpoints** every 10,000 steps
4. If you need to interrupt the training, use **Ctrl+C** to trigger a graceful shutdown that saves the model
5. The training logs estimated time remaining so you can check progress

## Folder Structure

After training, you'll have:

- `./ddpg_models/[timestamp]/`: Contains checkpoint models and log files
- `./ddpg_pyrace_tensorboard/[timestamp]/`: Contains TensorBoard logs

## Viewing TensorBoard Logs

You can view training progress with TensorBoard:

```bash
tensorboard --logdir ./ddpg_pyrace_tensorboard/
``` 