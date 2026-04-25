# nexabots_pybullet

PyBullet port of the Nexabots policy-gradient workflow for the hexapod.

## What this folder contains

- `src/envs/hexapod_pybullet_env.py`: PyBullet hexapod environment using local bundled assets in `assets/hex_locomotion/hex.xml`
- `src/algos/pg_train.py`: PG/PPO-style trainer and evaluator
- `checkpoints/`: saved trained weights

## Dependencies

Use the workspace virtual environment and install:

```powershell
D:/github/Hexapod-Reinforcement-Learning/.venv/Scripts/python.exe -m pip install pybullet numpy torch
```

## Train

Run from repository root:

```powershell
Set-Location d:/github/Hexapod-Reinforcement-Learning/nexabots_pybullet
D:/github/Hexapod-Reinforcement-Learning/.venv/Scripts/python.exe src/algos/pg_train.py --iters 5000 --batchsize 16 --save-every 500
```

## Resume training

```powershell
Set-Location d:/github/Hexapod-Reinforcement-Learning/nexabots_pybullet
D:/github/Hexapod-Reinforcement-Learning/.venv/Scripts/python.exe src/algos/pg_train.py --resume --iters 5000
```

## Evaluate trained model

```powershell
Set-Location d:/github/Hexapod-Reinforcement-Learning/nexabots_pybullet
D:/github/Hexapod-Reinforcement-Learning/.venv/Scripts/python.exe src/algos/pg_train.py --eval-only --resume
```

## Evaluate with GUI

```powershell
Set-Location d:/github/Hexapod-Reinforcement-Learning/nexabots_pybullet
D:/github/Hexapod-Reinforcement-Learning/.venv/Scripts/python.exe src/algos/pg_train.py --eval-only --resume --gui
```
