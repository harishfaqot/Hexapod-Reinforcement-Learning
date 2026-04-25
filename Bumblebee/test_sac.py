from stable_baselines3 import SAC
from hexapod_env import *

# Buat environment
env = HexapodEnv(is_train=False)

# Load model yang sudah dilatih
model = SAC.load("log_hexapod/model_sac.zip")

if model.observation_space.shape != env.observation_space.shape:
    raise RuntimeError(
        f"Model obs shape {model.observation_space.shape} is not compatible with env "
        f"obs shape {env.observation_space.shape}. Retrain SAC with current hexapod_env.py first."
    )

if model.action_space.shape != env.action_space.shape:
    raise RuntimeError(
        f"Model action shape {model.action_space.shape} is not compatible with env "
        f"action shape {env.action_space.shape}. Retrain SAC with current hexapod_env.py first."
    )

# Coba model untuk mengendalikan robot
obs, _ = env.reset()
for _ in range(100000):
    action, _ = model.predict(obs, deterministic=True)  # SAC works better with deterministic actions during testing
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        obs, _ = env.reset()
        env.save_logs_to_csv("test_log/logs_test_sac.csv")
        break