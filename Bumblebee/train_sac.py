from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from hexapod_env import *
import os

log_dir = "./log_hexapod/"
n_env = 40

def make_env():
    env = HexapodEnv(is_train=True)
    return env

# Learning rate scheduler (optional)
def linear_schedule(initial_value):
    def func(progress_remaining):
        return initial_value * progress_remaining
    return func

if __name__ == '__main__':
    os.makedirs(log_dir, exist_ok=True)
    env = SubprocVecEnv([make_env for _ in range(n_env)])
    env = VecMonitor(env, log_dir)  # Monitor the vectorized env directly
    
    model = SAC("MlpPolicy", env,
                batch_size=256,
                buffer_size=1000000,
                learning_rate=3e-4,
                gamma=0.99,
                verbose=1,
                ent_coef="auto",
                tensorboard_log=log_dir,
                learning_starts=10000,
                train_freq=1,
                gradient_steps=1)
    
    model.learn(total_timesteps=300000)
    # Always overwrite one fixed model path.
    model_path = os.path.join(log_dir, "model_sac.zip")
    model.save(model_path)
    print(f"Model saved to: {model_path}")