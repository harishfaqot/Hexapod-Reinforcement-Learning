import argparse
import sys
from pathlib import Path

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.envs.hexapod_pybullet_env import HexapodPyBulletEnv


def to_tensor(x):
    return T.tensor(x, dtype=T.float32).unsqueeze(0)


class NNPG(nn.Module):
    def __init__(self, obs_dim, act_dim, hid_dim=128, std_init=-0.6):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.fc1 = nn.Linear(obs_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, act_dim)

        self.log_std = nn.Parameter(T.full((1, act_dim), std_init, dtype=T.float32))

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)

    def sample_action(self, s):
        mean = self.forward(s)
        std = T.exp(self.log_std)
        return T.normal(mean, std)

    def log_probs(self, batch_states, batch_actions):
        means = self.forward(batch_states)
        log_std = self.log_std.expand_as(means)
        std = T.exp(log_std)
        var = std.pow(2)
        log_density = -((batch_actions - means).pow(2)) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std
        return log_density.sum(1, keepdim=True)


def calc_returns(gamma, rewards, terminals):
    returns = []
    for i in range(len(rewards)):
        g = T.tensor(0.0)
        for j in range(i, len(rewards)):
            g += (gamma ** (j - i)) * rewards[j].squeeze()
            if terminals[j]:
                break
        returns.append(g.view(1, 1))
    return T.cat(returns)


def update_ppo(policy, optim, batch_states, batch_actions, batch_adv, update_iters=8, clip_eps=0.2):
    log_probs_old = policy.log_probs(batch_states, batch_actions).detach()
    for _ in range(update_iters):
        log_probs_new = policy.log_probs(batch_states, batch_actions)
        ratio = T.exp(log_probs_new - log_probs_old)
        loss = -T.mean(T.min(ratio * batch_adv, ratio.clamp(1 - clip_eps, 1 + clip_eps) * batch_adv))
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optim.step()


def evaluate(env, policy, episodes=3, render=False):
    ep_rewards = []
    with T.no_grad():
        for _ in range(episodes):
            obs = env.reset()
            done = False
            total = 0.0
            while not done:
                act = policy(to_tensor(obs)).squeeze(0).cpu().numpy()
                obs, rew, done, _ = env.step(act)
                if render:
                    env.render()
                total += rew
            ep_rewards.append(total)
    return float(np.mean(ep_rewards))


def train(args):
    env = HexapodPyBulletEnv(gui=args.gui, target_vel=args.target_vel, episode_steps=args.episode_steps)
    policy = NNPG(env.obs_dim, env.act_dim, hid_dim=args.hid_dim)
    optim = T.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)

    ckpt = Path(args.checkpoint)
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    if args.resume and ckpt.exists():
        policy.load_state_dict(T.load(str(ckpt), map_location="cpu"))
        print(f"Loaded checkpoint: {ckpt}")

    if args.eval_only:
        score = evaluate(env, policy, episodes=args.eval_episodes, render=args.gui)
        print(f"Eval avg reward: {score:.3f}")
        env.close()
        return

    batch_states = []
    batch_actions = []
    batch_rewards = []
    batch_terminals = []

    ep_in_batch = 0
    reward_acc = 0.0

    for it in range(1, args.iters + 1):
        obs = env.reset()
        done = False

        while not done:
            action = policy.sample_action(to_tensor(obs)).detach()
            nobs, rew, done, _ = env.step(action.squeeze(0).cpu().numpy())
            rew = float(np.clip(rew, -3.0, 3.0))

            batch_states.append(to_tensor(obs))
            batch_actions.append(action)
            batch_rewards.append(T.tensor([[rew]], dtype=T.float32))
            batch_terminals.append(done)

            reward_acc += rew
            obs = nobs

        ep_in_batch += 1

        if ep_in_batch >= args.batchsize:
            states = T.cat(batch_states, dim=0)
            actions = T.cat(batch_actions, dim=0)
            rewards = T.cat(batch_rewards, dim=0)

            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
            adv = calc_returns(args.gamma, rewards, batch_terminals)
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)

            update_ppo(policy, optim, states, actions, adv, update_iters=args.ppo_update_iters, clip_eps=args.clip_eps)

            print(
                f"iter={it}/{args.iters} batch_ep_rew={reward_acc / max(ep_in_batch, 1):.3f} "
                f"obs_dim={env.obs_dim} act_dim={env.act_dim}"
            )

            batch_states.clear()
            batch_actions.clear()
            batch_rewards.clear()
            batch_terminals.clear()
            ep_in_batch = 0
            reward_acc = 0.0

        if it % args.save_every == 0:
            T.save(policy.state_dict(), str(ckpt))
            score = evaluate(env, policy, episodes=2, render=False)
            print(f"checkpoint={ckpt} eval_avg_rew={score:.3f}")

    T.save(policy.state_dict(), str(ckpt))
    print(f"Saved final checkpoint: {ckpt}")
    env.close()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--hid-dim", type=int, default=128)
    parser.add_argument("--ppo-update-iters", type=int, default=6)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--target-vel", type=float, default=0.2)
    parser.add_argument("--episode-steps", type=int, default=800)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/hexapod_pg_pybullet.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--gui", action="store_true", default=True)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
