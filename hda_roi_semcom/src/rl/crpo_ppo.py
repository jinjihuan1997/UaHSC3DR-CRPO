"""PPO-compatible constrained policy optimization for resource allocation."""
import csv
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.rl.resource_model import select_constraint_mode


def set_seed(seed):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class TwoHeadActorCritic(nn.Module):
    def __init__(self, obs_dim, n_kd, n_beta, hidden_dim=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.kd_head = nn.Linear(hidden_dim, n_kd)
        self.beta_head = nn.Linear(hidden_dim, n_beta)
        self.v_reward = nn.Linear(hidden_dim, 1)
        self.v_cost_rgb = nn.Linear(hidden_dim, 1)
        self.v_cost_depth = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        z = self.backbone(obs)
        return {
            "kd_logits": self.kd_head(z),
            "beta_logits": self.beta_head(z),
            "v_reward": self.v_reward(z).squeeze(-1),
            "v_cost_rgb": self.v_cost_rgb(z).squeeze(-1),
            "v_cost_depth": self.v_cost_depth(z).squeeze(-1),
        }

    def act(self, obs, deterministic=False):
        out = self.forward(obs)
        kd_dist = Categorical(logits=out["kd_logits"])
        beta_dist = Categorical(logits=out["beta_logits"])
        if deterministic:
            kd = torch.argmax(out["kd_logits"], dim=-1)
            beta = torch.argmax(out["beta_logits"], dim=-1)
        else:
            kd = kd_dist.sample()
            beta = beta_dist.sample()
        log_prob = kd_dist.log_prob(kd) + beta_dist.log_prob(beta)
        entropy = kd_dist.entropy() + beta_dist.entropy()
        action = torch.stack([kd, beta], dim=-1)
        return action, log_prob, entropy, out

    def evaluate_actions(self, obs, actions):
        out = self.forward(obs)
        kd_dist = Categorical(logits=out["kd_logits"])
        beta_dist = Categorical(logits=out["beta_logits"])
        kd = actions[:, 0]
        beta = actions[:, 1]
        log_prob = kd_dist.log_prob(kd) + beta_dist.log_prob(beta)
        entropy = kd_dist.entropy() + beta_dist.entropy()
        return log_prob, entropy, out


@dataclass
class PPOConfig:
    total_timesteps: int = 100_000
    n_steps: int = 2048
    update_epochs: int = 10
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    learning_rate: float = 3e-4
    ent_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    epsilon_rgb: float = 0.05
    epsilon_depth: float = 0.05
    hidden_dim: int = 128
    objective: str = "crpo"
    penalty_rgb: float = 1.0
    penalty_depth: float = 1.0
    lr_dual: float = 0.005

    @property
    def d_rgb(self):
        return self.epsilon_rgb

    @property
    def d_depth(self):
        return self.epsilon_depth


class RolloutBuffer:
    def __init__(self, n_steps, obs_dim, device):
        self.n_steps = int(n_steps)
        self.obs_dim = int(obs_dim)
        self.device = device
        self.reset()

    def reset(self):
        self.obs = np.zeros((self.n_steps, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, 2), dtype=np.int64)
        self.log_probs = np.zeros(self.n_steps, dtype=np.float32)
        self.rewards = np.zeros(self.n_steps, dtype=np.float32)
        self.cost_rgb = np.zeros(self.n_steps, dtype=np.float32)
        self.cost_depth = np.zeros(self.n_steps, dtype=np.float32)
        self.values_reward = np.zeros(self.n_steps, dtype=np.float32)
        self.values_cost_rgb = np.zeros(self.n_steps, dtype=np.float32)
        self.values_cost_depth = np.zeros(self.n_steps, dtype=np.float32)
        self.dones = np.zeros(self.n_steps, dtype=np.float32)
        self.q_rgb = np.zeros(self.n_steps, dtype=np.float32)
        self.r_depth = np.zeros(self.n_steps, dtype=np.float32)
        self.q_3d = np.zeros(self.n_steps, dtype=np.float32)
        self.reconstruction_gain = np.zeros(self.n_steps, dtype=np.float32)
        self.snr_rgb_db = np.zeros(self.n_steps, dtype=np.float32)
        self.snr_depth_db = np.zeros(self.n_steps, dtype=np.float32)
        self.k_d = np.zeros(self.n_steps, dtype=np.float32)
        self.k_rgb = np.zeros(self.n_steps, dtype=np.float32)
        self.beta_d = np.zeros(self.n_steps, dtype=np.float32)
        self.depth_payload_bits = np.zeros(self.n_steps, dtype=np.float32)
        self.depth_bit_budget = np.zeros(self.n_steps, dtype=np.float32)
        self.map_progress = np.zeros(self.n_steps, dtype=np.float32)
        self.view_importance = np.zeros(self.n_steps, dtype=np.float32)
        self.pos = 0

    def add(self, obs, action, log_prob, reward, done, values, info):
        i = self.pos
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.cost_rgb[i] = info["cost_rgb"]
        self.cost_depth[i] = info["cost_depth"]
        self.values_reward[i] = values["v_reward"]
        self.values_cost_rgb[i] = values["v_cost_rgb"]
        self.values_cost_depth[i] = values["v_cost_depth"]
        self.dones[i] = float(done)
        self.q_rgb[i] = info["q_rgb"]
        self.r_depth[i] = info["r_depth"]
        self.q_3d[i] = info["q_3d"]
        self.reconstruction_gain[i] = info["reconstruction_gain"]
        self.snr_rgb_db[i] = info["snr_rgb_db"]
        self.snr_depth_db[i] = info["snr_depth_db"]
        self.k_d[i] = info["k_d"]
        self.k_rgb[i] = info["k_rgb"]
        self.beta_d[i] = info["beta_d"]
        self.depth_payload_bits[i] = info["depth_payload_bits"]
        self.depth_bit_budget[i] = info["depth_bit_budget"]
        self.map_progress[i] = info["map_progress"]
        self.view_importance[i] = info["view_importance"]
        self.pos += 1

    def tensors(self):
        return {
            "obs": torch.as_tensor(self.obs, dtype=torch.float32, device=self.device),
            "actions": torch.as_tensor(self.actions, dtype=torch.long, device=self.device),
            "log_probs": torch.as_tensor(self.log_probs, dtype=torch.float32, device=self.device),
        }


def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_nonterminal = 1.0 - dones[t]
            next_value = float(last_value)
        else:
            next_nonterminal = 1.0 - dones[t]
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def ensure_csv(path, fieldnames):
    exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not exists:
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def append_csv(path, fieldnames, row):
    ensure_csv(path, fieldnames)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def train_crpo_ppo(env, cfg, *, seed=42, device="cpu", log_csv=None, save_path=None):
    set_seed(seed)
    obs_dim = int(env.observation_space.shape[0])
    policy = TwoHeadActorCritic(
        obs_dim,
        len(env.k_d_choices),
        len(env.beta_d_choices),
        hidden_dim=cfg.hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    buffer = RolloutBuffer(cfg.n_steps, obs_dim, device)

    obs, _ = env.reset(seed=seed)
    global_step = 0
    iteration = 0
    log_fields = [
        "iteration", "global_step", "avg_episode_reward", "avg_map_reward",
        "J_C_R", "J_C_D", "epsilon_R", "epsilon_D",
        "excess_R", "excess_D", "relative_excess_R", "relative_excess_D",
        "rgb_violation_rate", "depth_violation_rate",
        "avg_cost_rgb", "avg_cost_depth", "constraint_mode", "selected_constraint",
        "avg_penalized_reward",
        "avg_q_3d", "avg_Q_rgb", "avg_R_depth", "avg_reconstruction_gain",
        "avg_snr_rgb_db", "avg_snr_depth_db", "avg_k_d", "avg_k_rgb",
        "avg_beta_d", "avg_depth_payload_bits", "avg_depth_bit_budget", "avg_map_progress",
        "policy_entropy", "value_loss_reward", "value_loss_cost_rgb",
        "value_loss_cost_depth", "actor_loss",
        "lambda_rgb_cur", "lambda_depth_cur",
    ]
    # Truncate any existing log so a rerun never appends to stale data
    if log_csv is not None:
        os.makedirs(os.path.dirname(log_csv) or ".", exist_ok=True)
        with open(log_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writeheader()

    episode_returns = []
    current_episode_return = 0.0
    _lambda_rgb = 0.0
    _lambda_depth = 0.0

    while global_step < cfg.total_timesteps:
        buffer.reset()
        for _ in range(cfg.n_steps):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_tensor, log_prob, _, out = policy.act(obs_tensor, deterministic=False)
            action = action_tensor.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            values = {
                "v_reward": float(out["v_reward"].item()),
                "v_cost_rgb": float(out["v_cost_rgb"].item()),
                "v_cost_depth": float(out["v_cost_depth"].item()),
            }
            buffer.add(obs, action, float(log_prob.item()), float(reward), done, values, info)
            global_step += 1
            current_episode_return += float(reward)
            obs = next_obs
            if done:
                episode_returns.append(current_episode_return)
                current_episode_return = 0.0
                obs, _ = env.reset()
            if global_step >= cfg.total_timesteps:
                break

        with torch.no_grad():
            last_obs = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            last_out = policy.forward(last_obs)
            last_v_reward = float(last_out["v_reward"].item())
            last_v_cost_rgb = float(last_out["v_cost_rgb"].item())
            last_v_cost_depth = float(last_out["v_cost_depth"].item())

        adv_reward, ret_reward = compute_gae(
            buffer.rewards[:buffer.pos],
            buffer.values_reward[:buffer.pos],
            buffer.dones[:buffer.pos],
            last_v_reward,
            cfg.gamma,
            cfg.gae_lambda,
        )
        adv_cost_rgb, ret_cost_rgb = compute_gae(
            buffer.cost_rgb[:buffer.pos],
            buffer.values_cost_rgb[:buffer.pos],
            buffer.dones[:buffer.pos],
            last_v_cost_rgb,
            cfg.gamma,
            cfg.gae_lambda,
        )
        adv_cost_depth, ret_cost_depth = compute_gae(
            buffer.cost_depth[:buffer.pos],
            buffer.values_cost_depth[:buffer.pos],
            buffer.dones[:buffer.pos],
            last_v_cost_depth,
            cfg.gamma,
            cfg.gae_lambda,
        )
        adv_neg_rgb, _ = compute_gae(
            -buffer.cost_rgb[:buffer.pos],
            -buffer.values_cost_rgb[:buffer.pos],
            buffer.dones[:buffer.pos],
            -last_v_cost_rgb,
            cfg.gamma,
            cfg.gae_lambda,
        )
        adv_neg_depth, _ = compute_gae(
            -buffer.cost_depth[:buffer.pos],
            -buffer.values_cost_depth[:buffer.pos],
            buffer.dones[:buffer.pos],
            -last_v_cost_depth,
            cfg.gamma,
            cfg.gae_lambda,
        )
        j_c_r = float(np.mean(buffer.cost_rgb[:buffer.pos]))
        j_c_d = float(np.mean(buffer.cost_depth[:buffer.pos]))

        # Lagrangian dual update: λ ← max(0, λ + lr_dual·(J_C - ε))
        if cfg.objective == "lagrangian":
            _lambda_rgb   = max(0.0, _lambda_rgb   + cfg.lr_dual * (j_c_r - cfg.epsilon_rgb))
            _lambda_depth = max(0.0, _lambda_depth + cfg.lr_dual * (j_c_d - cfg.epsilon_depth))
            lam_r, lam_d = _lambda_rgb, _lambda_depth
        else:
            lam_r, lam_d = float(cfg.penalty_rgb), float(cfg.penalty_depth)

        penalized_rewards = (
            buffer.rewards[:buffer.pos]
            - lam_r * buffer.cost_rgb[:buffer.pos]
            - lam_d * buffer.cost_depth[:buffer.pos]
        )
        adv_penalty, ret_penalty = compute_gae(
            penalized_rewards,
            buffer.values_reward[:buffer.pos],
            buffer.dones[:buffer.pos],
            last_v_reward,
            cfg.gamma,
            cfg.gae_lambda,
        )

        excess_r = max(j_c_r - float(cfg.epsilon_rgb), 0.0)
        excess_d = max(j_c_d - float(cfg.epsilon_depth), 0.0)
        relative_excess_r = excess_r / (float(cfg.epsilon_rgb) + 1e-8)
        relative_excess_d = excess_d / (float(cfg.epsilon_depth) + 1e-8)
        if cfg.objective in ("penalty", "lagrangian"):
            mode, selected = cfg.objective, ""
            actor_adv = adv_penalty
            ret_actor_value = ret_penalty
        else:
            mode, selected = select_constraint_mode(j_c_r, j_c_d, cfg.epsilon_rgb, cfg.epsilon_depth)
            ret_actor_value = ret_reward
        if mode == "constraint_rgb":
            actor_adv = adv_neg_rgb
        elif mode == "constraint_depth":
            actor_adv = adv_neg_depth
        elif mode == "reward":
            actor_adv = adv_reward

        data = buffer.tensors()
        n = buffer.pos
        old_log_probs = data["log_probs"][:n]
        obs_batch = data["obs"][:n]
        action_batch = data["actions"][:n]
        actor_adv_t = torch.as_tensor(actor_adv, dtype=torch.float32, device=device)
        actor_adv_t = (actor_adv_t - actor_adv_t.mean()) / (actor_adv_t.std(unbiased=False) + 1e-8)
        ret_reward_t = torch.as_tensor(ret_actor_value, dtype=torch.float32, device=device)
        ret_cost_rgb_t = torch.as_tensor(ret_cost_rgb, dtype=torch.float32, device=device)
        ret_cost_depth_t = torch.as_tensor(ret_cost_depth, dtype=torch.float32, device=device)

        actor_losses = []
        entropies = []
        value_losses_reward = []
        value_losses_cost_rgb = []
        value_losses_cost_depth = []
        indices = np.arange(n)
        for _ in range(cfg.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, cfg.batch_size):
                mb_idx = indices[start:start + cfg.batch_size]
                mb_idx_t = torch.as_tensor(mb_idx, dtype=torch.long, device=device)
                log_prob, entropy, out = policy.evaluate_actions(
                    obs_batch.index_select(0, mb_idx_t),
                    action_batch.index_select(0, mb_idx_t),
                )
                ratio = torch.exp(log_prob - old_log_probs.index_select(0, mb_idx_t))
                mb_adv = actor_adv_t.index_select(0, mb_idx_t)
                pg1 = ratio * mb_adv
                pg2 = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * mb_adv
                actor_loss = -torch.min(pg1, pg2).mean()
                v_loss_reward = 0.5 * (out["v_reward"] - ret_reward_t.index_select(0, mb_idx_t)).pow(2).mean()
                v_loss_rgb = 0.5 * (out["v_cost_rgb"] - ret_cost_rgb_t.index_select(0, mb_idx_t)).pow(2).mean()
                v_loss_depth = 0.5 * (out["v_cost_depth"] - ret_cost_depth_t.index_select(0, mb_idx_t)).pow(2).mean()
                entropy_loss = entropy.mean()
                loss = (
                    actor_loss
                    + cfg.value_coef * (v_loss_reward + v_loss_rgb + v_loss_depth)
                    - cfg.ent_coef * entropy_loss
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optimizer.step()

                actor_losses.append(float(actor_loss.detach().cpu().item()))
                entropies.append(float(entropy_loss.detach().cpu().item()))
                value_losses_reward.append(float(v_loss_reward.detach().cpu().item()))
                value_losses_cost_rgb.append(float(v_loss_rgb.detach().cpu().item()))
                value_losses_cost_depth.append(float(v_loss_depth.detach().cpu().item()))

        iteration += 1
        recent_returns = episode_returns[-20:] if episode_returns else [current_episode_return]
        log_row = {
            "iteration": iteration,
            "global_step": global_step,
            "avg_episode_reward": float(np.mean(recent_returns)),
            "avg_map_reward": float(np.mean(buffer.rewards[:n])),
            "J_C_R": j_c_r,
            "J_C_D": j_c_d,
            "epsilon_R": float(cfg.epsilon_rgb),
            "epsilon_D": float(cfg.epsilon_depth),
            "excess_R": excess_r,
            "excess_D": excess_d,
            "relative_excess_R": relative_excess_r,
            "relative_excess_D": relative_excess_d,
            "rgb_violation_rate": float(np.mean(buffer.cost_rgb[:n] > 1e-8)),
            "depth_violation_rate": float(np.mean(buffer.cost_depth[:n] > 1e-8)),
            "avg_cost_rgb": j_c_r,
            "avg_cost_depth": j_c_d,
            "constraint_mode": mode,
            "selected_constraint": selected or "",
            "avg_penalized_reward": float(np.mean(penalized_rewards)),
            "avg_q_3d": float(np.mean(buffer.q_3d[:n])),
            "avg_Q_rgb": float(np.mean(buffer.q_rgb[:n])),
            "avg_R_depth": float(np.mean(buffer.r_depth[:n])),
            "avg_reconstruction_gain": float(np.mean(buffer.reconstruction_gain[:n])),
            "avg_snr_rgb_db": float(np.mean(buffer.snr_rgb_db[:n])),
            "avg_snr_depth_db": float(np.mean(buffer.snr_depth_db[:n])),
            "avg_k_d": float(np.mean(buffer.k_d[:n])),
            "avg_k_rgb": float(np.mean(buffer.k_rgb[:n])),
            "avg_beta_d": float(np.mean(buffer.beta_d[:n])),
            "avg_depth_payload_bits": float(np.mean(buffer.depth_payload_bits[:n])),
            "avg_depth_bit_budget": float(np.mean(buffer.depth_bit_budget[:n])),
            "avg_map_progress": float(np.mean(buffer.map_progress[:n])),
            "policy_entropy": float(np.mean(entropies)),
            "value_loss_reward": float(np.mean(value_losses_reward)),
            "value_loss_cost_rgb": float(np.mean(value_losses_cost_rgb)),
            "value_loss_cost_depth": float(np.mean(value_losses_cost_depth)),
            "actor_loss": float(np.mean(actor_losses)),
            "lambda_rgb_cur": _lambda_rgb,
            "lambda_depth_cur": _lambda_depth,
        }
        if log_csv:
            append_csv(log_csv, log_fields, log_row)
        print(
            f"[iter {iteration}] step={global_step} mode={mode} "
            f"r={log_row['avg_map_reward']:.4f} J_C_R={j_c_r:.4f} "
            f"J_C_D={j_c_d:.4f} q3d={log_row['avg_q_3d']:.3f} "
            f"Q={log_row['avg_Q_rgb']:.3f} R={log_row['avg_R_depth']:.3f}",
            flush=True,
        )

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save({
            "model_state": policy.state_dict(),
            "obs_dim": obs_dim,
            "n_kd": len(env.k_d_choices),
            "n_beta": len(env.beta_d_choices),
            "k_d_choices": env.k_d_choices,
            "beta_d_choices": env.beta_d_choices,
            "k_total": env.k_total,
            "q_req": env.q_req,
            "depth_req": env.depth_req,
            "episode_len": env.episode_len,
            "hidden_dim": cfg.hidden_dim,
            "objective": cfg.objective,
            "epsilon_rgb": cfg.epsilon_rgb,
            "epsilon_depth": cfg.epsilon_depth,
            "penalty_rgb": cfg.penalty_rgb,
            "penalty_depth": cfg.penalty_depth,
            "lr_dual": cfg.lr_dual,
        }, save_path)
        print(f"[saved] {save_path}")
    return policy


def load_policy(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    policy = TwoHeadActorCritic(
        ckpt["obs_dim"],
        ckpt["n_kd"],
        ckpt["n_beta"],
        hidden_dim=ckpt.get("hidden_dim", 128),
    ).to(device)
    policy.load_state_dict(ckpt["model_state"])
    policy.eval()
    return policy, ckpt
