"""ReinFlow (Zhang et al. 2025): PPO fine-tuning of a 1-pass flow / shortcut policy on PushT.

    python scripts/reinflow.py outputs/train/shortcut_s1000_40000 --out outputs/train/reinflow_s1000

The Euler step gets Gaussian noise, x1 = x0 + v(x0) + sigma * eps, so each action chunk has a tractable
log-probability. The vision encoder is frozen: global_cond is computed during the rollout and stored, so
PPO updates never touch images. Reward is the change in coverage plus 1 on success (the raw per-step
coverage reward pays more for hovering at 94% than for finishing). Training seeds start at 0 and the
held-out monitor uses 200000+, both disjoint from the 100000+ eval seeds. The saved checkpoint is a plain
FlowPolicy: evaluate it with eval_sweep.py / lerobot-eval, where the sampler is noise-free again.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import gym_pusht  # noqa: F401  registers PushT before the async env workers start
import numpy as np
import torch
from torch import nn

import lerobot_policy_flowpolicy  # noqa: F401  registers flow
from lerobot.envs.configs import PushtEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import NEW_ROLLOUT_OPTION, preprocess_observation
from lerobot.policies import make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot_policy_flowpolicy.flow.modeling_flow import FlowPolicy

CKPT = "checkpoints/last/pretrained_model"


class Noisy:
    """Replaces FlowModel.conditional_sample during rollouts: one noisy Euler step, recorded for PPO."""

    def __init__(self, m, log_std):
        self.m, self.log_std, self.rec, self.noise = m, log_std, [], True

    def mean(self, x0, cond):
        one = torch.ones(x0.shape[0], device=x0.device)
        return x0 + self.m._unet(x0, 0 * one, cond, d=one)

    def __call__(self, batch_size, global_cond=None, generator=None, noise=None):
        cfg = self.m.config
        x0 = torch.randn(batch_size, cfg.horizon, cfg.action_feature.shape[0], device=global_cond.device)
        x1 = self.mean(x0, global_cond)
        if self.noise:
            x1 = x1 + self.log_std.exp() * torch.randn_like(x1)
        self.rec.append((global_cond, x0, x1))
        return x1.clamp(-1, 1)  # only the executed copy is clamped; log-probs use the raw x1


def logp(noisy, cond, x0, x1, sl):
    mu = noisy.mean(x0, cond)
    return torch.distributions.Normal(mu[:, sl], noisy.log_std.exp()).log_prob(x1[:, sl]).sum((1, 2))


def rollout(env, policy, noisy, procs, seeds, device):
    """One batched episode per env. Returns per-chunk tensors (C, N, ...) plus per-episode success."""
    pre, post, env_pre, env_post = procs
    n = len(seeds)
    policy.reset()
    noisy.rec = []
    obs, _ = env.reset(seed=seeds, options={NEW_ROLLOUT_OPTION: True})
    k = policy.config.n_action_steps
    done, succ = np.zeros(n, bool), np.zeros(n, bool)
    prev_phi, rew, valid, dones = None, [], [], []
    for step in range(env.call("_max_episode_steps")[0]):
        if step % k == 0:
            rew.append(np.zeros(n)), valid.append(~done), dones.append(done.copy())
        o = preprocess_observation(obs)
        o["task"] = [""] * n
        with torch.no_grad():
            a = policy.select_action(pre(env_pre(o)))
        a = env_post({ACTION: post(a)})[ACTION]
        obs, r, term, trunc, info = env.step(a.cpu().numpy())
        phi = np.asarray(r, float)  # clip(coverage / 0.95): its change is a potential-based shaping reward
        live = ~done
        rew[-1] += live * ((phi - prev_phi if prev_phi is not None else 0) + term)
        succ |= live & term
        done |= np.asarray(term) | np.asarray(trunc)
        dones[-1] = done.copy()
        prev_phi = phi
        if done.all():
            break
    cond, x0, x1 = (torch.stack(z) for z in zip(*noisy.rec))
    return dict(
        cond=cond, x0=x0, x1=x1,
        rew=torch.tensor(np.array(rew), dtype=torch.float32, device=device),
        valid=torch.tensor(np.array(valid), device=device),
        done=torch.tensor(np.array(dones), dtype=torch.float32, device=device),
    ), succ  # fmt: skip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path, help="base flow/shortcut training dir")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--envs", type=int, default=24)
    p.add_argument("--rollouts", type=int, default=2, help="batched episodes per iteration")
    p.add_argument("--actor-lr", type=float, default=1e-5)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--std", type=float, default=0.05, help="initial noise std (normalised actions)")
    p.add_argument("--std-min", type=float, default=0.01)
    p.add_argument("--std-max", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--minibatch", type=int, default=128)
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--critic-warmup", type=int, default=5)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-episodes", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--sync", action="store_true", help="SyncVectorEnv (async env workers fail on macOS)")
    a = p.parse_args()
    torch.manual_seed(a.seed)
    # TF32 convs make the same chunk's log-prob differ by ~1e-2 across batch sizes, which PPO reads as a policy change.
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    dev = a.device

    base = a.run / CKPT
    policy = FlowPolicy.from_pretrained(base).to(dev).eval()  # eval mode: no random crop, GroupNorm only
    m = policy.diffusion
    m.rgb_encoder.requires_grad_(False)
    actor = [q for q in m.parameters() if q.requires_grad]
    log_std = nn.Parameter(torch.full((m.config.action_feature.shape[0],), float(np.log(a.std)), device=dev))
    noisy = Noisy(m, log_std)
    m.conditional_sample = noisy
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=str(base), preprocessor_overrides={"device_processor": {"device": dev}}
    )
    env_cfg = PushtEnv()
    env = next(iter(make_env(env_cfg, n_envs=a.envs, use_async_envs=not a.sync)["pusht"].values()))
    procs = (pre, post, *make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config))
    sl = slice(policy.config.n_obs_steps - 1, policy.config.n_obs_steps - 1 + policy.config.n_action_steps)

    critic = None
    opt_a = torch.optim.Adam([{"params": actor}, {"params": [log_std], "lr": 1e-3}], lr=a.actor_lr)
    a.out.mkdir(parents=True, exist_ok=True)
    log = open(a.out / "log.jsonl", "a")
    best, next_seed, t_start = 0.0, a.seed * 1_000_000, time.time()

    def evaluate(it):
        noisy.noise, s = False, []
        for i in range(0, a.eval_episodes, a.envs):
            s.append(rollout(env, policy, noisy, procs, list(range(200_000 + i, 200_000 + i + a.envs)), dev)[1])
        noisy.noise = True
        return 100 * np.concatenate(s)[: a.eval_episodes].mean()

    for it in range(a.iters + 1):
        if it % a.eval_every == 0:
            sr = evaluate(it)
            print(json.dumps({"iter": it, "heldout_success": sr, "min": (time.time() - t_start) / 60}), flush=True)
            log.write(json.dumps({"iter": it, "heldout_success": sr}) + "\n"), log.flush()
            if it == 0:
                best = sr  # the base policy: only save fine-tuned weights that beat it on held-out seeds
            elif sr >= best:
                best = sr
                shutil.copytree(base, a.out / CKPT, dirs_exist_ok=True)  # processors, train config
                policy.save_pretrained(a.out / CKPT)
        if it == a.iters:
            break

        # Collect.
        bufs, succ = [], []
        for _ in range(a.rollouts):
            b, s = rollout(env, policy, noisy, procs, list(range(next_seed, next_seed + a.envs)), dev)
            next_seed += a.envs
            bufs.append(b), succ.append(s)
        C = min(b["rew"].shape[0] for b in bufs)
        cat = lambda key: torch.cat([b[key][:C] for b in bufs], 1)  # noqa: E731  (C, N, ...)
        cond, x0, x1, rew, valid, done = (cat(k) for k in ("cond", "x0", "x1", "rew", "valid", "done"))
        tfeat = (torch.arange(C, device=dev, dtype=torch.float32) / C)[:, None, None].expand(C, cond.shape[1], 1)
        obs_v = torch.cat([cond, tfeat], -1)
        if critic is None:
            critic = nn.Sequential(
                nn.Linear(obs_v.shape[-1], 256), nn.Mish(), nn.Linear(256, 256), nn.Mish(), nn.Linear(256, 1)
            ).to(dev)
            opt_c = torch.optim.Adam(critic.parameters(), lr=a.critic_lr)

        # GAE over chunks; a chunk after the env is done is invalid, and the last chunk has no bootstrap.
        with torch.no_grad():
            v = critic(obs_v).squeeze(-1)
            adv, last = torch.zeros_like(rew), torch.zeros_like(rew[0])
            for t in reversed(range(C)):
                nv = v[t + 1] if t + 1 < C else torch.zeros_like(v[t])
                nonterm = 1 - done[t]
                delta = rew[t] + a.gamma * nv * nonterm - v[t]
                last = delta + a.gamma * a.lam * nonterm * last
                adv[t] = last
            ret = adv + v
            idx = valid.flatten().nonzero().squeeze(1)
            flat = lambda z: z.flatten(0, 1)[idx]  # noqa: E731
            cond, x0, x1, obs_v, adv, ret = map(flat, (cond, x0, x1, obs_v, adv, ret))
            old = torch.cat([logp(noisy, cond[j : j + 256], x0[j : j + 256], x1[j : j + 256], sl) for j in range(0, len(idx), 256)])
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # PPO.
        stats, update_actor = {"kl": 0.0, "clipfrac": 0.0, "actor_steps": 0}, it >= a.critic_warmup
        for ep in range(a.epochs):
            perm = torch.randperm(len(idx), device=dev)
            for j in range(0, len(idx), a.minibatch):
                mb = perm[j : j + a.minibatch]
                vloss = (critic(obs_v[mb]).squeeze(-1) - ret[mb]).square().mean()
                opt_c.zero_grad(), vloss.backward(), opt_c.step()
                if not update_actor:
                    continue
                new = logp(noisy, cond[mb], x0[mb], x1[mb], sl)
                ratio = (new - old[mb]).exp()
                if ep == 0 and j == 0:
                    assert (ratio - 1).abs().max() < 1e-3, f"ratio {ratio.min()}..{ratio.max()} at the first step"
                loss = -torch.min(ratio * adv[mb], ratio.clamp(1 - a.clip, 1 + a.clip) * adv[mb]).mean()
                opt_a.zero_grad(), loss.backward()
                nn.utils.clip_grad_norm_(actor, 1.0)
                opt_a.step()
                with torch.no_grad():
                    log_std.clamp_(np.log(a.std_min), np.log(a.std_max))
                    stats["kl"] = ((ratio - 1) - (new - old[mb])).mean().item()  # KL before this step
                    stats["clipfrac"] = ((ratio - 1).abs() > a.clip).float().mean().item()
                    stats["actor_steps"] += 1
                # Stop moving the actor once it has drifted target_kl from the rollout policy; the critic keeps fitting.
                update_actor = update_actor and stats["kl"] <= a.target_kl
        row = {
            "iter": it, "train_success": 100 * np.concatenate(succ).mean(), "return": rew.sum(0).mean().item(),
            "vloss": vloss.item(), "std": log_std.exp().tolist(), "samples": len(idx),
            "min": round((time.time() - t_start) / 60, 1), **stats,
        }  # fmt: skip
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + "\n"), log.flush()
    env.close()


if __name__ == "__main__":
    main()
