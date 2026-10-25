"""Evaluate trained PushT checkpoints: success vs. sampling steps, plus inference latency per action chunk.

    python scripts/eval_sweep.py outputs/train/*_40000 --out results/fast

Eval seeds are fixed (start at --seed), so every policy and step count sees the same initial states.
Finished variants (eval_info.json exists) are skipped, so the sweep can be killed and rerun.
"""

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

import lerobot_policy_flowpolicy  # noqa: F401  registers flow/bcmlp
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.utils.constants import OBS_STATE

STEPS = [1, 2, 4, 10, 100]
# Flow sampler variants: early small steps (mode decision) and warm-starting from the previous plan.
HYBRID = [
    {"step_schedule": "[0,0.1,1]"},
    {"step_schedule": "[0,0.05,0.15,1]"},
    {"num_inference_steps": 1, "warm_start": 0.3},
    {"num_inference_steps": 1, "warm_start": 0.6},
    {"step_schedule": "[0,0.1,1]", "warm_start": 0.3},
]


# Diffusion decides, flow executes. {run} becomes the flow run's name with flow -> diffusion (same seed).
GUIDE = "outputs/train/{run}/checkpoints/last/pretrained_model"
GUIDED = [
    {"guide_path": GUIDE, "guide_every": 0, "num_inference_steps": 1, "warm_start": 0.5},
    {"guide_path": GUIDE, "guide_every": 4, "num_inference_steps": 1, "warm_start": 0.5},
    {"guide_path": GUIDE, "guide_every": 2, "num_inference_steps": 1, "warm_start": 0.5},
]


# Best single fixes, for stacking on a distilled (reflow) student.
COMBO = [
    {"num_inference_steps": 1, "warm_start": 0.3},
    {"step_schedule": "[0,0.1,1]"},
    {"step_schedule": "[0,0.1,1]", "warm_start": 0.3},
    GUIDED[0],
    GUIDED[1],
]


# Stack the cheap inference-time knobs on the best few-step models (shortcut, reflow): replan more often
# (n_action_steps 4 instead of 8) or less often (12), lower noise temperature, warm start, diffusion guide.
EXPLORE = [
    {"num_inference_steps": 1, "n_action_steps": 4},
    {"num_inference_steps": 1, "n_action_steps": 4, "warm_start": 0.3},
    {"num_inference_steps": 1, "noise_scale": 0.5},
    {"num_inference_steps": 1, "noise_scale": 0.8},
    {"num_inference_steps": 1, "n_action_steps": 4, "noise_scale": 0.5},
    {"num_inference_steps": 2, "n_action_steps": 4},
    # Replanning more often hurt (mode switching between chunks), so also try committing longer.
    {"num_inference_steps": 1, "n_action_steps": 12},
    {"num_inference_steps": 2, "n_action_steps": 12},
    {"num_inference_steps": 1, "n_action_steps": 12, "noise_scale": 0.8},
    GUIDED[1],
]


def variants(policy_type: str, sets: list[str]) -> list[dict]:
    """Config overrides to evaluate for each policy type. `sets` picks HYBRID/GUIDED/COMBO instead of the default."""
    if sets:
        return [v for name in sets for v in globals()[name.upper()]] if policy_type == "flow" else []
    if policy_type == "flow":
        return [{"num_inference_steps": k} for k in STEPS]
    if policy_type == "diffusion":
        # DDPM is how DP was trained and evaluated; DDIM is its standard few-step sampler, so the
        # low-step comparison against flow isn't against a strawman.
        return [{"noise_scheduler_type": s, "num_inference_steps": k} for s in ("DDPM", "DDIM") for k in STEPS]
    return [{}]


def name(over: dict) -> str:
    if "guide_path" in over:
        return f"guide{over['guide_every']}_{over['num_inference_steps']}_w{over['warm_start']}"
    prefix = {"warm_start": "w", "n_action_steps": "a", "noise_scale": "n"}
    return "_".join(f"{prefix.get(k, '')}{v}" for k, v in over.items()) or "default"


@torch.no_grad()
def latency_ms(ckpt: Path, over: dict, device: str, iters: int = 50) -> float:
    """Median wall time of one predict_action_chunk call, batch size 1 (one control-loop query)."""
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    for k, v in over.items():
        setattr(cfg, k, json.loads(v) if isinstance(v, str) and v.startswith("[") else v)
    cfg.device = device
    policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg).to(device).eval()
    # Diffusion-family policies take an observation history; ACT takes only the current frame.
    hist = (getattr(cfg, "n_obs_steps", 1),) if hasattr(policy, "diffusion") else ()
    batch = {k: torch.randn(1, *hist, *f.shape, device=device) for k, f in cfg.input_features.items()}
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    times = []
    for i in range(iters + 10):
        policy.reset()
        sync()
        t0 = time.perf_counter()
        policy.predict_action_chunk(batch)
        sync()
        if i >= 10:  # warmup
            times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def run_eval(ckpt: Path, over: dict, out: Path, a: argparse.Namespace) -> dict:
    if not (out / "eval_info.json").exists():
        cmd = [
            "lerobot-eval", f"--policy.path={ckpt}", f"--policy.device={a.device}", "--env.type=pusht",
            f"--eval.n_episodes={a.episodes}", f"--eval.batch_size={a.batch}", "--eval.use_async_envs=false",
            f"--seed={a.seed}", f"--output_dir={out}",
            *[f"--policy.{k}={v}" for k, v in over.items()],
        ]  # fmt: skip
        subprocess.run(cmd, check=True)
    return json.loads((out / "eval_info.json").read_text())["overall"]


def random_baseline(a: argparse.Namespace) -> dict:
    """Lower bound: uniform random actions on the same eval seeds."""
    import gymnasium as gym
    import gym_pusht  # noqa: F401

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=300)
    env.action_space.seed(a.seed)
    succ, cov = [], []
    for i in range(a.episodes):
        env.reset(seed=a.seed + i)
        best, done, info = 0.0, False, {}
        while not done:
            _, r, term, trunc, info = env.step(env.action_space.sample())
            best, done = max(best, r), term or trunc
        succ.append(info.get("is_success", False))
        cov.append(best)
    return {"pc_success": 100 * sum(succ) / len(succ), "avg_max_reward": sum(cov) / len(cov)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=Path, help="training output dirs")
    p.add_argument("--out", type=Path, default=Path("results/fast"))
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--seed", type=int, default=100_000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--hybrid", action="store_true", help="only the flow HYBRID sampler variants")
    p.add_argument("--guided", action="store_true", help="only the diffusion-guided flow variants (GUIDED)")
    p.add_argument("--combo", action="store_true", help="only the COMBO variants (fixes stacked on reflow)")
    p.add_argument("--explore", action="store_true", help="only the EXPLORE variants (knobs stacked on shortcut/reflow)")
    p.add_argument("--guide-run", default=None, help="guide run name, {seed} filled in (default: flow -> diffusion)")
    a = p.parse_args()

    rows = []
    r = random_baseline(a)
    rows.append({"run": "random", "policy": "random", "variant": "-", **r, "latency_ms": 0.0})
    for run in a.runs:
        ckpt = run / "checkpoints/last/pretrained_model"
        ptype = json.loads((ckpt / "config.json").read_text())["type"]
        sets = [n for n in ("hybrid", "guided", "combo", "explore") if getattr(a, n)]
        for over in variants(ptype, sets):
            if "guide_path" in over:
                seed = run.name.split("_s")[1].split("_")[0]
                guide = a.guide_run.format(seed=seed) if a.guide_run else run.name.replace("flow", "diffusion", 1)
                over = {**over, "guide_path": GUIDE.format(run=guide)}
            try:
                m = run_eval(ckpt, over, a.out / run.name / name(over), a)
            except subprocess.CalledProcessError as e:  # e.g. OOM on a shared GPU: skip, rerun later
                print(f"FAILED {run.name} {name(over)}: {e}", flush=True)
                continue
            rows.append({
                "run": run.name, "policy": ptype, "variant": name(over),
                "pc_success": m["pc_success"], "avg_max_reward": m["avg_max_reward"],
                "latency_ms": latency_ms(ckpt, over, a.device),
            })  # fmt: skip
            print(rows[-1], flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    md = ["| run | policy | variant | success % | coverage | latency ms |", "|---|---|---|---|---|---|"]
    md += [
        f"| {x['run']} | {x['policy']} | {x['variant']} | {x['pc_success']:.1f} | "
        f"{x['avg_max_reward']:.3f} | {x['latency_ms']:.1f} |"
        for x in rows
    ]
    (a.out / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), file=sys.stderr)


if __name__ == "__main__":
    main()
