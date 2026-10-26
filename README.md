# FlowPolicy

Diffusion Policy with the DDPM objective replaced by flow matching, built as a [LeRobot](https://github.com/huggingface/lerobot) 0.6.1 plugin and evaluated on PushT. The goal is Diffusion Policy's accuracy at 1–4 network passes per action chunk instead of 100, which means far lower latency.

![Diffusion 1 step vs Diffusion 100 steps vs Shortcut flow 1 step, same 4 PushT episodes (unselected)](results/demo/demo.gif)

## Results (full-size models, PushT)

Three training seeds, 200 evaluation episodes each (eval seed 100000, same initial states for every policy). Mean ± std over seeds. Latency is the median per action chunk at batch size 1 on an RTX 4090; methods with the same number of network passes cost the same, because they share one U-Net. Gaps under about 5 points are within seed-to-seed noise.

### Headline: success vs. network passes

| Passes (latency) | Diffusion (ε, standard) | Diffusion (x0) | Flow | Shortcut | Flow + Diffusion polish |
|---|---|---|---|---|---|
| 1 (~6 ms) | 0.0 | 47.2 ± 5.2 | 51.7 ± 2.5 | **55.0 ± 1.3** | – |
| 2 (~10 ms) | 48.7 ± 9.8 (DDPM) | 47.2 ± 8.6 | 56.0 ± 4.8 | 57.2 ± 5.5 | **63.2 ± 0.6** (1 + 1 DDPM) |
| 3 (~15 ms) | – | – | – | – | **65.3 ± 2.6** (1 + 2 DDIM) |
| 4 (~17–19 ms) | 61.7 ± 1.4 (DDIM) | 46.3 ± 3.6 | 60.5 ± 4.9 | 61.2 ± 3.8 | **66.3 ± 1.4** (1 + 3 DDIM) |
| 100 (~383 ms) | **67.8 ± 4.3** (DDPM) | 47.5 ± 2.2 | 68.5 ± 3.5 | 69.2 ± 1.8 | – |

- **Flow + Diffusion polish is the best few-pass method.** One flow pass makes the plan; it is re-noised slightly and Diffusion runs only its last k denoising steps (SDEdit). With 3 polish steps it reaches 66.3%, within 1.5 points of 100-step Diffusion at ~20× lower latency, and about 5 points above every plain 4-pass sampler. Flow's few-step failures are mostly near misses, which is what the low-noise Diffusion steps fix.
- **At 1 pass, Flow beats Diffusion.** Shortcut gets 55.0% vs 47.2% for x0-prediction Diffusion; standard ε-prediction Diffusion collapses to 0% at 1 step.
- **At 100 passes every method lands at 68–69%**, so the ceiling is the data and network, not the sampler. Our Diffusion baseline matches LeRobot's published `diffusion_pusht` (65.4% over 500 episodes).
- x0-prediction Diffusion stays at 45–50% at every step count.

### Main results

| Method | Passes | Success % | Latency per chunk |
|---|---|---|---|
| Diffusion Policy (baseline, ε-prediction, DDPM) | 100 | 67.8 ± 4.3 | ~383 ms |
| Diffusion Policy (x0-prediction, DDIM) | 1 | 47.2 ± 5.2 | ~6 ms |
| Flow | 1 | 51.7 ± 2.5 | ~6 ms |
| Shortcut (Flow + step-size conditioning) | 1 | **55.0 ± 1.3** | ~6 ms |
| Flow + 1 DDPM polish step | 2 | **63.2 ± 0.6** | ~11 ms |
| Flow + 3 DDIM polish steps | 4 | **66.3 ± 1.4** | ~19 ms |
| Shortcut + 100-step Diffusion plan every 2nd chunk | mixed | **68.7 ± 2.3** | ~195 ms avg |
| Shortcut + 100-step Diffusion plan every 4th chunk | mixed | 65.7 ± 1.3 | ~100 ms avg |
| ACT (50 episodes per seed) | 1 | 26.7 ± 6.1 | ~3 ms |
| BC-MLP (50 episodes per seed) | 1 | 26.0 ± 10.6 | ~2 ms |

Polish latencies (11/15/19/27 ms for 1/2/3/5 steps) were measured while another job shared the GPU. Shortcut + polish performs the same as Flow + polish (62–68% across 1–5 steps) with a larger spread between seeds.

### Tried, did not help

- **Sample-and-select** (Bidirectional Decoding style): 8 or 32 candidate plans in one batch, keeping the one that continues the previous plan or the medoid. 0 to +5 points at 1 pass, within noise, and no gain on top of the Diffusion polish.
- **Replanning more often** (every 4 actions instead of 8): about −10 points, even when the candidate that continues the previous plan is kept, so the loss is not from switching modes.
- Committing longer (12 actions), lower noise temperature, uneven step schedules, warm-starting from the previous plan: no gain.
- Logit-normal or Beta t-sampling during training: no gain; logit-normal collapses 1-pass success to 3%.
- **ReinFlow** (PPO fine-tuning of the 1-pass Shortcut policy with learned Gaussian noise on the Euler step, `scripts/reinflow.py`): about 15 settings of learning rate, noise level, batch size and critic input, up to 180 iterations (~9k training episodes) each. Training and held-out success stayed flat; the best held-out checkpoint scores 50.0% at 200 episodes vs 53.5% for its base model. With 250M weights, one Adam step at lr 1e-6 already moves the policy past the trust region, so PPO needs lr ~1e-7 and learns very slowly.
- **Multi-step ReinFlow** (`--steps K`: learned noise at every Euler step, as in the ReinFlow paper, so the early steps get credit for the mode choice): 2 and 4 passes, 3 seeds at 4 passes plus learning-rate and noise variants, 100–300 PPO iterations each. At 4 passes the best checkpoints score 62.3 ± 5.1% at 200 episodes vs 61.2 ± 3.8% for the untuned model (+7, +1 and −4.5 points per seed); at 2 passes 56.0 / 63.0% vs 57.5 / 62.5%. Held-out success stayed flat during training. Results in [`results/rl_k`](results/rl_k/results.md).
- Reflow (Flow distilled from its own 10-step samples): no better than plain Flow at 200 episodes (49.5 / 53.8 / 60.5% at 1 / 2 / 4 passes).

Full per-variant tables are in [`results/RESULTS.md`](results/RESULTS.md).

## Layout

```
src/lerobot_policy_flowpolicy/
  flow/     FlowConfig / FlowPolicy: flow matching on the Diffusion Policy U-Net, plus reflow,
            shortcut, warm start, step schedules, t-sampling, noise temperature, the Diffusion guide,
            Diffusion polish (refine_*) and sample-and-select (n_candidates)
  bcmlp/    MLP behaviour-cloning baseline
scripts/
  train.sh        train one policy (diffusion | flow | bcmlp | act)
  eval_sweep.py   success vs. passes plus latency; --refine / --select / --guided / ... variant sets
  aggregate.py    mean ± std over seeds from results.csv
  breakdown.py    success, coverage, gross vs near-miss failures, pooled from eval_info.json
  demo.sh         side-by-side demo video
  remote.sh       sync, run and pull on a GPU server
tests/test_policies.py
results/          eval_info.json per run and variant (videos not included)
```

## Usage

```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -e .
pytest tests

# Full-size Diffusion Policy PushT config, 200k steps
scripts/train.sh flow 1000 200000
scripts/train.sh diffusion 1000 200000
# Reflow student from a trained flow teacher
NAME=reflow scripts/train.sh flow 1000 20000 --policy.teacher_path=outputs/train/flow_s1000_200000/checkpoints/last/pretrained_model
# Shortcut fine-tune
NAME=shortcut scripts/train.sh flow 1000 40000 --policy.shortcut=true --policy.init_path=outputs/train/flow_s1000_200000/checkpoints/last/pretrained_model

python scripts/eval_sweep.py outputs/train/*_200000 --out results/full
python scripts/breakdown.py results/full
# Flow + 3 DDIM polish steps from the same-seed Diffusion checkpoint
python scripts/eval_sweep.py outputs/train/flow_s1000_200000 --refine --episodes 200 --out results/refine200
```

`SMALL=1` trains a 4× smaller U-Net with AMP for 6 GB GPUs. It is a trend check, not a reproduction.
