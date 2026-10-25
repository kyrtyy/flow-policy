# FlowPolicy

Diffusion Policy with the DDPM objective replaced by flow matching, built as a [LeRobot](https://github.com/huggingface/lerobot) 0.6.1 plugin and evaluated on PushT. The goal is Diffusion Policy's accuracy at 1–4 network passes per action chunk instead of 100, which means far lower latency.

![Diffusion 1 step vs Diffusion 100 steps vs Shortcut flow 1 step, same 4 PushT episodes (unselected)](results/demo/demo.gif)

## Results (full-size models, PushT)

Three training seeds, 50 evaluation episodes each (eval seed 100000, same initial states for every policy). Mean ± std over seeds. Latency per action chunk was measured on a shared RTX 4090, so treat it as approximate. Policies with the same number of passes cost the same, because they share one U-Net. Re-running an identical configuration on another GPU moved success by up to 4 points, so gaps under about 5 points are noise.

### Headline

| Passes | Diffusion (ε-prediction, standard) | Diffusion (x0-prediction) | Flow | Best flow variant |
|---|---|---|---|---|
| 1 (~13 ms) | 0.0 | 50.7 ± 4.2 (DDIM) | 41.3 ± 4.2 | **55.3 ± 5.0** (shortcut) |
| 2 (~21 ms) | 47.3 ± 13.6 (DDIM) | 45.3 ± 2.3 (DDIM) | 46.7 ± 3.1 | **57.3 ± 3.1** (reflow) |
| 4 (~39 ms) | 55.3 ± 2.3 (DDIM) | 53.3 ± 5.0 (DDPM) | 55.3 ± 6.4 | 55.3 ± 3.1 (reflow) |
| 100 (~830 ms) | **60.7 ± 2.3** (DDPM) | 42.0 ± 4.0 (DDPM) | 60.7 ± 8.3 | 62.7 ± 4.2 (shortcut) |

- **The 2-pass result is the clearest win.** Reflow gets 57.3% at about 21 ms per chunk. That is +10 points over either Diffusion variant at the same cost, and within 3.4 points of 100-pass Diffusion at about 1/40 of the latency.
- **At 1 pass the comparison is closer than it first looked.** Standard Diffusion's 0% is an artifact of ε-prediction: an x0-prediction Diffusion model gets 50.7% at 1 pass. Shortcut (55.3%) leads it by less than the noise level, so "Flow beats Diffusion at 1 pass" is not established.
- x0-prediction Diffusion gets *worse* with more steps (42% at 100). At 1–2 steps it is good; at many steps it is the weakest variant.

### Main results

| Method | Passes | Success % | Latency per chunk |
|---|---|---|---|
| Diffusion Policy (baseline, ε-prediction, DDPM) | 100 | 60.7 ± 2.3 | ~830 ms |
| Diffusion Policy (x0-prediction, DDIM) | 1 | 50.7 ± 4.2 | ~13 ms |
| Flow | 1 | 41.3 ± 4.2 | ~13 ms |
| Reflow (Flow distilled from its own 10-step samples) | 2 | **57.3 ± 3.1** | ~21 ms |
| Shortcut (Flow + step-size conditioning) | 1 | **55.3 ± 5.0** | ~12 ms |
| Shortcut 1-pass + Diffusion plan every 4th chunk | mixed | **66.7 ± 3.1** | ~215 ms avg |
| ACT | 1 | 26.7 ± 6.1 | ~3 ms |
| BC-MLP | 1 | 26.0 ± 10.6 | ~2 ms |

- **Reproduction check:** LeRobot's published `diffusion_pusht` reports 65.4% over 500 episodes; our baseline gets 60.7%.
- **Shortcut control:** plain Flow trained the same 40k extra steps gets 47.3% at 1 pass, so the shortcut objective itself adds about 8 points.
- **Best overall:** shortcut with one 100-step Diffusion plan every 4th chunk (66.7%) beats 100-pass Diffusion at about a quarter of the average latency.

### Tried, did not help

- Uneven step schedules, warm-starting from the previous plan, lower noise temperature: no gain.
- Replanning more often (every 4 actions instead of 8): about −20 points. Committing longer (12 actions) also hurts.
- Logit-normal or Beta t-sampling during training: no gain; logit-normal collapses 1-pass success to 3%.

Full per-variant tables for the results above are in [`results/RESULTS.md`](results/RESULTS.md).

### Open

- Final evaluation at 200 episodes per seed, with latency measured on an idle GPU, to shrink the ±5-point noise.

## Layout

```
src/lerobot_policy_flowpolicy/
  flow/     FlowConfig / FlowPolicy: flow matching on the Diffusion Policy U-Net, plus reflow,
            shortcut, warm start, step schedules, t-sampling, noise temperature and the Diffusion guide
  bcmlp/    MLP behaviour-cloning baseline
scripts/
  train.sh        train one policy (diffusion | flow | bcmlp | act)
  eval_sweep.py   success vs. passes plus latency; --hybrid / --guided / --combo / --explore variants
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
```

`SMALL=1` trains a 4× smaller U-Net with AMP for 6 GB GPUs. It is a trend check, not a reproduction.
