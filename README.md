# FlowPolicy

Diffusion Policy with the DDPM objective replaced by flow matching, built as a [LeRobot](https://github.com/huggingface/lerobot) 0.6.1 plugin and evaluated on PushT. The goal is Diffusion Policy's accuracy at 1–4 network passes per action chunk instead of 100, which means far lower latency.

![Diffusion 1 step vs Diffusion 100 steps vs Flow 1 step](results/demo/demo.gif)

## Results (full-size models, PushT)

Three training seeds, 50 evaluation episodes each (eval seed 100000, same initial states for every policy). Mean ± std over seeds. Latency per action chunk was measured on a shared RTX 4090, so treat it as approximate. Policies with the same number of passes cost the same, because they share one U-Net.

| Method | Passes | Success % | Coverage | Latency |
|---|---|---|---|---|
| Diffusion, DDPM (baseline) | 100 | 60.7 ± 2.3 | 0.922 | ~830 ms |
| Diffusion, DDPM / DDIM | 4 | 54.7 / 55.3 | 0.93 | ~38 ms |
| Diffusion, DDPM / DDIM | 2 | 44.0 / 47.3 | 0.91–0.92 | ~21 ms |
| Diffusion, DDPM / DDIM | 1 | 0.0 | 0.156 | ~13 ms |
| Flow | 1 | 41.3 ± 4.2 | 0.903 | ~13 ms |
| Flow | 2 | 46.7 ± 3.1 | 0.915 | ~21 ms |
| Flow | 4 | 55.3 ± 6.4 | 0.926 | ~39 ms |
| Flow | 100 | 60.7 ± 8.3 | 0.926 | ~727 ms |
| Reflow (distilled from Flow, 10-step teacher) | 2 | **57.3 ± 3.1** | 0.919 | ~21 ms |
| Shortcut (Flow + step-size conditioning, 2 of 3 seeds so far) | 1 | **55.0 ± 7.1** | 0.903 | ~12 ms |
| Flow 1-step + Diffusion plan every 4th chunk | mixed | 60.7 ± 8.1 | 0.952 | — |
| ACT | 1 | 26.7 ± 6.1 | 0.710 | — |
| BC-MLP (1 of 3 seeds so far) | 1 | 38.0 | 0.770 | — |

Reproduction check: LeRobot's published `diffusion_pusht` reports 65.4% success and 0.955 coverage over 500 episodes. Our Diffusion baseline gets 60.7% and 0.922.

What did not help (full results in [`results/RESULTS.md`](results/RESULTS.md)):
- Uneven step schedules such as `[0, 0.1, 1]`, and warm-starting from the previous plan: about 50–52% at 1–2 passes.
- Logit-normal t-sampling (small model): 1-pass success fell to 2.7%. It barely trains near t=0, which is where a 1-pass sampler starts.
- Combining reflow with warm start or the Diffusion guide: no gain over reflow alone.

At full size, most 1-pass failures are near misses (final coverage ≥ 0.85), not wrong strategy. Those are the failures that reflow and shortcut reduce.

### Open

- Diffusion's 0% at 1 pass may be an artifact of ε-prediction. An x0-prediction ("sample") Diffusion baseline is training. Until it is evaluated, "Flow beats Diffusion at 1 pass" is not settled. The 2-pass claim (reflow 57% vs Diffusion 44–47%) is the safer headline.
- Remaining seeds still to finish: shortcut s1002, BC-MLP s1000/s1002, and beta t-sampling s1002.
- Final evaluation at 200 episodes per seed, with latency measured on an idle GPU.

## Layout

```
src/lerobot_policy_flowpolicy/
  flow/     FlowConfig / FlowPolicy: flow matching on the Diffusion Policy U-Net, plus reflow,
            shortcut, warm start, step schedules, t-sampling and the Diffusion guide
  bcmlp/    MLP behaviour-cloning baseline
scripts/
  train.sh        train one policy (diffusion | flow | bcmlp | act)
  eval_sweep.py   success vs. passes plus latency; --hybrid / --guided / --combo variants
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
