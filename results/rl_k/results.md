Multi-step ReinFlow (`scripts/reinflow.py --steps K`): learned noise at every Euler step, PPO from the Shortcut checkpoint.
200 episodes, eval seed 100000. Base = the same Shortcut checkpoint at K passes (results/final). Checkpoint = best 96-episode held-out check.

| run | passes | iter | success % | base % | coverage |
|---|---|---|---|---|---|
| k4b_s1000 | 4 | 60 | 64.5 | 57.5 | 0.938 |
| k4b_s1001 | 4 | 100 | 66.0 | 65.0 | 0.955 |
| k4b_s1002 | 4 | 90 | 56.5 | 61.0 | 0.948 |
| k4b_lr1e-6_s1000 | 4 | 30 | 65.0 | 57.5 | 0.913 |
| k4b_e03_s1000 | 4 | 70 | 62.5 | 57.5 | 0.939 |
| k4b_e01_s1000 | 4 | 60 | 64.5 | 57.5 | 0.929 |
| k2_s1000 | 2 | 120 | 56.0 | 57.5 | 0.917 |
| k2_s1001 | 2 | 260 | 63.0 | 62.5 | 0.945 |

Seeds 1000-1002 at 4 passes: 62.3 ± 5.1 vs 61.2 ± 3.8 for the base.
