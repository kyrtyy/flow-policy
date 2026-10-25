# Results (PushT, full-size models, 50 episodes per training seed, eval seed 100000)

Pooled over training seeds from each run's `eval_info.json` with `python scripts/breakdown.py results/<dir> ...`. Failures split by final coverage: gross (< 0.6) vs near miss (>= 0.85). `diffx0` = Diffusion with x0 ("sample") prediction. `flowcont` = Flow trained 40k extra steps (control for shortcut). `guide4_1_w0.5` = 1-pass shortcut with a 100-step Diffusion plan every 4th chunk.

| results | policy | variant | seeds | success % | coverage | fails | gross <0.6 | near >=0.85 |
|---|---|---|---|---|---|---|---|---|
| full | act | default | 3 | 26.7 ± 6.1 | 0.710 | 110 | 56 | 45 |
| full | bcmlp | default | 1 | 38.0 ± 0.0 | 0.770 | 31 | 12 | 16 |
| full | diffusion | DDIM_1 | 3 | 0.0 ± 0.0 | 0.156 | 150 | 148 | 0 |
| full | diffusion | DDIM_10 | 3 | 56.0 ± 0.0 | 0.945 | 66 | 4 | 46 |
| full | diffusion | DDIM_100 | 3 | 58.0 ± 8.7 | 0.945 | 63 | 2 | 43 |
| full | diffusion | DDIM_2 | 3 | 47.3 ± 13.6 | 0.923 | 79 | 12 | 54 |
| full | diffusion | DDIM_4 | 3 | 55.3 ± 2.3 | 0.932 | 67 | 10 | 47 |
| full | diffusion | DDPM_1 | 3 | 0.0 ± 0.0 | 0.156 | 150 | 148 | 0 |
| full | diffusion | DDPM_10 | 3 | 56.0 ± 9.2 | 0.921 | 66 | 8 | 37 |
| full | diffusion | DDPM_100 | 3 | 60.7 ± 2.3 | 0.922 | 59 | 9 | 32 |
| full | diffusion | DDPM_2 | 3 | 44.0 ± 2.0 | 0.913 | 84 | 10 | 58 |
| full | diffusion | DDPM_4 | 3 | 54.7 ± 7.0 | 0.927 | 68 | 9 | 44 |
| full | flow | 1 | 3 | 41.3 ± 4.2 | 0.903 | 88 | 17 | 63 |
| full | flow | 10 | 3 | 57.3 ± 2.3 | 0.927 | 64 | 10 | 44 |
| full | flow | 100 | 3 | 60.7 ± 8.3 | 0.926 | 59 | 9 | 36 |
| full | flow | 2 | 3 | 46.7 ± 3.1 | 0.915 | 80 | 14 | 61 |
| full | flow | 4 | 3 | 55.3 ± 6.4 | 0.926 | 67 | 10 | 49 |
| full_bcmlp | bcmlp | default | 2 | 20.0 ± 2.8 | 0.649 | 80 | 39 | 30 |
| diffx0_full | diffx0 | DDIM_1 | 3 | 50.7 ± 4.2 | 0.881 | 74 | 18 | 40 |
| diffx0_full | diffx0 | DDIM_10 | 3 | 40.0 ± 5.3 | 0.903 | 90 | 12 | 58 |
| diffx0_full | diffx0 | DDIM_100 | 3 | 39.3 ± 4.6 | 0.876 | 91 | 14 | 50 |
| diffx0_full | diffx0 | DDIM_2 | 3 | 45.3 ± 2.3 | 0.881 | 82 | 15 | 47 |
| diffx0_full | diffx0 | DDIM_4 | 3 | 42.0 ± 9.2 | 0.890 | 87 | 12 | 50 |
| diffx0_full | diffx0 | DDPM_1 | 3 | 40.7 ± 6.4 | 0.908 | 89 | 13 | 59 |
| diffx0_full | diffx0 | DDPM_10 | 3 | 51.3 ± 5.8 | 0.898 | 73 | 13 | 41 |
| diffx0_full | diffx0 | DDPM_100 | 3 | 42.0 ± 4.0 | 0.896 | 87 | 13 | 52 |
| diffx0_full | diffx0 | DDPM_2 | 3 | 44.0 ± 7.2 | 0.912 | 84 | 12 | 55 |
| diffx0_full | diffx0 | DDPM_4 | 3 | 53.3 ± 5.0 | 0.902 | 70 | 14 | 39 |
| reflow_full | reflow | 1 | 3 | 47.3 ± 3.1 | 0.899 | 79 | 18 | 53 |
| reflow_full | reflow | 10 | 3 | 58.7 ± 11.0 | 0.928 | 62 | 10 | 41 |
| reflow_full | reflow | 100 | 3 | 55.3 ± 7.0 | 0.918 | 67 | 13 | 44 |
| reflow_full | reflow | 2 | 3 | 57.3 ± 3.1 | 0.919 | 64 | 11 | 39 |
| reflow_full | reflow | 4 | 3 | 55.3 ± 3.1 | 0.939 | 67 | 5 | 49 |
| shortcut_full | shortcut | 1 | 3 | 55.3 ± 5.0 | 0.917 | 67 | 11 | 41 |
| shortcut_full | shortcut | 10 | 3 | 66.0 ± 3.5 | 0.939 | 51 | 8 | 33 |
| shortcut_full | shortcut | 100 | 3 | 62.7 ± 4.2 | 0.917 | 56 | 12 | 32 |
| shortcut_full | shortcut | 2 | 3 | 50.7 ± 3.1 | 0.907 | 74 | 12 | 49 |
| shortcut_full | shortcut | 4 | 3 | 48.7 ± 2.3 | 0.922 | 77 | 9 | 55 |
| flowcont_full | flowcont | 1 | 3 | 47.3 ± 2.3 | 0.924 | 79 | 11 | 59 |
| flowcont_full | flowcont | 10 | 3 | 56.0 ± 4.0 | 0.926 | 66 | 11 | 48 |
| flowcont_full | flowcont | 100 | 3 | 63.3 ± 6.1 | 0.949 | 55 | 6 | 40 |
| flowcont_full | flowcont | 2 | 3 | 50.7 ± 4.2 | 0.909 | 74 | 14 | 51 |
| flowcont_full | flowcont | 4 | 3 | 53.3 ± 1.2 | 0.918 | 70 | 13 | 51 |
| explore_shortcut | shortcut | 1_a12 | 3 | 42.7 ± 13.6 | 0.867 | 86 | 22 | 51 |
| explore_shortcut | shortcut | 1_a12_n0.8 | 3 | 49.3 ± 6.4 | 0.887 | 76 | 18 | 45 |
| explore_shortcut | shortcut | 1_a4 | 3 | 38.0 ± 6.0 | 0.917 | 93 | 11 | 75 |
| explore_shortcut | shortcut | 1_a4_n0.5 | 3 | 26.7 ± 11.7 | 0.874 | 110 | 13 | 75 |
| explore_shortcut | shortcut | 1_a4_w0.3 | 3 | 35.3 ± 8.1 | 0.907 | 97 | 10 | 77 |
| explore_shortcut | shortcut | 1_n0.5 | 3 | 45.3 ± 6.4 | 0.905 | 82 | 15 | 60 |
| explore_shortcut | shortcut | 1_n0.8 | 3 | 56.7 ± 5.8 | 0.916 | 65 | 13 | 42 |
| explore_shortcut | shortcut | 2_a12 | 3 | 46.7 ± 12.2 | 0.916 | 80 | 9 | 55 |
| explore_shortcut | shortcut | 2_a4 | 3 | 32.0 ± 5.3 | 0.911 | 102 | 8 | 80 |
| explore_shortcut | shortcut | guide4_1_w0.5 | 3 | 66.7 ± 3.1 | 0.948 | 50 | 7 | 35 |
