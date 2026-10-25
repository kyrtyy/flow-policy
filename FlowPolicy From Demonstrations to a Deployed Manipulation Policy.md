# FlowPolicy: From Demonstrations to a Deployed Manipulation Policy

Oct 21, 2026 · @kyrtyy

## The idea

Train a robot manipulation policy in MuJoCo simulation three ways: from demonstrations (imitation learning), from offline data and then with online interaction (offline-to-online RL), and with a safety constraint. Then compress the best policy and deploy it with TensorRT under a real-time control budget.

The centrepiece is a **flow-matching policy**. It applies the FLAME Lab result to robotics: replacing diffusion with flow matching to get comparable accuracy with far fewer sampling steps. For a robot that means lower control latency, and it makes this a personal research story rather than a followed tutorial.

## Why this project

It reuses existing strengths (generative models, reproduction against published numbers, evaluation harnesses, ablations, quantization and latency) and covers most of the gaps in one repo.

| JD requirement | Where it's covered |
| --- | --- |
| MuJoCo, sim training environments | Every phase |
| Imitation learning (behaviour cloning, DAgger) | Phase 1 |
| Transformers for sequential decisions (ACT) | Phase 1 |
| Diffusion-based policies | Phase 1 (Diffusion Policy baseline) |
| Ablations, robustness | Phase 2 |
| PPO/SAC, reward design, curriculum | Phase 3 |
| Offline RL, offline-to-online (IQL) | Phase 3 |
| Safe RL, constrained optimisation | Phase 4 |
| Edge deployment, quantisation | Phase 5 |
| VLA fine-tuning | Phase 6 (stretch, SmolVLA) |
| Experiment tracking, reproducibility | Throughout (W&B, seeds, tests) |

## Stack

- **Base:** Hugging Face LeRobot. It ships PyTorch implementations of ACT, Diffusion Policy and TD-MPC, plus small VLAs like SmolVLA, and MuJoCo sim environments (ALOHA transfer-cube and insertion, PushT) with human demonstration datasets on the Hub.
- **Version check:** APIs change between releases, so check the current docs and pin versions on day 1.
- **Own RL code:** Write SAC/PPO and IQL yourself in a separate module, not copied from LeRobot. Writing them is what proves you know the algorithms.
- **RL environment:** Use the state-based version of the same task if one exists; otherwise Gymnasium-Robotics FetchPush (MuJoCo), which has Minari offline datasets.
- **Hardware:** One RTX 4090 covers everything except the VLA stretch, which is still feasible at SmolVLA's size.

## Plan (about 4 weeks part-time)

### Phase 0: setup and a reproduction gate (2–3 days)

- [ ] Install LeRobot and MuJoCo. Train the stock Diffusion Policy on PushT and the stock ACT on ALOHA transfer-cube; check you land near the success rates LeRobot or the original papers report.
- [ ] Build the evaluation harness first: success rate over 50 episodes × 3 training seeds on fixed eval seeds, reported as mean ± std.
- [ ] Add a random policy and the demonstration or scripted policy as lower and upper bounds.
- [ ] Add pytest tests for the env wrappers and metrics.

This gate makes every later number defensible in an interview.

### Phase 1: imitation learning and the flow-matching policy (week 1)

- [ ] Train four policies on the same task and data: plain behaviour cloning (MLP) and ACT as simple baselines, Diffusion Policy as the published baseline, and the flow-matching policy (Diffusion Policy's architecture with the DDPM objective swapped for flow matching).
- [ ] Headline experiment: success rate against sampling steps (1, 2, 4, 10, 100), plus inference latency per action chunk. Diffusion degrades badly at few steps; the question is whether flow matching holds up.
- [ ] Optional: DAgger on ALOHA, with the sim's scripted expert as labeller, to show why plain BC suffers from compounding errors.

### Phase 2: robustness and ablations (3–4 days)

- [ ] Stress-test every policy under observation noise, shifted initial object poses, and injected action latency of 0–100 ms (links to the teleoperation-latency project).
- [ ] Ablate action chunk size, observation history length, and number of demonstrations (10/25/50/100%). The demo-count curve addresses the JD's "rapid adaptation with minimal data".
- [ ] Keep the negative results.

### Phase 3: RL and offline-to-online fine-tuning (week 2)

- [ ] Write SAC (and PPO if time allows) on the state-based task. Compare sparse success reward against a shaped one, and add a curriculum that widens the initial object-pose range as success improves. Plot learning curves per variant.
- [ ] Offline RL: train IQL on the demonstration data alone and compare it with BC.
- [ ] Offline-to-online: fine-tune the IQL policy with online interaction and compare sample efficiency against SAC from scratch (environment steps to reach 80% success). This is the key result of the phase.

### Phase 4: safe RL (3–4 days)

- [ ] Add a constraint, such as an end-effector keep-out zone or a velocity limit, with a cost signal.
- [ ] Implement Lagrangian SAC (or PPO): a learned multiplier that trades return against constraint cost.
- [ ] Report constraint violation rate against success rate, compared with unconstrained SAC and with reward-penalty shaping.

### Phase 5: deployment (3–4 days)

- [ ] Export the best imitation policy and the best RL policy to ONNX, then build TensorRT engines in FP16 and INT8.
- [ ] Measure latency, model size, and closed-loop success rate after quantization. Accuracy after quantization matters more than speed alone.
- [ ] Wrap each policy in a fixed-rate control loop (for example 50 Hz) and report whether it meets the deadline.
- [ ] State that this ran on a desktop GPU, not a Jetson.

### Phase 6: stretch, VLA fine-tuning (only if time allows)

- [ ] Fine-tune SmolVLA on the same task's demonstrations and compare it with ACT and the flow policy on success rate, data efficiency and latency. OpenVLA and π0 are 7B-class, so don't start there.

### Phase 7: write-up (2–3 days)

- [ ] README with one results table across all methods, the success-vs-sampling-steps plot, the offline-to-online sample-efficiency plot, the safety trade-off plot, and rollout GIFs of successes and failures.
- [ ] Public W&B report.
- [ ] Blog post, for online presence and documented research.

## If time is short

The 2-week version is Phases 0, 1, 2 and 5: the flow-matching policy, the baselines, robustness tests and TensorRT deployment. That alone covers imitation learning, transformer policies, diffusion and flow-based policies, ablations and quantised deployment. Add RL afterwards.

## CV bullets to write once measured

Fill in the bracketed parts with measured numbers; none go on the CV before they exist.

- Built a flow-matching visuomotor policy for MuJoCo manipulation that matched Diffusion Policy's success rate (\[X\]% vs \[Y\]%) with \[N\]× fewer sampling steps, cutting inference latency from \[a\] to \[b\] ms. Benchmarked against BC and ACT over 3 seeds × 50 episodes.
- Implemented SAC and IQL from scratch. Offline-to-online fine-tuning reached 80% success in \[N\]× fewer environment steps than SAC from scratch. A Lagrangian constraint cut safety violations by \[X\]% at a \[Y\]-point success cost.
- Deployed policies with ONNX and TensorRT INT8 at \[X\] ms per action (\[Y\]× faster), with a \[Z\]-point success drop, meeting a 50 Hz control budget under up to 100 ms of injected latency.

Start with the Phase 0 reproduction gate. If the stock Diffusion Policy doesn't reproduce, fix it before building on it.
