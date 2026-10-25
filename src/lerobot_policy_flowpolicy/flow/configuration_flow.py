from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


@PreTrainedConfig.register_subclass("flow")
@dataclass
class FlowConfig(DiffusionConfig):
    """Diffusion Policy's architecture and hyperparameters, trained with flow matching instead of DDPM.

    The DDPM noise-scheduler fields are inherited but unused. `num_inference_steps` is the number of
    Euler steps from noise (t=0) to action (t=1).
    """

    num_inference_steps: int | None = 10
    # Explicit Euler time grid from 0 to 1, e.g. [0, 0.1, 1]: small early steps commit to a mode,
    # one big step finishes it. Overrides num_inference_steps when set.
    step_schedule: list[float] | None = None
    # > 0: start each chunk at t=warm_start from the previous chunk's plan (shifted by n_action_steps)
    # mixed with noise, instead of from pure noise at t=0, so the policy keeps its earlier decision.
    warm_start: float = 0.0
    # Scale of the initial noise (sampling temperature); < 1 pulls few-step samples toward the mean action.
    noise_scale: float = 1.0
    # Reflow distillation: train on (noise, teacher's multi-step sample from that noise) pairs instead of
    # (noise, demo action), so paths are straight and one Euler step lands on a single mode.
    # Training distribution of t: "uniform", "logit_normal" (SD3: mass in the middle) or "beta" (pi0:
    # Beta(1.5, 1) toward t=0, the noisy end where the sampler's first step makes the decision).
    t_sampling: str = "uniform"
    # Shortcut model (Frans et al. 2024): the U-Net also sees the step size d, and a `shortcut_frac` share of
    # each batch trains self-consistency (one 2d step == two d steps), so few-step sampling stays accurate.
    shortcut: bool = False
    shortcut_frac: float = 0.25
    shortcut_min_steps: int = 128  # finest step size is 1 / shortcut_min_steps
    # Initialise weights from this checkpoint (missing keys, e.g. the new step-size encoder, keep their init).
    init_path: str | None = None
    teacher_path: str | None = None
    teacher_steps: int = 10
    # Hybrid "diffusion decides, flow executes": every `guide_every` chunks (0 = only the first chunk of an
    # episode), the chunk comes from this Diffusion Policy checkpoint with `guide_steps` DDPM steps. Chunks
    # in between come from flow, warm-started from the latest plan (set warm_start > 0).
    guide_path: str | None = None
    guide_every: int = 0
    guide_steps: int = 100
