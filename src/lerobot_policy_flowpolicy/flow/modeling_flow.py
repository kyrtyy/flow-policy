import torch
from safetensors.torch import load_file
from torch import Tensor

from torch import nn

from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel, DiffusionPolicy, DiffusionSinusoidalPosEmb
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters
from lerobot.utils.constants import ACTION, OBS_STATE

from ..common import masked_mse
from .configuration_flow import FlowConfig


class FlowModel(DiffusionModel):
    """Rectified flow: x_t = (1 - t) * noise + t * action, and the U-Net predicts v = action - noise.

    t in [0, 1] is scaled by `num_train_timesteps` before the U-Net's sinusoidal embedding, so the
    embedding sees the same range as in Diffusion Policy.
    """

    def __init__(self, config: FlowConfig):
        super().__init__(config)
        if config.shortcut:
            dim = config.diffusion_step_embed_dim
            self.d_encoder = nn.Sequential(
                DiffusionSinusoidalPosEmb(dim), nn.Linear(dim, dim * 4), nn.Mish(), nn.Linear(dim * 4, dim)
            )
            # Zero output at init: a shortcut model fine-tuned from a flow checkpoint starts out identical to it.
            nn.init.zeros_(self.d_encoder[-1].weight)
            nn.init.zeros_(self.d_encoder[-1].bias)
            # Add the step-size embedding to the U-Net's timestep embedding (no change to the U-Net itself).
            self.unet.diffusion_step_encoder.register_forward_hook(
                lambda mod, inp, out: out + self.d_encoder(self._d * self.config.num_train_timesteps)
            )

    def _unet(self, x: Tensor, t: Tensor, global_cond: Tensor | None, d: Tensor | None = None) -> Tensor:
        if self.config.shortcut:
            self._d = torch.zeros_like(t) if d is None else d
        return self.unet(x, t * self.config.num_train_timesteps, global_cond=global_cond)

    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)
        x = (
            noise
            if noise is not None
            else torch.randn(
                (batch_size, self.config.horizon, self.config.action_feature.shape[0]),
                dtype=dtype,
                device=device,
                generator=generator,
            )
            * self.config.noise_scale
        )
        cfg = self.config
        ts = cfg.step_schedule or [i / self.num_inference_steps for i in range(self.num_inference_steps + 1)]
        ts = torch.tensor(ts, dtype=x.dtype, device=x.device)
        prev = getattr(self, "_prev_plan", None)
        if cfg.warm_start > 0 and prev is not None and prev.shape == x.shape:
            # The horizon has advanced by n_action_steps since the last plan; repeat its last action.
            shifted = torch.cat([prev[:, cfg.n_action_steps :], prev[:, -1:].expand(-1, cfg.n_action_steps, -1)], 1)
            s = cfg.warm_start
            x = (1 - s) * x + s * shifted
            ts = s + (1 - s) * ts
        for t0, t1 in zip(ts[:-1], ts[1:], strict=True):
            t = t0.expand(x.shape[0])
            x = x + (t1 - t0) * self._unet(x, t, global_cond, d=(t1 - t0).expand(x.shape[0]))
        # Actions are MIN_MAX-normalised to [-1, 1]; Diffusion Policy clips to the same range.
        x = x.clamp(-1, 1) if cfg.clip_sample else x
        self._prev_plan = x
        return x

    def _sample_t(self, n: int, like: Tensor) -> Tensor:
        kind = self.config.t_sampling
        if kind == "uniform":
            return torch.rand(n, device=like.device, dtype=like.dtype)
        if kind == "logit_normal":
            return torch.sigmoid(torch.randn(n, device=like.device, dtype=like.dtype))
        if kind == "beta":  # pi0: (s - t) / s ~ Beta(1.5, 1), s = 0.999
            u = torch.distributions.Beta(1.5, 1.0).sample((n,)).to(like.device, like.dtype)
            return 0.999 * (1 - u)
        raise ValueError(f"unknown t_sampling {kind!r}")

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        global_cond = self._prepare_global_conditioning(batch)
        x1 = batch[ACTION]
        x0 = torch.randn_like(x1)
        if self.config.teacher_path:
            x1 = self._teacher_sample(batch, x0)
        t = self._sample_t(x1.shape[0], x1)
        xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        pred = self._unet(xt, t, global_cond)
        loss = masked_mse(pred, x1 - x0, batch, self.config.do_mask_loss_for_padding)
        if self.config.shortcut:
            loss = (1 - self.config.shortcut_frac) * loss + self.config.shortcut_frac * self._shortcut_loss(
                x0, x1, global_cond
            )
        return loss

    def _shortcut_loss(self, x0: Tensor, x1: Tensor, global_cond: Tensor) -> Tensor:
        """One step of size 2d must match two consecutive steps of size d (targets without grad)."""
        n, m = x0.shape[0], self.config.shortcut_min_steps
        levels = int(torch.log2(torch.tensor(m)).item())
        d = 2.0 ** -torch.randint(1, levels + 1, (n,), device=x0.device).to(x0.dtype)  # 1/2 ... 1/m
        t = torch.rand(n, device=x0.device, dtype=x0.dtype) * (1 - 2 * d)
        xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        # The finest level has no smaller shortcut to compose: its half steps are the instantaneous flow (d=0).
        d_half = torch.where(d <= 1.0 / m, torch.zeros_like(d), d)
        with torch.no_grad():
            s1 = self._unet(xt, t, global_cond, d=d_half)
            s2 = self._unet(xt + d[:, None, None] * s1, t + d, global_cond, d=d_half)
            target = (s1 + s2) / 2
        pred = self._unet(xt, t, global_cond, d=2 * d)
        return nn.functional.mse_loss(pred, target)


    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        cfg = self.config
        k = getattr(self, "_chunk", 0)
        self._chunk = k + 1
        if not cfg.guide_path or (k % cfg.guide_every if cfg.guide_every else k):
            return super().generate_actions(batch, noise)
        if getattr(self, "_guide", None) is None:
            from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

            g = DiffusionPolicy.from_pretrained(cfg.guide_path).diffusion.to(get_device_from_parameters(self)).eval()
            g.num_inference_steps = cfg.guide_steps
            self.__dict__["_guide"] = g  # not a submodule: never trained or saved
        g = self._guide
        plan = g.conditional_sample(batch[OBS_STATE].shape[0], global_cond=g._prepare_global_conditioning(batch))
        self._prev_plan = plan  # flow chunks after this warm-start from the diffusion plan
        start = cfg.n_obs_steps - 1
        return plan[:, start : start + cfg.n_action_steps]

    @torch.no_grad()
    def _teacher_sample(self, batch: dict[str, Tensor], x0: Tensor) -> Tensor:
        if getattr(self, "_teacher", None) is None:
            # Plain attribute via __dict__, so the teacher is neither trained, moved by .to(), nor saved.
            t = FlowPolicy.from_pretrained(self.config.teacher_path).diffusion.to(x0.device).eval()
            t.num_inference_steps = self.config.teacher_steps
            self.__dict__["_teacher"] = t
        t = self._teacher
        return t.conditional_sample(x0.shape[0], global_cond=t._prepare_global_conditioning(batch), noise=x0)


class FlowPolicy(DiffusionPolicy):
    """Diffusion Policy with the DDPM objective swapped for flow matching. Rollout logic is inherited."""

    config_class = FlowConfig
    name = "flow"

    def __init__(self, config: FlowConfig, **kwargs):
        # Skip DiffusionPolicy.__init__ so we don't build a DiffusionModel we'd throw away.
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self._queues = None
        self.diffusion = FlowModel(config)  # attribute name kept so inherited methods work
        if config.init_path:
            missing, unexpected = self.load_state_dict(
                load_file(f"{config.init_path}/model.safetensors", device="cpu"), strict=False
            )
            assert not unexpected and all("d_encoder" in k for k in missing), (missing, unexpected)
        if config.teacher_path:
            # Reflow starts from the teacher's weights; from_pretrained of the student overwrites them.
            # ponytail: needs teacher_path to exist when loading the student too; drop it from the saved config if that bites.
            # CPU load: no GPU copy of the teacher just to initialise (it would OOM a 6 GB card at eval time).
            self.load_state_dict(load_file(f"{config.teacher_path}/model.safetensors", device="cpu"))
        self.reset()

    def reset(self):
        super().reset()
        self.diffusion._prev_plan = None  # new episode: no plan to warm-start from
        self.diffusion._chunk = 0
